"""H7-c self-play generator: TB-V3 Resolve / fast5 against itself with candidate models (docs/h7-c1.md).

Both sides are the frozen TB-V3 engine and fast5 runtime with the same models, built by the candidate factory
``analysis.tb_v3_net.make_tb_v3_net_agent`` (cycle 1: the frozen H3 / H6, recorded as ``frozen_models``). One agent
per colour per game (E4-v2's cooldown is per agent), seeded from the game seed.

Seeds (fixed before any game, docs/h6-value.md H7-c1 approval): ``--run-seed`` is a name space in 8800-8999, never a
benchmark / evaluation seed (8800 is the throughput measurement's, so training runs use 8801-8999); game ``i`` of cycle ``c`` has the seed ``derive_seed(run_seed, 'h7c', c, i)`` and the
opening seed ``derive_seed(run_seed, 'h7c-opening', c, i)`` (``derive_seed`` of the benchmark: SHA-256). Every seed of
the run is checked for collisions; each record keeps the run seed, cycle, index and both derived seeds.

Opening: the benchmark's generator (black's centre, 2 random moves within radius 2). Then the first 4 moves are
sampled by ``hybrid.h7c_targets.sample_move`` (proven wins played, proven losses excluded, T = 1; no distribution:
the engine's move, logged), and the rest are the engine's own moves.

Per move the record keeps the route, the engine's and the played move, the sampling event, the raw root visits
(``[row, col, visits, mean value]``), the proven-loss candidates and the policy target of
``hybrid.h7c_targets.policy_target``. Per game: result, provenance (engine hash, fast5 runtime hash of the value
device, the four model SHA-256 and their hash, code commit and dirty flag) and the sampling / target formats.

    python -u scripts/h7c_selfplay.py --cycle 1 --run-seed 8801 --games 1000 --workers 10 \\
        --policy-checkpoint runs/h3_policy_64x4/best.pt --value-checkpoint runs/h6_value_head_v3/best.pt \\
        --device cuda --jsonl runs/h7c/c1/selfplay_8801.jsonl --output runs/h7c/c1/selfplay_8801.json

Resume: rerun the same command; finished game indices in ``--jsonl`` are skipped, and a file of another run
(seed, cycle, models, engine, runtime, sampling) is refused. A dirty tree is refused (``--allow-dirty`` only for
tests). Progress goes to stdout about once a minute.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7c_targets as targets  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, WHITE, Game  # noqa: E402

FORMAT = 'h7c-selfplay-v1'
RUN_SEEDS = range(8801, 9000)   # training runs
MEASURE_SEED = 8800              # worker reproducibility / throughput games only (scripts/h7c_throughput.py)
OPENING = {'random_plies': 2, 'radius': 2}
RUNTIME = 'fast5'
_WORKER: dict = {}


def game_seed(run_seed: int, cycle: int, index: int) -> int:
    from run_mcts_v8_benchmark import derive_seed

    return derive_seed(run_seed, 'h7c', cycle, index)


def opening_seed(run_seed: int, cycle: int, index: int) -> int:
    from run_mcts_v8_benchmark import derive_seed

    return derive_seed(run_seed, 'h7c-opening', cycle, index)


def check_seeds(run_seed: int, cycle: int, indices, *, measurement: bool = False) -> None:
    from hybrid.h6_labels import check_reserved_seed

    if measurement != (run_seed == MEASURE_SEED) or (not measurement and run_seed not in RUN_SEEDS):
        raise ValueError(f'--run-seed {run_seed}: H7-c training runs use {RUN_SEEDS.start}-{RUN_SEEDS.stop - 1}; '
                         f'{MEASURE_SEED} is for the throughput measurement only')
    check_reserved_seed(run_seed)
    if cycle < 1:
        raise ValueError('--cycle starts at 1')
    seeds = [s for i in indices for s in (game_seed(run_seed, cycle, i), opening_seed(run_seed, cycle, i))]
    if len(set(seeds)) != len(seeds):
        raise ValueError('derived seeds collide; choose another run seed')


def run_identity(cycle: int, run_seed: int, models: dict, device: str) -> dict:
    """What every game of one run shares; a resumed file must match it."""
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net

    return {'format': FORMAT, 'cycle': cycle, 'run_seed': run_seed, 'engine_id': tb_v3.ENGINE_ID,
            'engine_hash': tb_v3.ENGINE_HASH, 'runtime': RUNTIME, 'runtime_hash': tb_v3_fast5.runtime_hash(device),
            'value_device': device.split(':')[0], 'models': models, 'model_hash': tb_v3_net.model_hash(models),
            'frozen_models': models == tb_v3.MODELS, 'opening_rule': OPENING,
            'sampling': {'format': targets.SAMPLING_FORMAT, 'plies': targets.SAMPLE_PLIES,
                         'temperature': targets.TEMPERATURE},
            'target_format': targets.TARGET_FORMAT}


def _init_worker(policy_checkpoint: str, value_checkpoint: str, device: str) -> None:
    from analysis import tb_v2_runtime, tb_v3_fast5, tb_v3_net
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    tb_v2_runtime.install()
    tb_v3_fast5.install()
    policy = RootPolicy(policy_checkpoint, threads=1)
    leaf = ValueLeaf(value_checkpoint, device=device, threads=1, expected_sha256=None)
    models = tb_v3_net.check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    _WORKER.update(policy=policy, leaf=leaf, models=models, policy_checkpoint=policy_checkpoint,
                   value_checkpoint=value_checkpoint)


def _make_agent(seed: int):
    from analysis import tb_v3_net

    w = _WORKER
    return tb_v3_net.make_tb_v3_net_agent(w['policy'], w['leaf'], seed, w['policy_checkpoint'], w['value_checkpoint'])


def _compact(record: dict, agent) -> dict:
    info = getattr(agent, 'attack_info', None) or {}
    return {'changed': record['changed'], 'root_switch': record['root']['switch'],
            'vct2_switched': record['vct2']['switched'], 'simulations': record['tree']['simulations'],
            'e4': {k: info[k] for k in ('ran', 'skip', 'base_route') if k in info}}


def play_game(task: dict, make_agent=None, max_plies: int | None = None) -> dict:
    """One self-play game (``max_plies`` only for tests: such a game is marked truncated and never trained on)."""
    from run_mcts_v8_benchmark import _move_record, derive_seed, make_opening

    make_agent = make_agent or _make_agent
    started_game = perf_counter()
    seed = task['game_seed']
    opening = make_opening(task['opening_seed'], OPENING['random_plies'], OPENING['radius'])
    game = Game()
    for move in opening:
        game.play(*move)
    agents = {BLACK: make_agent(derive_seed(seed, 'black')), WHITE: make_agent(derive_seed(seed, 'white'))}
    rng = Random(derive_seed(seed, 'sample'))
    plies, truncated, events = [], False, {}
    while not game.done:
        if max_plies is not None and len(game.history) >= max_plies:
            truncated = True
            break
        color, agent = game.to_play, agents[game.to_play]
        started = perf_counter()
        engine_move = agent.select_move(game)
        seconds = perf_counter() - started
        diag = agent.diagnostics
        record = _move_record(len(game.history), seconds, diag)
        route, visits, losses = diag.v8_route, targets.raw_visits(diag), targets.proven_losses(record)
        if len(game.history) - len(opening) < targets.SAMPLE_PLIES:
            played, event = targets.sample_move(route, engine_move, visits, losses, rng)
        else:
            played, event = [int(engine_move[0]), int(engine_move[1])], 'engine'
        if tuple(played) not in set(game.legal_moves()):
            raise RuntimeError(f'ply {len(game.history)}: {event} move {played} is not legal')
        events[event] = events.get(event, 0) + 1
        plies.append({'ply': len(game.history), 'color': 'black' if color == BLACK else 'white', 'route': route,
                      'engine_move': [int(engine_move[0]), int(engine_move[1])], 'played': played, 'event': event,
                      'seconds': round(seconds, 4), 'visits': visits, 'proven_loss': losses,
                      'target': targets.policy_target(route, engine_move, visits, losses),
                      **_compact(record, agent)})
        game.play(*played)
    winner = None if truncated else game.winner
    result = 'truncated' if truncated else 'draw' if winner is None else 'black' if winner == BLACK else 'white'
    return {**task['identity'], 'game_index': task['game_index'], 'game_seed': seed,
            'opening_seed': task['opening_seed'], 'opening': [list(m) for m in opening],
            'moves': [list(m) for m in game.history], 'length': len(game.history), 'result': result,
            'winner': winner, 'real_draw': not truncated and winner is None and len(game.history) == 15 * 15,
            'truncated': truncated, 'events': events, 'plies': plies,
            'game_seconds': round(perf_counter() - started_game, 2), 'git': task['git']}


def _play_chunk(tasks: list[dict]) -> list[dict]:
    return [play_game(t, max_plies=t.get('max_plies')) for t in tasks]


def load_finished(path: Path, identity: dict) -> dict[int, dict]:
    """Finished games of ``path`` by index; refuses a file of another run."""
    done = {}
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            other = {k: row.get(k) for k in identity}
            if other != identity:
                bad = sorted(k for k in identity if other[k] != identity[k])
                raise ValueError(f'{path}: game {row.get("game_index")} is of another run ({bad})')
            done[row['game_index']] = row
    return done


def summarize(rows: list[dict]) -> dict:
    results, routes, events, kinds = {}, {}, {}, {}
    seconds = sorted(p['seconds'] for r in rows for p in r['plies'])
    for r in rows:
        results[r['result']] = results.get(r['result'], 0) + 1
        for k, v in r['events'].items():
            events[k] = events.get(k, 0) + v
        for p in r['plies']:
            routes[p['route']] = routes.get(p['route'], 0) + 1
            kinds[p['target']['kind']] = kinds.get(p['target']['kind'], 0) + 1
    lengths = [r['length'] for r in rows]
    return {'games': len(rows), 'results': results, 'moves': len(seconds), 'routes': routes, 'events': events,
            'target_kinds': kinds, 'length_mean': round(sum(lengths) / len(lengths), 2) if lengths else None,
            'move_seconds': {'mean': round(sum(seconds) / len(seconds), 4) if seconds else None,
                             'p99': seconds[int(0.99 * (len(seconds) - 1))] if seconds else None,
                             'max': seconds[-1] if seconds else None},
            'game_minutes': round(sum(r['game_seconds'] for r in rows) / 60, 1)}


def run(args, log=print) -> dict:
    """Generate (or resume) the games ``[start, start + games)``; returns the run summary payload."""
    from analysis import tb_v3_net
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    indices = list(range(args.start, args.start + args.games))
    check_seeds(args.run_seed, args.cycle, indices, measurement=getattr(args, 'measurement', False))
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): self-play data must name its code commit')
    models = tb_v3_net.model_identity(args.policy_checkpoint, args.value_checkpoint)
    identity = run_identity(args.cycle, args.run_seed, models, args.device)
    finished = load_finished(args.jsonl, identity)
    todo = [i for i in indices if i not in finished]
    tasks = [{'identity': identity, 'git': git, 'game_index': i, 'game_seed': game_seed(args.run_seed, args.cycle, i),
              'opening_seed': opening_seed(args.run_seed, args.cycle, i),
              **({'max_plies': args.max_plies} if getattr(args, 'max_plies', None) else {})} for i in todo]
    log(f'h7c self-play: cycle {args.cycle}, run seed {args.run_seed}, games {len(indices)} '
        f'({len(finished)} already done), workers {args.workers}, model {identity["model_hash"][:12]}'
        f'{" (frozen)" if identity["frozen_models"] else ""}', flush=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress('h7c-selfplay', len(tasks), 'games')
    started = perf_counter()
    if tasks:
        chunks = [[t] for t in tasks]
        with args.jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker,
                initargs=(str(args.policy_checkpoint), str(args.value_checkpoint), args.device)) as pool:
            def keep(_, rows):
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['game_index']] = row
                out.flush()
            run_chunks(pool, _play_chunk, chunks, progress, len, unit_name='games', on_result=keep)
    rows = [finished[i] for i in indices]
    return {'format': f'{FORMAT}-summary', 'identity': identity, 'git': git, 'start': args.start,
            'games': args.games, 'workers': args.workers, 'wall_seconds': round(perf_counter() - started, 1),
            'summary': summarize(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--run-seed', type=int, required=True, help='8800-8999')
    parser.add_argument('--games', type=int, required=True)
    parser.add_argument('--start', type=int, default=0, help='first game index')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--value-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help='value device (cuda canonical; cpu certified)')
    parser.add_argument('--jsonl', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only: data from a dirty tree')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
