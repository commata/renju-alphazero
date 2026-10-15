"""Strength gate A3: TB-RT30-v1 (a deadline on every move) replays the E3-A games (docs/web-ser5.md §9.3).

E3-A is TB-Engine-v1's adoption run: 2 x 25 opening pairs (seeds 8415, 8416), each opening played
with both colours against ``v8:full+vct2atk``. This script plays the same games with TB-RT30-v1 in
TB-Engine-v1's seat: the same openings, the same game seeds (the engine gets ``derive_seed(game, 'v8')``
and the opponent ``derive_seed(game, 'v7')``, as in ``run_mcts_v8_benchmark.py``) and the same opponent.
Both engines are deterministic and RT30 equals TB-Engine-v1 when no deadline event happens, so a game
should follow its E3-A record until the first RT30 move that a deadline changed; ``--reference`` pairs
every game with its record and reports the paired score difference and where the games split.

The exact speed-ups (``rt30.fast``) are installed in every worker: they never change a move, so the
opponent plays its own moves, only faster. RT30's time depends on the CPU it gets: keep ``--workers``
at or below the physical cores (SER5: 4) and keep the machine otherwise idle; the report records both.

    python scripts/run_rt30_match.py --checkpoint runs/h3_policy_64x4/best.pt --workers 4 \\
        --reference docs/mcts-v8-results/e3/puct_policy_vct2_50k_8415.jsonl \\
                    docs/mcts-v8-results/e3/puct_policy_vct2_50k_8416.jsonl \\
        --games-jsonl runs/rt30/a3_e3.jsonl --output runs/rt30/a3_e3.json 2>&1 | tee runs/rt30/a3_e3.log

Other opponents (``--opponent``, no reference): ``tb1`` (TB-Engine-v1 itself, no deadline: minutes on
some moves), ``v7`` (frozen MCTS-v7), ``v5`` (fast smoke opponent). ``--games-jsonl`` keeps every
finished game; a rerun with the same arguments continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from random import Random
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hybrid.progress import Progress  # noqa: E402
from renju import BLACK, WHITE, Game, IllegalMove  # noqa: E402
from scripts.run_mcts_v8_benchmark import derive_seed, make_opening  # noqa: E402

E3_OPPONENT = 'v8:full+vct2atk'
OPPONENTS = ('e3', 'tb1', 'v7', 'v5')  # e3: E3-A's opponent; v5: fast smoke opponent
SCORE = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
BOOTSTRAP = 10_000
_WORKER: dict = {}


def _init_worker(checkpoint, uniform_prior):
    os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    policy = None
    if checkpoint:
        from analysis.tb_engine_v1 import check_checkpoint
        from hybrid.h4_policy import RootPolicy

        check_checkpoint(checkpoint)
        policy = RootPolicy(checkpoint, threads=1)
    elif not uniform_prior:
        raise ValueError('pass --checkpoint, or --uniform-prior for a smoke run (not TB-RT30-v1)')
    _WORKER.update(policy=policy, uniform_prior=uniform_prior)


def make_opponent(name: str, seed: int):
    if name == 'e3':
        from scripts.run_mcts_v8_benchmark import make_opponent as make_benchmark_opponent

        return make_benchmark_opponent(E3_OPPONENT, seed, {})
    if name == 'tb1':
        from analysis.tb_engine_v1 import make_agent

        if _WORKER['policy'] is None:
            raise ValueError('the tb1 opponent needs --checkpoint')
        return make_agent(_WORKER['policy'], seed=seed)
    if name == 'v7':
        from agents import MCTSV7Agent

        return MCTSV7Agent(seed=seed)
    if name == 'v5':
        from agents import MCTSV5Agent
        from search.mcts_v6 import V5_FINAL

        agent = MCTSV5Agent(seed=seed, stage='a', **V5_FINAL)
        agent.name = 'MCTS-v5-final'
        return agent
    raise ValueError(f'unknown opponent {name!r}')


def engine_module(name: str):
    from rt30 import tb_rt30_v1, tb_rt30_v2

    return {'v1': tb_rt30_v1, 'v2': tb_rt30_v2}[name]


def play_one(task: dict) -> dict:
    from rt30 import fast
    from rt30.deadline import Deadline

    rt30 = engine_module(task.get('engine', 'v1')).make_agent(
        _WORKER['policy'], seed=derive_seed(task['seed'], 'v8'), uniform_prior=_WORKER['uniform_prior'])
    fast.install_all()
    opponent = make_opponent(task['opponent'], derive_seed(task['seed'], 'v7'))
    rt_color = BLACK if task['rt30_color'] == 'black' else WHITE
    game = Game()
    for move in task['opening']:
        game.play(*move)
    rt_moves, opp_seconds = [], []
    started_game = perf_counter()
    while not game.done:
        if task.get('max_plies') and len(game.history) >= task['max_plies']:
            break
        is_rt = game.to_play == rt_color
        started = perf_counter()
        if is_rt:
            move = rt30.select_move(game, deadline=Deadline.after(task['deadline']))
        else:
            move = opponent.select_move(game)
        seconds = perf_counter() - started
        if is_rt:
            info, diag = rt30.rt_info, rt30.diagnostics
            rt_moves.append({'ply': len(game.history), 'seconds': round(seconds, 4), 'route': diag.v8_route,
                             'timeouts': list(info.timeouts), 'skipped': list(info.skipped),
                             'rescued': info.rescued, 'move': [int(move[0]), int(move[1])],
                             **{k: getattr(info, k) for k in ('attack_cap_hit', 'attack_retried', 'attack_retry_win')
                                if hasattr(info, k)}})
        else:
            opp_seconds.append(round(seconds, 4))
        try:
            game.play(*move)
        except IllegalMove as exc:
            who = rt30.name if is_rt else getattr(opponent, 'name', task['opponent'])
            raise RuntimeError(f'{task["key"]}: {who} returned illegal move {move}: {exc}') from exc
    if not game.done:
        result = 'unfinished'
    else:
        result = 'draw' if game.winner is None else 'win' if game.winner == rt_color else 'loss'
    return {'key': task['key'], 'run_seed': task['run_seed'], 'pair': task['pair'], 'seed': task['seed'],
            'opponent': task['opponent'], 'rt30_color': task['rt30_color'], 'result': result,
            'winner': game.winner, 'length': len(game.history), 'moves': [list(m) for m in game.history],
            'rt30_moves': rt_moves, 'opponent_move_seconds': opp_seconds,
            'game_seconds': round(perf_counter() - started_game, 2)}


def build_tasks(args) -> list[dict]:
    """Openings and game seeds exactly as ``run_mcts_v8_benchmark.build_tasks`` makes them."""
    tasks = []
    for run_seed in args.seeds:
        for pair in range(args.pairs):
            opening = make_opening(derive_seed(run_seed, 'opening', pair), args.opening_random_plies,
                                   args.opening_radius)
            for color in (BLACK, WHITE):
                name = 'black' if color == BLACK else 'white'
                prefix = 'rt30' if args.engine == 'v1' else f'rt30{args.engine}'  # v1 keys as in A3's JSONL
                tasks.append({'key': f'{prefix}@{args.opponent}/{run_seed}/{pair}/{name}', 'run_seed': run_seed,
                              'engine': args.engine,
                              'pair': pair, 'seed': derive_seed(run_seed, 'game', pair, color),
                              'opponent': args.opponent, 'rt30_color': name,
                              'opening': [list(m) for m in opening], 'deadline': args.deadline,
                              'max_plies': args.max_plies})
    return tasks


def wilson(points: float, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = points / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def bootstrap(values: list[float], seed: int = 0) -> list[float] | None:
    """95% interval of the mean, resampling opening pairs."""
    if not values:
        return None
    rng = Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAP))
    return [round(means[int(0.025 * BOOTSTRAP)], 3), round(means[int(0.975 * BOOTSTRAP) - 1], 3)]


def score(games) -> float | None:
    games = [g for g in games if g['result'] != 'unfinished']
    return round(statistics.fmean(SCORE[g['result']] for g in games), 4) if games else None


def pair_means(games) -> dict[tuple, float]:
    """Mean score per opening pair (both colours), keyed by (run seed, pair)."""
    out: dict[tuple, list] = {}
    for g in games:
        out.setdefault((game_id(g)[0], g['pair']), []).append(SCORE[g['result']])
    return {k: statistics.fmean(v) for k, v in out.items()}


def game_id(record: dict) -> tuple:
    """(run seed, pair, colour) of a game of this script or of a run_mcts_v8_benchmark record."""
    if 'rt30_color' in record:
        return record['run_seed'], record['pair'], record['rt30_color']
    return int(record['key'].split('/')[-3]), record['pair'], record['v8_color']


def load_reference(paths) -> dict[tuple, dict]:
    out = {}
    for path in paths or ():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                out[game_id(record)] = record
    return out


def divergence(game: dict, ref: dict) -> dict:
    """Where ``game`` leaves its reference record and why.

    ``same``: the same game. ``deadline``: the first different move is an RT30 move with a deadline
    event (a timeout or a skipped phase), the expected kind. ``rt30_no_deadline``: an RT30 move
    without one, and ``opponent``: an opponent move. Neither should happen (both engines are
    deterministic); they point at a different policy evaluation (machine, torch) or a setup error.
    """
    moves, ref_moves = game['moves'], ref['moves']
    ply = next((i for i, (a, b) in enumerate(zip(moves, ref_moves)) if a != b), None)
    if ply is None:
        if len(moves) == len(ref_moves):
            return {'kind': 'same', 'ply': None}
        ply = min(len(moves), len(ref_moves))
    rt = next((m for m in game['rt30_moves'] if m['ply'] == ply), None)
    if rt is None:
        kind = 'opponent'
    else:
        kind = 'deadline' if rt['timeouts'] or rt['skipped'] else 'rt30_no_deadline'
    return {'kind': kind, 'ply': ply}


def paired_summary(games: list[dict], reference: dict[tuple, dict]) -> dict:
    finished = [g for g in games if g['result'] != 'unfinished' and game_id(g) in reference]
    refs = [reference[game_id(g)] for g in finished]
    mine, theirs = pair_means(finished), pair_means(refs)
    diffs = [mine[k] - theirs[k] for k in sorted(set(mine) & set(theirs))]
    splits = [divergence(g, r) for g, r in zip(finished, refs)]
    changed = [{'key': g['key'], 'rt30': g['result'], 'reference': r['result'], **s}
               for g, r, s in zip(finished, refs, splits) if g['result'] != r['result']]
    return {
        'games': len(finished), 'reference_score': score(refs), 'rt30_score': score(finished),
        'diff': round(statistics.fmean(diffs), 4) if diffs else None, 'diff_ci95_pairs': bootstrap(diffs),
        'divergence': {k: sum(s['kind'] == k for s in splits)
                       for k in ('same', 'deadline', 'rt30_no_deadline', 'opponent')},
        'result_changed': changed,
    }


def summarize(games: list[dict], reference: dict[tuple, dict] | None = None) -> dict:
    finished = [g for g in games if g['result'] != 'unfinished']
    n = len(finished)
    points = sum(SCORE[g['result']] for g in finished)
    moves = [m for g in games for m in g['rt30_moves']]
    seconds = sorted(m['seconds'] for m in moves)
    pick = (lambda q: seconds[min(len(seconds) - 1, int(q * len(seconds)))]) if seconds else (lambda q: None)
    pairs: dict[tuple, list] = {}
    for g in finished:
        pairs.setdefault((g['run_seed'], g['pair']), []).append(g['result'])
    out = {
        'games': len(games), 'finished': n, 'unfinished': len(games) - n,
        'wins': sum(g['result'] == 'win' for g in finished), 'draws': sum(g['result'] == 'draw' for g in finished),
        'losses': sum(g['result'] == 'loss' for g in finished),
        'score': score(finished), 'ci95_wilson': wilson(points, n),
        'ci95_pairs': bootstrap(list(pair_means(finished).values())),
        'by_color': {c: score([g for g in finished if g['rt30_color'] == c]) for c in ('black', 'white')},
        'pairs_both_won': sum(sorted(r) == ['win', 'win'] for r in pairs.values()),
        'pairs_both_lost': sum(sorted(r) == ['loss', 'loss'] for r in pairs.values()),
        'rt30_moves': len(moves),
        'rt30_seconds': {'median': statistics.median(seconds) if seconds else None, 'p95': pick(0.95),
                         'p99': pick(0.99), 'max': seconds[-1] if seconds else None,
                         'over_27.5s': sum(x > 27.5 for x in seconds), 'over_30s': sum(x >= 30 for x in seconds)},
        'rt30_timeouts': {p: sum(p in m['timeouts'] for m in moves) for p in ('attack', 'stage_vct', 'root', 'vct2')},
        'rt30_moves_with_timeout': sum(bool(m['timeouts'] or m['skipped']) for m in moves),
        'rt30_rescued': sum(m['rescued'] for m in moves),
        **({k: sum(bool(m.get(k)) for m in moves) for k in ('attack_cap_hit', 'attack_retried', 'attack_retry_win')}
           if any('attack_retried' in m for m in moves) else {}),
        'rt30_routes': {r: sum(m['route'] == r for m in moves) for r in sorted({m['route'] for m in moves})},
        'mean_length': round(statistics.fmean(g['length'] for g in games), 1) if games else None,
        'game_minutes': round(sum(g['game_seconds'] for g in games) / 60, 1),
    }
    if reference:
        out['vs_reference'] = paired_summary(games, reference)
    return out


def load_finished(path: Path | None, keys: set[str]) -> dict:
    if path is None or not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            if record['key'] in keys:
                done[record['key']] = record
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, help='H3 policy checkpoint (TB-RT30-v1 and the tb1 opponent)')
    parser.add_argument('--uniform-prior', action='store_true', help='no checkpoint: smoke run, not TB-RT30-v1')
    parser.add_argument('--engine', choices=('v1', 'v2'), default='v1', help='TB-RT30-v1 or the v2 candidate')
    parser.add_argument('--opponent', choices=OPPONENTS, default='e3')
    parser.add_argument('--seeds', type=int, nargs='+', default=[8415, 8416], help="run seeds (E3-A's by default)")
    parser.add_argument('--pairs', type=int, default=25, help='opening pairs per seed')
    parser.add_argument('--opening-random-plies', type=int, default=2)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--deadline', type=float, help="RT30's deadline per move (default TB-RT30-v1's 27.5 s)")
    parser.add_argument('--max-plies', type=int, help='stop games at this ply (smoke runs; such games are unfinished)')
    parser.add_argument('--reference', type=Path, nargs='+', help='E3-A JSONL records to pair the games with')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--games-jsonl', type=Path, help='finished games, one per line (resume)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    module = engine_module(args.engine)
    NAME, config_hash = module.NAME, module.config_hash
    config = module.TB_RT30_V1 if args.engine == 'v1' else module.TB_RT30_V2

    if args.checkpoint is None and not args.uniform_prior:
        parser.error('pass --checkpoint, or --uniform-prior for a smoke run')
    if args.opponent == 'tb1' and args.checkpoint is None:
        parser.error('the tb1 opponent needs --checkpoint')
    if args.reference and args.opponent != 'e3':
        parser.error('--reference records are E3-A games: use --opponent e3')
    args.deadline = args.deadline or config['deadline_seconds']
    reference = load_reference(args.reference)

    tasks = build_tasks(args)
    if reference:
        missing = [t['key'] for t in tasks if (t['run_seed'], t['pair'], t['rt30_color']) not in reference]
        if missing:
            parser.error(f'{len(missing)} games have no reference record, e.g. {missing[0]}')
        bad = [t['key'] for t in tasks
               if reference[(t['run_seed'], t['pair'], t['rt30_color'])]['moves'][:len(t['opening'])] != t['opening']]
        if bad:
            parser.error(f'{len(bad)} openings differ from the reference, e.g. {bad[0]}')
    done = load_finished(args.games_jsonl, {t['key'] for t in tasks})
    todo = [t for t in tasks if t['key'] not in done]
    print(f'{NAME} vs {args.opponent}: {len(tasks)} games ({len(done)} already done), workers={args.workers}, '
          f'deadline={args.deadline:g}s, config {config_hash()[:12]}', flush=True)
    progress = Progress('rt30-match', len(todo), 'games')
    results = []
    if todo:
        init = (str(args.checkpoint) if args.checkpoint else None, args.uniform_prior)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=init) as pool:
            pending = {pool.submit(play_one, t) for t in todo}
            while pending:
                finished, pending = wait(pending, timeout=progress.every, return_when=FIRST_COMPLETED)
                for future in finished:
                    game = future.result()
                    results.append(game)
                    if args.games_jsonl is not None:
                        args.games_jsonl.parent.mkdir(parents=True, exist_ok=True)
                        with args.games_jsonl.open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(game) + '\n')
                    ref = reference.get(game_id(game))
                    vs = f", E3-A {ref['result']} ({divergence(game, ref)['kind']})" if ref else ''
                    progress.update(1, extra=f"{game['key']} {game['result']}{vs} ({game['length']} plies, "
                                             f"{game['game_seconds']:.0f}s); score {score(list(done.values()) + results)}")
                if not finished:
                    progress.heartbeat(f'{len(pending)} games running/pending')
    games = sorted(list(done.values()) + results, key=game_id)
    summary = summarize(games, reference)
    report = {'format': 'rt30-match-v1', 'engine': NAME if args.checkpoint else f'{NAME} (uniform prior)',
              'engine_config_hash': config_hash(),
              'opponent': E3_OPPONENT if args.opponent == 'e3' else args.opponent,
              'seeds': args.seeds, 'pairs': args.pairs, 'deadline': args.deadline, 'workers': args.workers,
              'max_plies': args.max_plies, 'opening_random_plies': args.opening_random_plies,
              'opening_radius': args.opening_radius, 'reference': [str(p) for p in args.reference or ()],
              'machine': {'platform': platform.platform(), 'cpu_count': os.cpu_count()},
              'summary': summary, 'games': games}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'vs_reference'}, indent=1), flush=True)
    if 'vs_reference' in summary:
        print(json.dumps(summary['vs_reference'], indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
