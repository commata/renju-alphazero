"""A3.1 / v2 development check: an engine at the A3 divergence moves, in its game state (docs/web-ser5.md §9.5).

A fresh-agent replay (``run_rt30_replay.py``) does not reproduce a game's tree move: PUCT rollouts
follow the agent's random stream, which had advanced through the game. So this script replays each
game of the manifest (``docs/rt30-eval/a3-divergence-v1.json``) from its opening: the engine gets the
A3 / E3-A seed (``derive_seed(game seed, 'v8')``) and searches every one of its own moves up to the
divergence move, as in the game; the recorded moves are played (both sides), so the position at the
divergence ply is the game's and the engine's random stream is the game's as long as its own prefix
moves equal the record (``prefix_mismatches`` counts the ones that did not). The opponent is not run.

Engines (``--engine``): ``tb1`` (TB-Engine-v1 with the exact speed-ups, no deadline: its real time on
this machine), ``v1`` (TB-RT30-v1) and ``v2`` (the TB-RT30-v2 candidate), both with ``--deadline``
(default 27.5 s) on every move. ``--workers`` games run at once: 4 reproduces A3's CPU load, 1 a
single engine search on the machine.

``recovered``: the engine played TB-Engine-v1's move at the divergence ply (for the ``control`` cases
of ``a3-root-control-v1.json``: it kept TB-Engine-v1's tree move, which TB1 proved SAFE late). The summary splits it by
cause (``attack`` / ``root``) and by what the divergence did to the A3 result, and bounds the score
that recovering those moves could return (``points_if_recovered``: the A3-minus-E3-A points of the
recovered games, an optimistic bound: a recovered game can still split later).

    python scripts/run_rt30_divergence.py --checkpoint runs/h3_policy_64x4/best.pt --engine v1 --workers 4 \\
        --games-jsonl runs/rt30/a31_v1_w4.jsonl --output runs/rt30/a31_v1_w4.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hybrid.progress import Progress  # noqa: E402
from scripts.run_mcts_v8_benchmark import derive_seed  # noqa: E402

MANIFEST = ROOT / 'docs' / 'rt30-eval' / 'a3-divergence-v1.json'
E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
ENGINES = ('tb1', 'v1', 'v2')
SCORE = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
PHASES = ('v8_attack_seconds', 'v8_vct_seconds', 'v8_tree_seconds', 'v8_root_seconds', 'v8_vct2_seconds')
_WORKER: dict = {}


def _init_worker(checkpoint, uniform_prior):
    os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    policy = None
    if checkpoint:
        from analysis.tb_engine_v1 import check_checkpoint
        from hybrid.h4_policy import RootPolicy

        check_checkpoint(checkpoint)
        policy = RootPolicy(checkpoint, threads=1)
    _WORKER.update(policy=policy, uniform_prior=uniform_prior)


def make_engine(name: str, seed: int):
    from rt30 import fast

    policy, uniform = _WORKER['policy'], _WORKER['uniform_prior']
    if name == 'tb1':
        from analysis.mcts_v8_agent import MCTSV8Agent
        from analysis.tb_engine_v1 import TB_ENGINE_V1

        config = {**TB_ENGINE_V1, **({'puct_prior': 'uniform'} if uniform else {})}
        agent = MCTSV8Agent(seed=seed, root_policy=None if uniform else policy, **config)
    elif name == 'v1':
        from rt30 import tb_rt30_v1

        agent = tb_rt30_v1.make_agent(policy, seed=seed, uniform_prior=uniform)
    elif name == 'v2':
        from rt30 import tb_rt30_v2

        agent = tb_rt30_v2.make_agent(policy, seed=seed, uniform_prior=uniform)
    else:
        raise ValueError(f'unknown engine {name!r}')
    fast.install_all()
    return agent


def record_of(key: str) -> dict:
    """The E3-A game record of a manifest key ``<file stem>/<line>/<ply>``."""
    stem, line, _ = key.split('/')
    with (E3 / f'{stem}.jsonl').open(encoding='utf-8') as handle:
        for i, text in enumerate(handle):
            if i == int(line):
                return json.loads(text)
    raise KeyError(key)


def rt_fields(agent) -> dict:
    info = getattr(agent, 'rt_info', None)
    if info is None:
        return {}
    out = {'timeouts': list(info.timeouts), 'skipped': list(info.skipped), 'rescued': info.rescued}
    for name in ('attack_cap_hit', 'attack_retried', 'attack_retry_seconds', 'attack_retry_win'):
        if hasattr(info, name):
            out[name] = getattr(info, name)
    return out


def play_case(task: dict) -> dict:
    from renju import BLACK, Game
    from rt30.deadline import Deadline

    record = record_of(task['key'])
    ply = int(task['key'].split('/')[-1])
    color = BLACK if record['v8_color'] == 'black' else -BLACK
    agent = make_engine(task['engine'], derive_seed(record['seed'], 'v8'))
    own = {m['ply'] for m in record['v8_moves']}
    opening = min(own | {ply})  # the recorded moves before the engine's first own move
    game = Game()
    for move in record['moves'][:opening]:
        game.play(*move)
    mismatches, prefix_seconds, result = [], 0.0, None
    started = perf_counter()
    for i in range(len(game.history), ply + 1):
        if game.to_play == color:
            deadline = Deadline.after(task['deadline']) if task['engine'] != 'tb1' else None
            t = perf_counter()
            move = agent.select_move(game, deadline=deadline) if deadline else agent.select_move(game)
            seconds = perf_counter() - t
            if i < ply:
                prefix_seconds += seconds
                if list(move) != record['moves'][i]:
                    mismatches.append(i)
            else:
                diag = agent.diagnostics
                result = {'move': [int(move[0]), int(move[1])], 'seconds': round(seconds, 4), 'route': diag.v8_route,
                          'tree_move': list(diag.v8_v7_move) if diag.v8_v7_move else None,
                          'root_checked': [[list(m), s] for m, s in diag.v8_root_checked or ()],
                          'phases': {p[3:-8]: round(getattr(diag, p), 4) for p in PHASES}, **rt_fields(agent)}
        game.play(*record['moves'][i])
    return {**{k: task[k] for k in ('key', 'game', 'kind', 'result', 'tb1_move', 'tree_move_rec')},
            **result, 'recovered': result['move'] == task['tb1_move'], 'prefix_mismatches': mismatches,
            'prefix_seconds': round(prefix_seconds, 2), 'case_seconds': round(perf_counter() - started, 2)}


def delta(change: str) -> float:
    """A3 minus E3-A points of a case ('win>loss' -> -1.0)."""
    before, after = change.split('>')
    return SCORE[after] - SCORE[before]


def summarize(rows: list[dict]) -> dict:
    def block(subset):
        rec = [r for r in subset if r['recovered']]
        return {'cases': len(subset), 'recovered': len(rec),
                'a3_points': round(sum(delta(r['result']) for r in subset), 2),
                'points_if_recovered': round(-sum(delta(r['result']) for r in rec), 2)}

    out = {'cases': len(rows), 'recovered': sum(r['recovered'] for r in rows),
           'by_kind': {k: block([r for r in rows if r['kind'] == k]) for k in sorted({r['kind'] for r in rows})},
           'result_changed': block([r for r in rows if delta(r['result']) != 0]),
           'loss_making': block([r for r in rows if delta(r['result']) < 0]),
           'gain_making': block([r for r in rows if delta(r['result']) > 0]),
           'prefix_mismatch_cases': sum(bool(r['prefix_mismatches']) for r in rows),
           'tree_move_as_recorded': sum(r['tree_move'] == r['tree_move_rec'] for r in rows if r['kind'] != 'attack'),
           'seconds': {'median': statistics.median(r['seconds'] for r in rows) if rows else None,
                       'max': max((r['seconds'] for r in rows), default=None),
                       'over_27.5s': sum(r['seconds'] > 27.5 for r in rows)},
           'phase_median': {}}
    for kind, phase in (('attack', 'attack'), ('root', 'root'), ('control', 'root')):
        values = [r['phases'][phase] for r in rows if r['kind'] == kind and r['phases'][phase] > 0]
        out['phase_median'][f'{kind}:{phase}'] = round(statistics.median(values), 2) if values else None
    for name in ('attack_cap_hit', 'attack_retried', 'attack_retry_win'):
        if any(name in r for r in rows):
            out[name] = sum(bool(r.get(name)) for r in rows)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', choices=ENGINES, required=True)
    parser.add_argument('--checkpoint', type=Path, help='H3 policy checkpoint')
    parser.add_argument('--uniform-prior', action='store_true', help='no checkpoint: smoke run only')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--kinds', help='comma list of case kinds to run (default: all of the manifest)')
    parser.add_argument('--limit', type=int, help='first N cases (smoke)')
    parser.add_argument('--deadline', type=float, default=27.5, help='v1/v2: seconds per move')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--games-jsonl', type=Path, help='finished cases, one per line (resume)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.checkpoint is None and not args.uniform_prior:
        parser.error('pass --checkpoint, or --uniform-prior for a smoke run')

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    kinds = set(args.kinds.split(',')) if args.kinds else {c['kind'] for c in manifest['cases']}
    cases = [c for c in manifest['cases'] if c['kind'] in kinds][:args.limit]
    tasks = [{'key': c['key'], 'game': c['game'], 'kind': c['kind'], 'result': c['result'], 'tb1_move': c['tb1_move'],
              'tree_move_rec': c['tb1_root_checked'][0][0] if c['tb1_root_checked'] else None,
              'engine': args.engine, 'deadline': args.deadline} for c in cases]
    done = {}
    if args.games_jsonl is not None and args.games_jsonl.exists():
        for line in args.games_jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['key']] = row
    todo = [t for t in tasks if t['key'] not in done]
    print(f'{args.engine}: {len(tasks)} cases ({len(done)} already done), workers={args.workers}, '
          f'deadline={args.deadline if args.engine != "tb1" else None}', flush=True)
    progress = Progress(f'divergence-{args.engine}', len(todo), 'cases')
    rows = []
    if todo:
        init = (str(args.checkpoint) if args.checkpoint else None, args.uniform_prior)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=init) as pool:
            pending = {pool.submit(play_case, t) for t in todo}
            while pending:
                finished, pending = wait(pending, timeout=progress.every, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    rows.append(row)
                    if args.games_jsonl is not None:
                        args.games_jsonl.parent.mkdir(parents=True, exist_ok=True)
                        with args.games_jsonl.open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(row) + '\n')
                    progress.update(1, extra=f"{row['game']} {row['kind']} {'recovered' if row['recovered'] else 'not'} "
                                             f"({row['seconds']:.1f}s, case {row['case_seconds']:.0f}s)")
                if not finished:
                    progress.heartbeat(f'{len(pending)} cases running/pending')
    order = {t['key']: i for i, t in enumerate(tasks)}
    rows = sorted(list(done.values()) + rows, key=lambda r: order.get(r['key'], len(order)))
    summary = summarize(rows)
    report = {'format': 'rt30-divergence-v1', 'engine': args.engine, 'manifest': manifest['version'],
              'deadline': args.deadline if args.engine != 'tb1' else None, 'workers': args.workers,
              'uniform_prior': args.uniform_prior,
              'machine': {'platform': platform.platform(), 'cpu_count': os.cpu_count()},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
