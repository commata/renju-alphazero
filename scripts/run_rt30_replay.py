"""Replay recorded E3-A positions with TB-Engine-v1: TB1 vs TB1-fast equivalence and timing (docs/web-ser5.md §5).

Each selected position (the board before a V8 move of the E3-A games) is searched by a fresh
agent with a fixed seed, once per arm:

    frozen  TB-Engine-v1 as frozen
    fast    the same engine with ``rt30.fast_windows`` installed (exact speed-up)

With both arms the run is the equivalence check: every position must give the same move and the
same diagnostics (every field except the ``*_seconds`` timings). With one arm it is a timing
benchmark (the mini-PC baseline). A replay does not reproduce the recorded move: the recorded
agent's random stream had advanced through the game, the replay starts a fresh one.

The tree route needs the H3 policy (``--checkpoint``, checked against the frozen hashes). Without
it, ``--uniform-prior`` runs the TB1 configuration with ``puct_prior='uniform'``: a valid
equivalence check of the tactical modules, but not TB1 itself (the output says so).

    python scripts/run_rt30_replay.py --checkpoint runs/h3_policy_64x4/best.pt --select slow `
        --workers 4 --output runs/rt30/equivalence_slow.json 2>&1 | Tee-Object runs/rt30/equivalence_slow.log
    python scripts/run_rt30_replay.py --checkpoint runs/h3_policy_64x4/best.pt --arms frozen `
        --select sample --sample 300 --workers 1 --output runs/rt30/baseline_frozen.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid.progress import Progress, run_chunks  # noqa: E402

E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
DEFAULT_JSONL = [E3 / 'puct_policy_vct2_50k_8415.jsonl', E3 / 'puct_policy_vct2_50k_8416.jsonl']
ARMS = ('frozen', 'fast')
PHASES = {'attack': 'v8_attack_seconds', 'stage_vct': 'v8_vct_seconds', 'tree': 'v8_tree_seconds',
          'root': 'v8_root_seconds', 'vct2': 'v8_vct2_seconds', 'v7_modules': 'v7_module_seconds'}
DEADLINE = 27.5

_WORKER: dict = {}


def load_positions(paths, select: str, slow_seconds: float, sample: int, seed: int, limit: int | None):
    rows = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle):
                record = json.loads(line)
                for move in record['v8_moves']:
                    rows.append({'key': f"{Path(path).stem}/{line_no}/{move['ply']}", 'moves': record['moves'][:move['ply']],
                                 'recorded_seconds': move['seconds'], 'recorded_route': move['route']})
    if select == 'slow':
        rows = [r for r in rows if r['recorded_seconds'] > slow_seconds]
    elif select == 'sample':
        rows = Random(seed).sample(rows, min(sample, len(rows)))
    return rows[:limit] if limit else rows


def _init_worker(checkpoint, uniform_prior):
    from analysis.tb_engine_v1 import TB_ENGINE_V1

    config = dict(TB_ENGINE_V1)
    policy = None
    if checkpoint:
        from analysis.tb_engine_v1 import check_checkpoint
        from hybrid.h4_policy import RootPolicy

        check_checkpoint(checkpoint)
        policy = RootPolicy(checkpoint, threads=1)
    elif uniform_prior:
        config['puct_prior'] = 'uniform'
    else:
        raise ValueError('pass --checkpoint, or --uniform-prior for a policy-free equivalence check')
    _WORKER.update(config=config, policy=policy)


def _diagnostics(diag) -> dict:
    return {k: v for k, v in vars(diag).items() if not k.endswith('_seconds')}


def _search(moves, arm: str, seed: int):
    from analysis.mcts_v8_agent import MCTSV8Agent
    from renju import Game
    from rt30 import fast_windows

    if arm == 'fast':
        fast_windows.install()
    else:
        fast_windows.uninstall()
    game = Game()
    for move in moves:
        game.play(*move)
    agent = MCTSV8Agent(seed=seed, root_policy=_WORKER['policy'], **_WORKER['config'])
    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    diag = agent.diagnostics
    return {'move': list(move), 'seconds': round(seconds, 4), 'route': diag.v8_route,
            'phases': {name: round(getattr(diag, field), 4) for name, field in PHASES.items()},
            'nodes': {'attack': diag.v8_attack_nodes, 'stage_vct': diag.v8_vct_nodes, 'root': diag.v8_root_nodes,
                      'vct2': diag.v8_vct2_nodes}}, _diagnostics(diag)


def _first_difference(a: dict, b: dict):
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            return key
    return None


def run_chunk(chunk):
    rows_out = []
    for row in chunk['rows']:
        out = {'key': row['key'], 'recorded_seconds': row['recorded_seconds'], 'recorded_route': row['recorded_route']}
        diags = {}
        for arm in chunk['arms']:
            out[arm], diags[arm] = _search(row['moves'], arm, chunk['seed'])
        if len(chunk['arms']) == 2:
            a, b = (diags[arm] for arm in chunk['arms'])
            ra, rb = (out[arm] for arm in chunk['arms'])
            same = ra['move'] == rb['move'] and a == b
            out['identical'] = same
            if not same:
                out['first_difference'] = 'move' if ra['move'] != rb['move'] else _first_difference(a, b)
        rows_out.append(out)
    return rows_out


def _dist(values):
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {'n': len(values), 'median': round(statistics.median(values), 3), 'p95': round(pick(0.95), 3),
            'p99': round(pick(0.99), 3), 'max': round(values[-1], 3), 'sum': round(sum(values), 1),
            f'over_{DEADLINE}s': sum(v > DEADLINE for v in values)}


def summarize(rows, arms):
    summary = {'positions': len(rows)}
    for arm in arms:
        summary[arm] = {'seconds': _dist([r[arm]['seconds'] for r in rows]),
                        'phases': {name: _dist([r[arm]['phases'][name] for r in rows if r[arm]['phases'][name] > 0])
                                   for name in PHASES}}
    if len(arms) == 2:
        summary['identical'] = sum(r['identical'] for r in rows)
        summary['different'] = [{'key': r['key'], 'first_difference': r['first_difference']}
                                for r in rows if not r['identical']]
        a, b = (sum(r[arm]['seconds'] for r in rows) for arm in arms)
        summary['speedup_total'] = round(a / b, 3) if b else None
    return summary


def _commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jsonl', type=Path, nargs='+', default=DEFAULT_JSONL)
    parser.add_argument('--select', choices=('all', 'slow', 'sample'), default='sample')
    parser.add_argument('--slow-seconds', type=float, default=30.0, help='--select slow: recorded seconds above this')
    parser.add_argument('--sample', type=int, default=200)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42, help='sample seed and agent seed')
    parser.add_argument('--arms', default='frozen,fast', help="'frozen,fast' (equivalence), 'frozen' or 'fast'")
    parser.add_argument('--checkpoint', type=Path, help='H3 policy checkpoint of TB-Engine-v1')
    parser.add_argument('--uniform-prior', action='store_true', help='no checkpoint: uniform PUCT prior (not TB1)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=1, help='positions per task')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    arms = tuple(a for a in args.arms.split(',') if a)
    if not arms or set(arms) - set(ARMS):
        parser.error(f'--arms: one or both of {ARMS}')
    if args.checkpoint is None and not args.uniform_prior:
        parser.error('pass --checkpoint, or --uniform-prior for a policy-free equivalence check')

    positions = load_positions(args.jsonl, args.select, args.slow_seconds, args.sample, args.seed, args.limit)
    chunks = [{'rows': positions[i:i + args.chunk], 'arms': arms, 'seed': args.seed}
              for i in range(0, len(positions), args.chunk)]
    print(f'positions={len(positions)} arms={",".join(arms)} workers={args.workers} '
          f'engine={"TB-Engine-v1" if args.checkpoint else "TB1 config, uniform prior"}', flush=True)
    progress = Progress('rt30-replay', len(positions), 'positions')
    init = (str(args.checkpoint) if args.checkpoint else None, args.uniform_prior)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=init) as pool:
        results = run_chunks(pool, run_chunk, chunks, progress, lambda c: len(c['rows']))
    rows = [row for chunk in results for row in chunk]
    summary = summarize(rows, arms)
    report = {'format': 'rt30-replay-v1', 'engine': 'TB-Engine-v1' if args.checkpoint else 'tb1-uniform-prior',
              'arms': arms, 'select': args.select, 'slow_seconds': args.slow_seconds, 'seed': args.seed,
              'workers': args.workers, 'git_commit': _commit(), 'jsonl': [str(p) for p in args.jsonl],
              'machine': {'platform': platform.platform(), 'processor': platform.processor(),
                          'cpu_count': os.cpu_count(), 'python': platform.python_version()},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    if len(arms) == 2 and summary['identical'] != len(rows):
        print(f'NOT IDENTICAL: {len(rows) - summary["identical"]} position(s)', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
