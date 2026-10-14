"""Full-budget safety re-check of the moves an RT30 run changed (docs/web-ser5.md §5.2 gate).

``run_rt30_replay.py --baseline`` classifies a changed move only by the statuses the baseline's own
search gave it; a move the baseline never checked reads UNKNOWN. This re-checks every changed move
(and the baseline's move, for comparison) with budgets far above an online search:

    vct1  ``_BudgetedSolver.status_after``: SAFE / UNSAFE / UNKNOWN (UNSAFE is a proven loss)
    vct2  the selective depth-2 check (``SelectiveSolver``): UNSAFE proven, SAFE = no targeted VCT2 found

The gate value is ``changed_move_proven_unsafe`` (changed moves with vct1 or vct2 UNSAFE): it must be 0.
Lost wins (``lost_win``) are not re-checked: the baseline's own_vct already is a proof.

    python scripts/check_rt30_changed_moves.py --run runs/rt30/rt27_h3_s60.json --workers 4 \\
        --output runs/rt30/recheck_rt27_h3_s60.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid.progress import Progress, run_chunks  # noqa: E402

E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'


def position(key: str):
    from renju import Game

    stem, line, ply = key.rsplit('/', 2)
    with (E3 / f'{stem}.jsonl').open(encoding='utf-8') as handle:
        record = json.loads(handle.readlines()[int(line)])
    game = Game()
    for move in record['moves'][:int(ply)]:
        game.play(*move)
    return game


def check(task):
    from analysis.mcts_v8 import _BudgetedSolver
    from analysis.selective_vct import SelectiveSolver
    from rt30 import fast_windows

    fast_windows.install()
    game = position(task['key'])
    move = tuple(task['move'])
    started = perf_counter()
    vct1 = _BudgetedSolver(**task['vct1_budget'])
    status1 = vct1.status_after(game, move)
    vct2 = SelectiveSolver(**task['vct2_budget'])
    status2 = vct2._bounded(game, move, None, None, lambda g: vct2.after_move(g, 2)[0])
    return {**{k: task[k] for k in ('key', 'who', 'kind', 'move')}, 'vct1': status1, 'vct1_nodes': vct1.nodes_used,
            'vct2': status2, 'vct2_nodes': vct2.nodes_used, 'seconds': round(perf_counter() - started, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='a run_rt30_replay.py output with vs_baseline')
    parser.add_argument('--vct1-nodes', type=int, default=800_000)
    parser.add_argument('--vct2-nodes', type=int, default=200_000)
    parser.add_argument('--no-baseline-moves', action='store_true', help='re-check only the changed moves')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run = json.loads(args.run.read_text(encoding='utf-8'))
    vs = run['summary'].get('vs_baseline')
    if not vs:
        parser.error(f'{args.run} has no vs_baseline (run the replay with --baseline)')
    budgets = {'vct1_budget': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': args.vct1_nodes},
               'vct2_budget': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': args.vct2_nodes}}
    tasks = []
    for change in vs['changed']:
        if change.get('kind') in ('lost_win', 'route_only'):
            continue
        tasks.append({'key': change['key'], 'who': 'run', 'kind': change.get('kind'), 'move': change['run'][1],
                      **budgets})
        if not args.no_baseline_moves:
            tasks.append({'key': change['key'], 'who': 'baseline', 'kind': change.get('kind'),
                          'move': change['baseline'][1], **budgets})
    print(f'{len(tasks)} checks from {args.run} (vct1 {args.vct1_nodes:,} nodes, vct2 {args.vct2_nodes:,})', flush=True)
    progress = Progress('rt30-recheck', len(tasks), 'checks')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = run_chunks(pool, check, tasks, progress, lambda t: 1, unit_name='checks')
    changed = [r for r in rows if r['who'] == 'run']
    unsafe = [r['key'] for r in changed if 'UNSAFE' in (r['vct1'], r['vct2'])]
    summary = {'run': str(args.run), 'manifest': run.get('manifest'), 'engine': run.get('engine'),
               'changed_moves_checked': len(changed),
               'changed_move_proven_unsafe': len(unsafe), 'unsafe_keys': unsafe,
               'changed_vct1': {s: sum(r['vct1'] == s for r in changed) for s in ('SAFE', 'UNKNOWN', 'UNSAFE')},
               'gate_pass': not unsafe}
    report = {'format': 'rt30-recheck-v1', **budgets, 'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if not unsafe else 1


if __name__ == '__main__':
    sys.exit(main())
