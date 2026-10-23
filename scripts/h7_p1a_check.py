"""H7 P1a-1 function-level check: the native forbidden points equal ``renju.rules.forbidden_reason``.

Two parts, both against the Python function (the oracle), any difference fails:
    cells  every empty cell of every board of the timing positions (``h7_timing.py positions``: E3 A moves,
           web probes, H6-4b) and of the shared rule corpus (``tests/rule_validation.py``: hand-made
           five / overline / double-four / double-three / recursive-three fixtures in four rotations, V6
           replays, dense synthetic boards)
    trace  every ``forbidden_reason`` call V8-C makes (the black legality checks inside its VCF calls) while
           it checks the baseline's tree move of each corpus position with TB-Engine-v2's solver (fast3 +
           table, ``h6_5d.check_status``): the boards the engine really asks about. Each call runs both and
           the Python result is used, so the search itself is the frozen one.
The engine-level check (decisions with fast3 and fast4 identical on the corpus) is ``h7_timing.py paired``.

    python -u scripts/h7_p1a_check.py --positions docs/h7/h7_timing_positions.json --corpus docs/h7/h7_equiv_corpus.json \\
        --baseline runs/h7_timing/base_cpu.json --workers 12 --output docs/h7/h7_p1a_check.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT, ROOT / 'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FORMAT = 'h7-p1a-check-v1'


def check_board(board) -> dict:
    """Every empty cell of one board: Python and native reasons."""
    from renju.rules import forbidden_reason
    from rt30.native._forbidden import forbidden

    out = Counter()
    bad = []
    for r in range(15):
        for c in range(15):
            if board[r][c] != 0:
                continue
            want, got = forbidden_reason(board, r, c), forbidden(board, r, c)
            out[str(want)] += 1
            if want != got:
                bad.append([r, c, want, got])
    return {'reasons': dict(out), 'mismatches': bad}


def trace_task(task: dict) -> dict:
    """V8-C's check of the tree move with every forbidden_reason call compared (Python result used)."""
    from hybrid.h6_fast import install
    from renju import rules
    from rt30.native._forbidden import forbidden
    from scripts import h6_5d
    from search import mcts

    install('fast3')
    calls, bad = Counter(), []

    def both(board, row, col):
        want = rules.forbidden_reason(board, row, col)
        got = forbidden(board, row, col)
        calls[str(want)] += 1
        if want != got and len(bad) < 5:
            bad.append({'board': [list(x) for x in board], 'cell': [row, col], 'python': want, 'native': got})
        return want

    saved = mcts.forbidden_reason
    mcts.forbidden_reason = both
    try:
        status = h6_5d.check_status(task['moves'], task['tree_move'], h6_5d.STANDARD, table=True)
    finally:
        mcts.forbidden_reason = saved
    return {'id': task['id'], 'status': status['status'], 'nodes': status['nodes'], 'calls': dict(calls),
            'mismatches': bad}


def _boards(task):
    from renju import Game

    game = Game()
    for move in task['moves']:
        game.play(*move)
    return check_board(game.board)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='h7_timing measure: the tree move per position')
    parser.add_argument('--rule-corpus', type=int, default=3000, help='positions of tests/rule_validation.py')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    load = (lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    positions = load(args.positions)['positions'][:args.limit]
    keep = set(load(args.corpus)['ids'])
    tree = {r['id']: r['decision']['tree_move'] for r in load(args.baseline)['rows']
            if r['decision']['route'] == 'tree' and r['decision']['tree_move']}
    traces = [{'id': p['id'], 'moves': p['moves'], 'tree_move': tree[p['id']]} for p in positions
              if p['id'] in keep and p['id'] in tree]
    from hybrid.progress import Progress, run_chunks
    from rule_validation import positions as rule_positions

    cells = {'boards': 0, 'cells': Counter(), 'mismatches': []}

    def add_cells(result):
        cells['boards'] += 1
        cells['cells'].update(result['reasons'])
        cells['mismatches'].extend(result['mismatches'][:5])

    for _, game in rule_positions(7, args.rule_corpus):
        add_cells(check_board(game.board))
    progress = Progress('h7-p1a cells', len(positions), 'positions')
    trace_rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=get_context('spawn')) as pool:
        run_chunks(pool, _chunk_boards, [[p] for p in positions], progress, len, unit_name='positions',
                   on_result=lambda _, rows: [add_cells(r) for r in rows])
        progress = Progress('h7-p1a trace', len(traces), 'positions')
        run_chunks(pool, _chunk_traces, [[t] for t in traces], progress, len, unit_name='positions',
                   on_result=lambda _, rows: trace_rows.extend(rows))
    trace_calls = Counter()
    for r in trace_rows:
        trace_calls.update(r['calls'])
    trace_bad = [m for r in trace_rows for m in r['mismatches']]
    summary = {'cells': {'boards': cells['boards'], 'checked': sum(cells['cells'].values()),
                         'reasons': dict(cells['cells']), 'mismatches': len(cells['mismatches'])},
               'trace': {'positions': len(trace_rows), 'calls': sum(trace_calls.values()), 'reasons': dict(trace_calls),
                         'mismatches': len(trace_bad)}}
    summary['verdict'] = 'PASS' if not cells['mismatches'] and not trace_bad else 'FAIL'
    result = {'format': FORMAT, 'summary': summary, 'cell_mismatches': cells['mismatches'][:20],
              'trace_mismatches': trace_bad[:20], 'traces': trace_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if summary['verdict'] == 'PASS' else 1


def _chunk_boards(tasks):
    return [_boards(t) for t in tasks]


def _chunk_traces(tasks):
    return [trace_task(t) for t in tasks]


if __name__ == '__main__':
    raise SystemExit(main())
