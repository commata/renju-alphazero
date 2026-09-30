"""V8 functional gates on the fixed probe sets (docs/mcts-v8-teacher.md §7).

Currently: V8-A on ``must_defend_vct``. A probe passes only if the V8 move is in
``correct_moves``, outside ``avoid_moves`` AND V8 itself proved it VCT1-SAFE (a
fixture-correct move reached by fallback does not count). Budget exhaustion is
reported separately. Prints one row per probe and a summary with the time
distribution; exits 1 on any failure.

    python scripts/run_v8_gates.py [--symmetries 0 1 ...] [--output runs/v8_gates.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics, mcts_search_v8  # noqa: E402
from renju import Game  # noqa: E402

PROBES = ROOT / 'tests/fixtures/vct_probes_v1.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--symmetries', type=int, nargs='*', default=None,
                        help='D4 indices to run (default: all 8)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    probes = [p for p in json.loads(PROBES.read_text(encoding='utf-8'))['probes']
              if p['kind'] == 'must_defend_vct'
              and (args.symmetries is None or p['symmetry'] in args.symmetries)]
    rows, failures = [], 0
    for probe in probes:
        game = Game()
        for move in probe['moves']:
            game.play(*move)
        diag = SearchDiagnostics()
        started = perf_counter()
        move = list(mcts_search_v8(game, **V8_DEFAULTS, random=Random(args.seed), diagnostics=diag))
        checked = {tuple(m): status for m, status in diag.v8_vct_checked}
        proven = checked.get(tuple(move)) == 'SAFE'
        ok = move in probe['correct_moves'] and move not in probe['avoid_moves'] and proven
        failures += not ok
        row = {'id': probe['id'], 'ok': ok, 'proven_safe': proven, 'move': move, 'v7_move': list(diag.v8_v7_move),
               'route': diag.v8_route, 'vct_calls': diag.v8_vct_calls, 'vct_nodes': diag.v8_vct_nodes,
               'budget_exhausted': diag.v8_vct_budget_exhausted,
               'checked': [[list(m), s] for m, s in diag.v8_vct_checked],
               'seconds': round(perf_counter() - started, 2)}
        rows.append(row)
        print(f"{'PASS' if ok else 'FAIL'} {row['id']:<24} v8 {move} v7 {row['v7_move']} "
              f"{row['route']} calls {row['vct_calls']} nodes {row['vct_nodes']} "
              f"{'EXHAUSTED ' if row['budget_exhausted'] else ''}{row['seconds']}s", flush=True)
    seconds = sorted(r['seconds'] for r in rows)
    nodes = sum(r['vct_nodes'] for r in rows)
    summary = {'gate': 'must_defend_vct', 'config': {k: V8_DEFAULTS[k] for k in
               ('stage_vct_safety', 'vct_vcf_node_limit', 'vct_call_limit', 'vct_node_budget')},
               'passed': len(rows) - failures, 'total': len(rows),
               'proven_safe': sum(r['proven_safe'] for r in rows),
               'budget_exhausted': sum(r['budget_exhausted'] for r in rows),
               'mean_seconds': round(sum(seconds) / len(seconds), 2) if seconds else 0.0,
               'median_seconds': seconds[len(seconds) // 2] if seconds else 0.0,
               'p95_seconds': seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))] if seconds else 0.0,
               'max_seconds': seconds[-1] if seconds else 0.0,
               'nodes_per_second': round(nodes / sum(seconds)) if seconds and sum(seconds) else 0,
               'rows': rows}
    print(f"must_defend_vct: {summary['passed']}/{summary['total']} passed, proven SAFE "
          f"{summary['proven_safe']}, budget exhausted {summary['budget_exhausted']}; "
          f"seconds mean {summary['mean_seconds']} median {summary['median_seconds']} "
          f"p95 {summary['p95_seconds']} max {summary['max_seconds']}; "
          f"{summary['nodes_per_second']} VCF nodes/s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
