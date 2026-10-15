"""How much of TB-Engine-v1 fits a move budget if the engine were faster, or ran V8-B in parallel (docs/web-ser5.md §9.7).

Two inputs, both TB-Engine-v1's own phase times (attack = V8-B, vct = V8-A, tree, root = V8-C, vct2):

- ``--divergence``: a ``run_rt30_divergence.py --engine tb1`` output (SER5 times at the A3 divergence moves);
- the E3-A records (desktop times; scaled to SER5 by the median desktop / SER5 ratio of the divergence
  run's moves, which ``--divergence`` provides).

For an exact speed-up ``s`` (every phase ``s`` times faster, moves unchanged) a move "fits" when its
time is within ``--budget`` (the deadline minus the final margin):

    sequential  sum of the phases (TB-Engine-v1's order)
    parallel    V8-B in its own process beside the rest: max(V8-B, others + tree + V8-C [+ VCT2 when
                V8-C switched the move; otherwise VCT2 runs beside, on the tree move]); a V8-B win ends
                the move when V8-B ends

A move that fits is TB-Engine-v1's move (both changes are exact). This is a bound on the time side
only; it says nothing about moves a time policy (cap, retry, split) gets right without fitting.

    python scripts/predict_rt30_recovery.py --divergence docs/rt30-results/a31_tb1_w4.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
DEFAULT_JSONL = [E3 / 'puct_policy_vct2_50k_8415.jsonl', E3 / 'puct_policy_vct2_50k_8416.jsonl']
SPEEDUPS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def times(seconds, phases: dict, switched: bool, own_vct: bool) -> tuple[float, float]:
    attack, vct, tree, root, vct2 = (phases.get(k, 0.0) for k in ('attack', 'vct', 'tree', 'root', 'vct2'))
    other = max(0.0, seconds - attack - vct - tree - root - vct2)
    if own_vct:
        return seconds, attack
    main = other + vct + tree + root + (vct2 if switched else 0.0)
    side = (other + tree + vct2) if not switched else 0.0
    return seconds, max(attack, main, side)


def record_phases(move: dict) -> dict:
    return {'attack': move['attack']['seconds'], 'vct': move['vct']['seconds'], 'tree': move['tree']['seconds'],
            'root': move['root']['seconds'], 'vct2': move['vct2']['seconds']}


def changed(result: str) -> bool:
    before, after = result.split('>')
    return before != after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--divergence', type=Path, required=True, help='run_rt30_divergence.py --engine tb1 output')
    parser.add_argument('--jsonl', type=Path, nargs='+', default=DEFAULT_JSONL)
    parser.add_argument('--budget', type=float, default=26.0, help='deadline 27.5 s minus the final margin 1.5 s')
    args = parser.parse_args(argv)

    run = json.loads(args.divergence.read_text(encoding='utf-8'))
    if run['engine'] != 'tb1':
        parser.error('--divergence must be a tb1 run (TB-Engine-v1 without a deadline)')
    lines = {}
    cases = []
    for row in run['rows']:
        stem, line, ply = row['key'].split('/')
        if (stem, line) not in lines:
            lines[(stem, line)] = json.loads((E3 / f'{stem}.jsonl').read_text(encoding='utf-8').splitlines()[int(line)])
        record = next(m for m in lines[(stem, line)]['v8_moves'] if m['ply'] == int(ply))
        switched = row['kind'] != 'attack' and row['move'] != row['tree_move']
        cases.append({'kind': row['kind'], 'changed': changed(row['result']), 'ratio': record['seconds'] / row['seconds'],
                      'times': times(row['seconds'], row['phases'], switched, row['route'] == 'own_vct')})
    ratio = statistics.median(c['ratio'] for c in cases)

    moves = []
    for path in args.jsonl:
        for text in Path(path).read_text(encoding='utf-8').splitlines():
            for move in json.loads(text)['v8_moves']:
                phases = {k: v / ratio for k, v in record_phases(move).items()}
                moves.append(times(move['seconds'] / ratio, phases, bool(move['root'].get('switch')),
                                   move['route'] == 'own_vct'))

    out = {'budget': args.budget, 'desktop_per_ser5': round(ratio, 2), 'e3a_moves': len(moves), 'rows': []}
    for s in SPEEDUPS:
        row = {'speedup': s}
        for i, name in enumerate(('sequential', 'parallel')):
            row[f'e3a_over_budget_{name}'] = sum(t[i] / s > args.budget for t in moves)
            for kind in ('attack', 'root'):
                ks = [c for c in cases if c['kind'] == kind]
                row[f'{kind}_fit_{name}'] = f"{sum(c['times'][i] / s <= args.budget for c in ks)}/{len(ks)}"
                ch = [c for c in ks if c['changed']]
                row[f'{kind}_result_changed_fit_{name}'] = f"{sum(c['times'][i] / s <= args.budget for c in ch)}/{len(ch)}"
        out['rows'].append(row)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
