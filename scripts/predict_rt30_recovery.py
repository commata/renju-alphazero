"""How much of TB-Engine-v1 fits a move budget if the engine were faster, or ran V8-B in parallel (docs/web-ser5.md §9.7).

Two inputs, both TB-Engine-v1's own phase times (attack = V8-B, vct = V8-A, tree, root = V8-C, vct2):

- ``--divergence``: a ``run_rt30_divergence.py --engine tb1`` output (SER5 times at the A3 divergence moves);
- the E3-A records (desktop times; scaled to SER5 by the median desktop / SER5 ratio of the divergence
  run's moves, which ``--divergence`` provides).

For an exact speed-up ``s`` (every phase ``s`` times faster, moves unchanged) a move "fits" when its
time is within ``--budget`` (the deadline minus the final margin), in three schedules:

    v1_timing   TB-RT30-v1's schedule (X1 alone): V8-B capped at 8 s, V8-C done by 22 s (the VCT2
                reserve), the move by the budget
    sequential  TB-Engine-v1's order with no V8-B cap: the sum of the phases (an engine with only the
                move deadline; not a proposal, the cap protects the moves where V8-B finds nothing)
    x2          V8-B in its own process beside the rest (X2): max(V8-B, others + tree + V8-C + VCT2);
                a V8-B win ends the move when V8-B ends
    x2_x3       X2 plus VCT2 started on the tree move beside V8-C (X3): VCT2 stays serial only when
                V8-C switched the move

A move that fits is TB-Engine-v1's move (the changes are exact), so for the A3 divergence moves the
report also gives the A3 points those moves carry (``points``: E3-A minus A3 result of the games whose
first divergence fits; an upper bound on what A3 gets back, since a game can split again later). For
all E3-A moves it counts the moves over the budget: those are where a run can still leave TB1's game.
This is a bound on the time side only.

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


SCHEDULES = ('v1_timing', 'sequential', 'x2', 'x2_x3')
CAP, ROOT_END = 8.0, 22.0  # TB-RT30-v1: attack_seconds; deadline - vct2_reserve - final_margin


def times(seconds, phases: dict, switched: bool, own_vct: bool, budget: float) -> dict:
    """Move time per schedule, for a speed-up of 1 (see the module docstring). ``v1_timing`` is a
    factor: the move fits at speed-up s when the factor / s <= 1."""
    attack, vct, tree, root, vct2 = (phases.get(k, 0.0) for k in ('attack', 'vct', 'tree', 'root', 'vct2'))
    other = max(0.0, seconds - attack - vct - tree - root - vct2)
    if own_vct:
        v1 = attack / CAP
        return {'v1_timing': v1, 'sequential': seconds, 'x2': attack, 'x2_x3': attack}
    v1 = max(attack / CAP, (other + attack + vct + tree + root) / ROOT_END, seconds / budget)
    main = other + vct + tree + root
    x3_main = main + (vct2 if switched else 0.0)
    x3_side = 0.0 if switched else other + tree + vct2
    return {'v1_timing': v1, 'sequential': seconds, 'x2': max(attack, main + vct2), 'x2_x3': max(attack, x3_main, x3_side)}


def record_phases(move: dict) -> dict:
    return {'attack': move['attack']['seconds'], 'vct': move['vct']['seconds'], 'tree': move['tree']['seconds'],
            'root': move['root']['seconds'], 'vct2': move['vct2']['seconds']}


SCORE = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}


def delta(result: str) -> float:
    """E3-A minus A3 points of a game ('win>loss' -> 1.0)."""
    before, after = result.split('>')
    return SCORE[before] - SCORE[after]


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
        cases.append({'kind': row['kind'], 'delta': delta(row['result']), 'ratio': record['seconds'] / row['seconds'],
                      'times': times(row['seconds'], row['phases'], switched, row['route'] == 'own_vct', args.budget)})
    ratio = statistics.median(c['ratio'] for c in cases)

    moves = []
    for path in args.jsonl:
        for text in Path(path).read_text(encoding='utf-8').splitlines():
            for move in json.loads(text)['v8_moves']:
                phases = {k: v / ratio for k, v in record_phases(move).items()}
                moves.append(times(move['seconds'] / ratio, phases, bool(move['root'].get('switch')),
                                   move['route'] == 'own_vct', args.budget))

    out = {'budget': args.budget, 'desktop_per_ser5': round(ratio, 2), 'e3a_moves': len(moves),
           'a3_points_at_stake': round(sum(c['delta'] for c in cases), 2), 'rows': []}
    for s in SPEEDUPS:
        row = {'speedup': s}
        for name in SCHEDULES:
            limit = 1.0 if name == 'v1_timing' else args.budget
            fits = [c for c in cases if c['times'][name] / s <= limit]
            row[name] = {
                'e3a_over_budget': sum(t[name] / s > limit for t in moves),
                'attack_fit': f"{sum(c['kind'] == 'attack' for c in fits)}/{sum(c['kind'] == 'attack' for c in cases)}",
                'root_fit': f"{sum(c['kind'] == 'root' for c in fits)}/{sum(c['kind'] == 'root' for c in cases)}",
                'result_changed_fit': f"{sum(c['delta'] != 0 for c in fits)}/{sum(c['delta'] != 0 for c in cases)}",
                'points': round(sum(c['delta'] for c in fits), 2),
            }
        out['rows'].append(row)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
