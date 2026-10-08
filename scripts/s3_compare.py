"""S3-VCT2: paired comparison of puct_policy_vct2 against puct_policy (docs/mcts-v8-teacher.md §12.24).

Both arms play the same openings and seeds against ``v8:full``. Reported per arm (via
``summarize_h5.summarize``) and between arms:

- end-to-end cost ratio R = sum of V8 move seconds (vct2) / sum of V8 move seconds (baseline),
  plus per-move median and p95 of each arm and the game-time ratio;
- paired score difference over opening pairs (bootstrap 95%);
- S3-VCT2 mechanism: moves checked, played moves proven lost, switches.

Decision (fixed before the runs, §12.24):
    ADOPT   R <= 1.5, both arms' time ratio vs the opponent <= 1.5, no safety violation,
            paired score diff >= -0.05
    OPTIMIZE (go to (c))  everything but R <= 1.5
    REJECT  score diff < -0.05 or a safety violation

    python scripts/s3_compare.py --baseline runs/s3/base_8411.json runs/s3/base_8412.json \\
        --arm runs/s3/vct2_8411.json runs/s3/vct2_8412.json --output runs/s3/s3_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'scripts', ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from summarize_h5 import merge, paired, summarize  # noqa: E402

MAX_RATIO = 1.5
MIN_DIFF = -0.05
ARMS = ('puct_policy', 'puct_policy_vct2')


def _seconds(arm: dict) -> list[float]:
    return [m['seconds'] for g in arm['games'] for m in g['v8_moves']]


def _dist(values: list[float]) -> dict:
    values = sorted(values)
    return {'n': len(values), 'median': round(median(values), 3),
            'p95': round(values[min(len(values) - 1, int(0.95 * len(values)))], 3), 'max': round(values[-1], 3)}


def mechanism(arm: dict) -> dict:
    moves = [m for g in arm['games'] for m in g['v8_moves'] if m.get('vct2', {}).get('checked')]
    return {'moves_checked': len(moves),
            'played_move_proven_lost': sum(m['vct2']['checked'][0][1] == 'UNSAFE' for m in moves),
            'switched': sum(m['vct2']['switched'] for m in moves),
            'seconds': _dist([m['vct2']['seconds'] for m in moves]) if moves else None,
            'nodes_median': median(m['vct2']['nodes'] for m in moves) if moves else None}


def compare(base_runs: list[dict], arm_runs: list[dict], arms=ARMS) -> dict:
    base_arms, arm_arms = merge(base_runs), merge(arm_runs)
    if len(base_arms) != 1 or len(arm_arms) != 1:
        raise ValueError('give one arm per side')
    base, arm = next(iter(base_arms.values())), next(iter(arm_arms.values()))
    if (base['arm'], arm['arm']) != tuple(arms):
        raise ValueError(f"expected arms {arms}, got {(base['arm'], arm['arm'])}")
    if sorted(base['seeds']) != sorted(arm['seeds']):
        raise ValueError(f"seeds differ: {base['seeds']} vs {arm['seeds']}")
    rows = {'baseline': summarize(base), 'vct2': summarize(arm)}
    ratio = round(sum(_seconds(arm)) / sum(_seconds(base)), 3)
    game_ratio = round(sum(g['game_seconds'] for g in arm['games']) / sum(g['game_seconds'] for g in base['games']), 3)
    diff = paired(arm, base)
    cost = {'end_to_end_ratio': ratio, 'game_time_ratio': game_ratio,
            'move_seconds': {'baseline': _dist(_seconds(base)), 'vct2': _dist(_seconds(arm))}}
    reasons = []
    if any(r['safety_violations'] for r in rows.values()):
        reasons.append('safety violation')
    if diff['diff'] is None or diff['diff'] < MIN_DIFF:
        reasons.append(f'paired score diff {diff["diff"]} < {MIN_DIFF}')
    decision = 'REJECT' if reasons else None
    if decision is None:
        slow = [f'end-to-end ratio {ratio} > {MAX_RATIO}'] if ratio > MAX_RATIO else []
        slow += [f'{k} time ratio {r["time_ratio"]} > {MAX_RATIO}' for k, r in rows.items()
                 if r['time_ratio'] is not None and r['time_ratio'] > MAX_RATIO]
        decision, reasons = ('OPTIMIZE', slow) if slow else ('ADOPT', [])
    return {'rows': rows, 'paired_diff_vct2_minus_baseline': diff, 'cost': cost,
            'mechanism': mechanism(arm), 'decision': {'decision': decision, 'reasons': reasons}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--baseline', type=Path, nargs='+', required=True)
    parser.add_argument('--arm', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))  # noqa: E731
    result = compare([load(p) for p in args.baseline], [load(p) for p in args.arm])
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
