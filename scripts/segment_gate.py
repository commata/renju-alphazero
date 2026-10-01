"""Stability + promotion gate for one training segment (docs/stage8-plan.md §12.14).

Combines the self-play health of the segment (``analyze_self_play_health``) with the
champion match written by the orchestrator at the segment end
(``<run>/external_eval/genNNN_h2h.json``, the checkpoint vs ``--h2h-anchor``):

- ``STOP``    the health gate is STOP (short-game collapse or replay reuse runaway):
              do not train further on this branch; run forensic_short_games.py;
- ``PROMOTE`` health is not STOP and the checkpoint beats the champion
              (score > 0.55 and pair-level p < 0.05): it becomes the new champion;
- ``HOLD``    otherwise (keep training; the champion stays).

Exit code: 0 PROMOTE, 1 HOLD, 2 STOP, so a PowerShell loop can branch on $LASTEXITCODE.

    python scripts/segment_gate.py runs/stage8_s2_adaptive --from 1120 --to 1160 \
        --reference-reuse 6.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from analyze_self_play_health import analyze  # noqa: E402

PROMOTE_SCORE = 0.55
PROMOTE_P = 0.05
EXIT = {'PROMOTE': 0, 'HOLD': 1, 'STOP': 2}


def decide(health_level: str, h2h: dict | None) -> tuple[str, str]:
    if health_level == 'STOP':
        return 'STOP', 'self-play health STOP'
    if h2h is None:
        return 'HOLD', 'no champion match at the segment end'
    s = h2h['summary']
    p = s.get('p_pairs_two_sided', s['p_two_sided'])
    reason = (f"vs {h2h.get('anchor', {}).get('label', '?')}: {s['a_score']:.3f}, "
              f"pairs {s.get('a_pair_wins', '?')}-{s.get('a_pair_losses', '?')}, p={p:.3g}")
    if s['a_score'] > PROMOTE_SCORE and p < PROMOTE_P:
        return 'PROMOTE', reason
    return 'HOLD', reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--from', dest='start', type=int, required=True)
    parser.add_argument('--to', dest='end', type=int, required=True,
                        help='segment end generation (the checkpoint that was evaluated)')
    parser.add_argument('--reference-reuse', type=float, default=6.4)
    parser.add_argument('--history-from', type=int,
                        help='include earlier windows so the two-window STOP rules can fire '
                             '(default: one segment before --from)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    window = args.end - args.start
    history = args.history_from if args.history_from is not None else args.start - window
    health = analyze(args.run, history, args.end, window, history, args.reference_reuse)
    path = args.run / 'external_eval' / f'gen{args.end:03d}_h2h.json'
    h2h = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    decision, reason = decide(health['gate']['level'], h2h)
    last = health['windows'][-1]
    print(f"segment {args.start}-{args.end}: <=10 plies {last['short_share']:.0%}, "
          f"reuse {last['reuse'] if last['reuse'] is not None else float('nan'):.2f}, "
          f"6-ply entropy {last['prefix6_entropy']:.2f}, health {health['gate']['level']} "
          f"{'; '.join(health['gate']['flags'])}")
    print(f'decision: {decision} ({reason})')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'decision': decision, 'reason': reason,
                                           'health': health}, indent=1), encoding='utf-8')
    return EXIT[decision]


if __name__ == '__main__':
    raise SystemExit(main())
