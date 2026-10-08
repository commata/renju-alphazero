"""S2: raw H3 policy at the probe and witness positions (docs/mcts-v8-teacher.md §12.21).

Needs torch (the H3 checkpoint). For a probe position (default P92) the whole legal
distribution is saved, so it can be joined later with ``s2_position_truth.py`` (rank and
probability of every saving move). For each VCT2 witness position (``--witness``, from
``s1_loss_analysis.py --witness-output``) it records the rank and probability of the move
V8 actually played there (the depth-2 losing move) and the policy top 10. Ranks are over all
legal moves (1 = highest probability); coordinates are 1-indexed.

    python scripts/s2_policy_diag.py --policy-checkpoint runs/h3_policy_64x4/best.pt \\
        --witness docs/mcts-v8-results/s2_witness_positions.json --output runs/s2/policy_diag.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.run_mcts_v8_benchmark import _git_commit, load_policy, provenance  # noqa: E402
from scripts.run_s1_probes import DEFAULT_PROBES, replay  # noqa: E402


def ranked(probs: dict) -> list[tuple]:
    return sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))


def describe(probs: dict, moves_1idx=(), top: int = 10) -> dict:
    order = ranked(probs)
    rank = {m: i + 1 for i, (m, _) in enumerate(order)}
    picks = {}
    for r, c in moves_1idx:
        move = (r - 1, c - 1)
        picks[f'{r},{c}'] = {'rank': rank.get(move), 'prob': round(probs.get(move, 0.0), 6)}
    return {'legal': len(order), 'top': [[[m[0] + 1, m[1] + 1], round(p, 6)] for m, p in order[:top]],
            'moves': picks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--names', nargs='*', default=['P92'])
    parser.add_argument('--witness', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    policy = load_policy(args.policy_checkpoint)
    probe_file = json.loads(args.probes.read_text(encoding='utf-8'))
    payload = {'format': 's2-policy-diag-v1', 'git_commit': _git_commit(),
               'provenance': provenance(args.policy_checkpoint), 'probes': {}, 'witness': []}
    for name in args.names:
        game = replay([tuple(m) for m in probe_file['moves']], probe_file['probes'][name]['plies_played'])
        probs = policy(game)
        payload['probes'][name] = {
            **describe(probs, [(3, 4), (4, 4), (9, 7), (10, 9), (11, 13), (12, 12), (14, 14), (12, 13), (13, 11)]),
            'distribution': [[[m[0] + 1, m[1] + 1], round(p, 6)] for m, p in ranked(probs)],
        }
    if args.witness is not None:
        for pos in json.loads(args.witness.read_text(encoding='utf-8'))['positions']:
            game = replay([tuple(m) for m in pos['moves']], len(pos['moves']))
            payload['witness'].append({key: pos[key] for key in ('seed', 'pair', 'v8_color', 'ply',
                                                                  'decisive_move_1idx', 'causal_status')}
                                      | describe(policy(game), [tuple(pos['decisive_move_1idx'])]))
    print(json.dumps({k: v for k, v in payload.items() if k != 'probes'}
                     | {'probes': {n: {k: v for k, v in p.items() if k != 'distribution'}
                                   for n, p in payload['probes'].items()}}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
