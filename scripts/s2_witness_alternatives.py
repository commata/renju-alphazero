"""S2: were the policy's preferred moves safe where V8 played a depth-2 losing move? (§12.22)

For each VCT2 witness position (``s2_witness_positions.json``; by default the CONFIRMED ones)
the decisive move V8 played and the policy's top ``--top`` moves there (from
``s2_policy_diag.py`` output) are each checked at depth 0, 1, 2 with the bounded solver of
``s1_loss_analysis`` (node/call budget, no time cut). ``SAFE`` means no loss found within
depth 2 (NO_VCT2_FOUND_WITHIN_HORIZON), not a proven defence.

    python scripts/s2_witness_alternatives.py --policy-diag runs/s2/policy_diag.json \\
        --output runs/s2/witness_alternatives.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402
from scripts.s1_loss_analysis import _lost_depth  # noqa: E402

DEFAULT_WITNESS = ROOT / 'docs' / 'mcts-v8-results' / 's2_witness_positions.json'


def check(moves, move_1idx, budget) -> dict:
    started = perf_counter()
    history = [tuple(m) for m in moves]
    depth, status = _lost_depth([*history, (move_1idx[0] - 1, move_1idx[1] - 1)], len(history), budget)
    return {'move': list(move_1idx), 'lost_depth': depth,
            'status': 'PROVEN_LOSS' if depth is not None else status,
            'seconds': round(perf_counter() - started, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--witness', type=Path, default=DEFAULT_WITNESS)
    parser.add_argument('--policy-diag', type=Path, required=True)
    parser.add_argument('--top', type=int, default=2, help='policy top-N moves checked per position')
    parser.add_argument('--all', action='store_true', help='TENTATIVE positions too')
    parser.add_argument('--node-limit', type=int, default=20_000)
    parser.add_argument('--call-limit', type=int, default=50_000)
    parser.add_argument('--node-budget', type=int, default=3_000_000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    budget = {'node_limit': args.node_limit, 'call_limit': args.call_limit, 'node_budget': args.node_budget}
    witness = json.loads(args.witness.read_text(encoding='utf-8'))['positions']
    diag = {(w['seed'], w['pair'], w['v8_color']): w
            for w in json.loads(args.policy_diag.read_text(encoding='utf-8'))['witness']}
    rows = []
    for pos in witness:
        if pos['causal_status'] != 'CONFIRMED' and not args.all:
            continue
        policy = diag[(pos['seed'], pos['pair'], pos['v8_color'])]
        top = [tuple(m) for m, _ in policy['top'][:args.top]]
        played = tuple(pos['decisive_move_1idx'])
        checks = []
        for rank, move in enumerate(top, 1):
            checks.append({'policy_rank': rank, 'prob': policy['top'][rank - 1][1], 'played': move == played,
                           **check(pos['moves'], move, budget)})
            print(pos['seed'], pos['pair'], checks[-1], flush=True)
        if played not in top:
            info = policy['moves'][f'{played[0]},{played[1]}']
            checks.append({'policy_rank': info['rank'], 'prob': info['prob'], 'played': True,
                           **check(pos['moves'], played, budget)})
            print(pos['seed'], pos['pair'], checks[-1], flush=True)
        rows.append({key: pos[key] for key in ('seed', 'pair', 'v8_color', 'ply', 'decisive_move_1idx',
                                                'causal_status', 'engine_status')} | {'checks': checks})
    payload = {'format': 's2-witness-alternatives-v1', 'git_commit': _git_commit(), 'budget': budget,
               'inputs': {str(args.witness): file_sha256(args.witness),
                          str(args.policy_diag): file_sha256(args.policy_diag)},
               'positions': rows}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
