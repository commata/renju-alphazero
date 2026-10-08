"""S1: does each arm's P93 move keep white's proven win? (docs/mcts-v8-teacher.md §12.20)

P93 (white to move after black's 93rd move (3,4)) is a proven VCT2 win for white. For every
distinct move an arm played there (from ``run_s1_probes.py`` outputs), the position after it
is classified with ``ThreatSolver``: black's replies at depth 1, stopping at the first SAFE
reply. All replies UNSAFE means the move keeps a proven win (white's move is a VCT2 witness);
a SAFE reply is a refutation within this class (the win may still exist deeper). The solver
is deterministic (node limit per VCF call, no time cut).

    python scripts/s1_verify_p93.py runs/s1/probes_full.json runs/s1/probes_puct_heur.json \\
        runs/s1/probes_puct_policy.json --output runs/s1/p93_verification.json
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

from analysis.threats import ThreatSolver, decision_status  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit  # noqa: E402
from scripts.run_s1_probes import DEFAULT_PROBES, replay  # noqa: E402

FORMAT = 's1-p93-verification-v1'


def classify_move(moves, plies: int, move_1idx, node_limit: int) -> dict:
    """Black's best status after white plays ``move_1idx`` (1-indexed) at P93."""
    game = replay(moves, plies)
    game.play(move_1idx[0] - 1, move_1idx[1] - 1)
    solver = ThreatSolver(node_limit=node_limit)
    started = perf_counter()
    counts, per_move = solver.decision(game, 1, stop_at_safe=True)
    status = decision_status(counts)
    refutation = next(([m[0] + 1, m[1] + 1] for m, (s, _) in per_move.items() if s == 'SAFE'), None)
    return {'move': list(move_1idx), 'black_best_status': status, 'keeps_proven_win': status == 'UNSAFE',
            'refutation': refutation, 'replies_checked': sum(counts.values()),
            'seconds': round(perf_counter() - started, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('probe_results', type=Path, nargs='+')
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--node-limit', type=int, default=20_000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    probe_file = json.loads(args.probes.read_text(encoding='utf-8'))
    moves = [tuple(m) for m in probe_file['moves']]
    plies = probe_file['probes']['P93']['plies_played']
    played = {}
    for path in args.probe_results:
        for run in json.loads(path.read_text(encoding='utf-8'))['runs']:
            if run['probe'] == 'P93':
                played.setdefault(run['arm'], []).append(tuple(run['played']))
    verdicts = {}
    for move in sorted({m for runs in played.values() for m in runs}):
        verdicts[move] = classify_move(moves, plies, move, args.node_limit)
        print(json.dumps(verdicts[move]), flush=True)
    arms = {arm: {'runs': len(runs), 'kept_proven_win': sum(verdicts[m]['keeps_proven_win'] for m in runs),
                  'played': {f'{m[0]},{m[1]}': runs.count(m) for m in sorted(set(runs))}}
            for arm, runs in sorted(played.items())}
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'node_limit': args.node_limit,
               'inputs': [str(p) for p in args.probe_results], 'arms': arms, 'moves': list(verdicts.values())}
    print(json.dumps(arms, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
