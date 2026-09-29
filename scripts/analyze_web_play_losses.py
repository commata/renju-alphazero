"""Locate where an AI lost a web-play game, using bounded threat proofs (read-only).

For every analyzed AI decision of a ``logs/web_play/<game>/moves.csv`` this replays
the game and classifies each legal AI move with ``analysis.threats.ThreatSolver``:

- ``UNSAFE``: the opponent has a proven forced win (five, VCF, or with
  ``--vct-depth d`` up to ``d`` quiet threat moves followed by a VCF).
- ``SAFE``: the search finished without a cut and found no such win.
- ``UNKNOWN``: a node budget or four-chain cut prevented a proof.

A decision is proven lost only when ``safe == 0 and unknown == 0``. The solver's
exact scope and caveats are documented in ``analysis/threats.py``.

    python scripts/analyze_web_play_losses.py logs/web_play --losses-only --tail 6
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.threats import SAFE, UNKNOWN, ThreatSolver  # noqa: E402
from renju import Game  # noqa: E402
from search.mcts_v5 import _winning_moves  # noqa: E402


def stage_label(row: dict) -> str:
    if row.get('az_top_visits'):
        return 'alphazero'
    if row.get('forced_policy_stage'):
        return f"stage{row['forced_policy_stage']}"
    if row.get('v7_own_vcf_found') == 'True':
        return 'own_vcf'
    return row.get('simulation_mode') or '?'


def verdict(chosen_status: str, counts: Counter, vct_depth: int) -> str:
    if not counts[SAFE] and not counts[UNKNOWN]:
        return 'LOST(VCF)' if vct_depth == 0 else f'LOST(VCT{vct_depth})'
    return f'chosen={chosen_status}'


def analyze(game_dir: Path, *, tail: int, solver: ThreatSolver, vct_depth: int = 0) -> None:
    with open(game_dir / 'moves.csv', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    ai_plies = [int(r['ply']) for r in rows if r['actor'] not in ('HUMAN', 'OPENING_RULE')]
    first = ai_plies[-tail] if len(ai_plies) >= tail else (ai_plies[0] if ai_plies else 0)
    print(f'== {game_dir.name}')
    game = Game()
    for row in rows:
        ply = int(row['ply'])
        move = (int(row['row0']), int(row['col0']))
        if ply in ai_plies and ply >= first:
            threats = len(_winning_moves(game, -game.to_play))
            _, per_move = solver.decision(game, vct_depth)
            status = {move: result[0] for move, result in per_move.items()}
            counts = Counter(status.values())
            safe = sorted(m for m, s in status.items() if s == SAFE)
            shown = ' '.join(f'({r + 1},{c + 1})' for r, c in safe[:6])
            print(f"  ply {ply:>3} {stage_label(row):<9} sims={row.get('selected_simulations') or 0:>3} "
                  f"opp_win_pts={threats} move=({row['row']},{row['col']}) "
                  f"safe={counts[SAFE]:>3} unknown={counts[UNKNOWN]:>3} unsafe={counts[UNSAFE]:>3} "
                  f"{verdict(status[move], counts, vct_depth):<16} {shown}", flush=True)
        game.play(*move)
    print(f'  winner={game.winner}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('path', type=Path, help='a web-play game directory or a directory of them')
    parser.add_argument('--tail', type=int, default=8, help='analyze the last N AI moves')
    parser.add_argument('--node-limit', type=int, default=100_000,
                        help='VCF node budget per probe; a cut probe is UNKNOWN')
    parser.add_argument('--four-chain', type=int, default=6,
                        help='max own-four chain before a line is UNKNOWN')
    parser.add_argument('--vct-depth', type=int, default=0,
                        help='quiet threat moves allowed to the opponent before its VCF '
                             '(0 = VCF only; 1 is slow, see analysis.threats)')
    parser.add_argument('--losses-only', action='store_true', help='skip games the AI won')
    args = parser.parse_args()
    solver = ThreatSolver(node_limit=args.node_limit, four_chain=args.four_chain)
    dirs = [args.path] if (args.path / 'moves.csv').exists() else sorted(
        p for p in args.path.iterdir() if (p / 'moves.csv').exists())
    for game_dir in dirs:
        if args.losses_only:
            meta = json.loads((game_dir / 'game.json').read_text(encoding='utf-8'))
            if meta.get('result') != 'HUMAN_WIN':
                continue
        analyze(game_dir, tail=args.tail, solver=solver, vct_depth=args.vct_depth)


if __name__ == '__main__':
    main()
