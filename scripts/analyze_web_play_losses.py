"""Locate where an AI lost a web-play game, using bounded VCF search (read-only).

For every analyzed AI decision of a ``logs/web_play/<game>/moves.csv`` this replays
the game and classifies each legal AI move by what the opponent can do next:

- ``UNSAFE``: the opponent has a VCF (found certificate) or an immediate five.
- ``SAFE``: the VCF search finished without a budget cut and found none.
- ``UNKNOWN``: undecided because the node budget ran out or the AI's own four
  chain went deeper than ``--four-chain``.

An AI four is answered by the opponent's forced block; the position after the
block is then classified as a decision of its own (SAFE if any reply is SAFE,
else UNKNOWN if any is UNKNOWN, else UNSAFE). A decision is proven lost against
VCF only when ``safe == 0 and unknown == 0``.

Scope of SAFE: the VCF solver is the frozen V7 one (``search.mcts_v7``). It skips
lines where the defender's forced block itself makes a four (and positions where
the defender already has a winning point), so SAFE means "no VCF of that class",
not "no forced win". Three-based threats (VCT) are not searched at all.
``max_fours`` is set above the number of fours the empty board can hold, so the
solver's only cut is the node budget, which it reports.

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

from renju import EMPTY, Game  # noqa: E402
from search.mcts_v5 import _winning_moves  # noqa: E402
from search.mcts_v7 import _find_vcf_with_stats  # noqa: E402

SAFE, UNKNOWN, UNSAFE = 'SAFE', 'UNKNOWN', 'UNSAFE'
_RANK = {SAFE: 0, UNKNOWN: 1, UNSAFE: 2}


def _unbounded_max_fours(game: Game) -> int:
    # Every four in a VCF line spends an attacker stone and a defender block.
    empties = sum(row.count(EMPTY) for row in game.board)
    return empties // 2 + 1


def after_move_status(game: Game, *, node_limit: int, four_chain: int, depth: int = 0) -> str:
    """Status of the side that just moved (the opponent is to move)."""
    opponent = game.to_play
    if game.done:
        return UNSAFE if game.winner == opponent else SAFE
    if _winning_moves(game, opponent):
        return UNSAFE
    ours = _winning_moves(game, -opponent)
    if len(ours) >= 2:
        return SAFE  # one block cannot stop two legal fives
    if len(ours) == 1:
        block = ours[0]
        if block not in game.legal_moves():
            return SAFE  # the only block is a black forbidden point
        if depth >= four_chain:
            return UNKNOWN
        game.play(*block)
        try:
            counts = decision_counts(game, node_limit=node_limit, four_chain=four_chain,
                                     depth=depth + 1, stop_at_safe=True)
        finally:
            game.undo()
        return decision_status(counts)
    found, _, exhausted = _find_vcf_with_stats(
        game, opponent, max_fours=_unbounded_max_fours(game), node_limit=node_limit)
    if found is not None:
        return UNSAFE
    return UNKNOWN if exhausted else SAFE


def classify_moves(game: Game, *, node_limit: int, four_chain: int, depth: int = 0,
                   stop_at_safe: bool = False) -> dict[tuple[int, int], str]:
    result = {}
    for move in game.legal_moves():
        game.play(*move)
        try:
            result[move] = after_move_status(game, node_limit=node_limit,
                                             four_chain=four_chain, depth=depth)
        finally:
            game.undo()
        if stop_at_safe and result[move] == SAFE:
            break
    return result


def decision_counts(game: Game, **kwargs) -> Counter:
    return Counter(classify_moves(game, **kwargs).values())


def decision_status(counts: Counter) -> str:
    if counts[SAFE]:
        return SAFE
    return UNKNOWN if counts[UNKNOWN] else UNSAFE


def stage_label(row: dict) -> str:
    if row.get('forced_policy_stage'):
        return f"stage{row['forced_policy_stage']}"
    if row.get('v7_own_vcf_found') == 'True':
        return 'own_vcf'
    return row.get('simulation_mode') or '?'


def verdict(chosen_status: str, counts: Counter) -> str:
    if not counts[SAFE] and not counts[UNKNOWN]:
        return 'LOST(VCF)'
    return f'chosen={chosen_status}'


def analyze(game_dir: Path, *, tail: int, node_limit: int, four_chain: int) -> None:
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
            status = classify_moves(game, node_limit=node_limit, four_chain=four_chain)
            counts = Counter(status.values())
            safe = sorted(m for m, s in status.items() if s == SAFE)
            shown = ' '.join(f'({r + 1},{c + 1})' for r, c in safe[:6])
            print(f"  ply {ply:>3} {stage_label(row):<9} sims={row.get('selected_simulations') or 0:>3} "
                  f"opp_win_pts={threats} move=({row['row']},{row['col']}) "
                  f"safe={counts[SAFE]:>3} unknown={counts[UNKNOWN]:>3} unsafe={counts[UNSAFE]:>3} "
                  f"{verdict(status[move], counts):<16} {shown}", flush=True)
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
    parser.add_argument('--losses-only', action='store_true', help='skip games the AI won')
    args = parser.parse_args()
    dirs = [args.path] if (args.path / 'moves.csv').exists() else sorted(
        p for p in args.path.iterdir() if (p / 'moves.csv').exists())
    for game_dir in dirs:
        if args.losses_only:
            meta = json.loads((game_dir / 'game.json').read_text(encoding='utf-8'))
            if meta.get('result') != 'HUMAN_WIN':
                continue
        analyze(game_dir, tail=args.tail, node_limit=args.node_limit, four_chain=args.four_chain)


if __name__ == '__main__':
    main()
