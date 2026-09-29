"""Locate where an AI lost a web-play game, using bounded VCF proofs (read-only).

For every AI move of a finished ``logs/web_play/<game>/moves.csv`` this replays the
game and counts the AI's *VCF-safe* moves at that decision: legal moves after which
the opponent has no VCF. An AI four is answered by the opponent's forced block
before the opponent VCF is checked (the AI may chain fours). The last AI decision
with at least one safe move is the latest point the game could still be saved
against a VCF; a decision with zero safe moves was already lost.

Limits: VCF only (four-only sequences). A loss through a three-based threat (VCT)
shows up as "all safe" at one AI decision and "zero safe" at the next one. Results
depend on ``--node-limit``; an exhausted probe counts as no VCF found.

    python scripts/analyze_web_play_losses.py logs/web_play --tail 8
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from renju import Game  # noqa: E402
from search.mcts_v5 import _winning_moves  # noqa: E402
from search.mcts_v7 import _find_vcf_with_stats  # noqa: E402

MAX_FOUR_CHAIN = 6


def opponent_wins(game: Game, *, node_limit: int, depth: int = 0) -> bool:
    """After our move (opponent to move): does the opponent have a forced VCF win?"""
    if game.done:
        return game.winner == game.to_play
    opponent = game.to_play
    if _winning_moves(game, opponent):
        return True
    ours = _winning_moves(game, -opponent)
    if len(ours) >= 2:
        return False
    if len(ours) == 1:
        block = ours[0]
        if block not in game.legal_moves():
            return False  # the only block is a black forbidden point
        if depth >= MAX_FOUR_CHAIN:
            return False
        game.play(*block)
        try:
            return not safe_moves(game, node_limit=node_limit, depth=depth + 1, limit=1)
        finally:
            game.undo()
    found, _, _ = _find_vcf_with_stats(game, opponent, max_fours=20, node_limit=node_limit)
    return found is not None


def safe_moves(game: Game, *, node_limit: int, depth: int = 0,
               limit: int | None = None) -> list[tuple[int, int]]:
    result = []
    for move in game.legal_moves():
        game.play(*move)
        try:
            ok = (game.done and game.winner is not None) or not opponent_wins(
                game, node_limit=node_limit, depth=depth)
        finally:
            game.undo()
        if ok:
            result.append(move)
            if limit is not None and len(result) >= limit:
                break
    return result


def stage_label(row: dict) -> str:
    if row.get('forced_policy_stage'):
        return f"stage{row['forced_policy_stage']}"
    if row.get('v7_own_vcf_found') == 'True':
        return 'own_vcf'
    return row.get('simulation_mode') or '?'


def analyze(game_dir: Path, *, tail: int, node_limit: int) -> None:
    with open(game_dir / 'moves.csv', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    ai_plies = [int(r['ply']) for r in rows if r['actor'] not in ('HUMAN', 'OPENING_RULE')]
    first = ai_plies[-tail] if len(ai_plies) >= tail else (ai_plies[0] if ai_plies else 0)
    print(f'== {game_dir.name}')
    game = Game()
    for row in rows:
        ply = int(row['ply'])
        if ply in ai_plies and ply >= first:
            chosen = (int(row['row0']), int(row['col0']))
            safe = safe_moves(game, node_limit=node_limit)
            marker = 'chosen-safe' if chosen in safe else ('LOST' if not safe else 'chosen-UNSAFE')
            shown = ' '.join(f'({r + 1},{c + 1})' for r, c in safe[:8])
            print(f"  ply {ply:>3} {stage_label(row):<9} sims={row.get('selected_simulations') or 0:>3} "
                  f"move=({row['row']},{row['col']}) safe={len(safe):>3} {marker:<13} {shown}", flush=True)
        game.play(int(row['row0']), int(row['col0']))
    print(f'  winner={game.winner}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('path', type=Path, help='a web-play game directory or a directory of them')
    parser.add_argument('--tail', type=int, default=8, help='analyze the last N AI moves')
    parser.add_argument('--node-limit', type=int, default=20000)
    parser.add_argument('--losses-only', action='store_true', help='skip games the AI won')
    args = parser.parse_args()
    dirs = [args.path] if (args.path / 'moves.csv').exists() else sorted(
        p for p in args.path.iterdir() if (p / 'moves.csv').exists())
    for game_dir in dirs:
        if args.losses_only and 'HUMAN_WIN' not in (game_dir / 'game.json').read_text(encoding='utf-8'):
            continue
        analyze(game_dir, tail=args.tail, node_limit=args.node_limit)


if __name__ == '__main__':
    main()
