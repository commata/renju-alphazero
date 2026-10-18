"""Rule cross-check: our black forbidden points against an external engine's (docs/external-engine.md §6).

Plays seeded random games (each move next to the stones, ``--radius``, so threes and fours form often)
with our ``Game``, and at every black-to-move position compares ``our_forbidden_points`` with the
engine's ``YXSHOWFORBID`` answer. Every difference is written with its move list for a manual check
against the RIF rules (docs/rules.md); no difference is fixed automatically in either engine.
No match is played and nothing here is training data.

    python scripts/external_rule_crosscheck.py --engine C:/engines/rapfi/pbrain-rapfi.exe \\
        --engine-name rapfi --games 2000 --seed 1 --output runs/external/rule_crosscheck.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.external_engine import EngineConfig, PiskvorkEngine, our_forbidden_points  # noqa: E402
from hybrid.progress import Progress  # noqa: E402
from renju import BLACK, Game  # noqa: E402
from renju.rules import SIZE, forbidden_reason  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty  # noqa: E402

FORMAT = 'external-rule-crosscheck-v1'


def random_move(game: Game, random: Random, radius: int) -> tuple[int, int]:
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if game.board[r][c]]
    legal = game.legal_moves()
    near = [m for m in legal if any(max(abs(m[0] - r), abs(m[1] - c)) <= radius for r, c in stones)]
    return random.choice(near or legal)


def check_game(engine: PiskvorkEngine, random: Random, radius: int) -> tuple[int, int, list[dict]]:
    """(positions compared, forbidden points seen, mismatches) for one random game."""
    game, compared, seen, mismatches = Game(), 0, 0, []
    while not game.done:
        if game.to_play == BLACK and game.history:
            ours, theirs = our_forbidden_points(game), engine.forbidden_points(game.history)
            compared, seen = compared + 1, seen + len(ours)
            if ours != theirs:
                mismatches.append({
                    'moves': [list(m) for m in game.history],
                    'ours_only': [[*m, forbidden_reason(game.board, *m)] for m in sorted(ours - theirs)],
                    'engine_only': sorted(map(list, theirs - ours))})
        game.play(*random_move(game, random, radius))
    return compared, seen, mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--engine', required=True)
    parser.add_argument('--engine-arg', action='append', default=[])
    parser.add_argument('--engine-name', default='engine')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error('--games must be positive')
    config = EngineConfig(command=(args.engine, *args.engine_arg), name=args.engine_name)
    random = Random(args.seed)
    compared = seen = 0
    mismatches: list[dict] = []
    progress = Progress('rule-crosscheck', args.games, 'games')
    with PiskvorkEngine(config) as engine:
        about = engine.about()
        print(f'engine={config.label} ({about}) games={args.games} seed={args.seed}', flush=True)
        for _ in range(args.games):
            n, s, found = check_game(engine, random, args.radius)
            compared, seen = compared + n, seen + s
            mismatches.extend(found)
            progress.update(1, extra=f'{compared:,} positions, {len(mismatches)} mismatches')
    summary = {'games': args.games, 'positions': compared, 'our_forbidden_points': seen,
               'mismatch_positions': len(mismatches)}
    print(json.dumps(summary, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format': FORMAT, 'seed': args.seed, 'radius': args.radius, 'git_commit': _git_commit(),
            'git_dirty': _git_dirty(), 'engine': {**config.provenance(), 'about': about},
            'summary': summary, 'mismatches': mismatches}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
