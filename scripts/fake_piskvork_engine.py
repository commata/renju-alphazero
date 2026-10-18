"""A tiny Piskvork engine for tests and smoke runs of ``scripts/run_external_match.py`` (docs/external-engine.md).

It plays legal moves by this project's rules (a seeded random move next to the stones), and can
misbehave on purpose so the adapter's failure paths are tested without a real engine:

    --mode legal       (default) a legal move
    --mode forbidden   as black, a forbidden point when one exists (the rule-mismatch path)
    --mode silent      never answers BOARD (the timeout path)
    --mode crash       exits on BOARD
    --mode garbage     answers BOARD with a non-move line
    --forbid-lie       YXSHOWFORBID reports no forbidden points (the cross-check mismatch path)
    --log FILE         append every received line (protocol tests)

    python scripts/fake_piskvork_engine.py --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from renju import BLACK, Game  # noqa: E402
from renju.rules import SIZE  # noqa: E402

from agents.external_engine import our_forbidden_points  # noqa: E402


def say(line: str) -> None:
    print(line, flush=True)


def position(lines: list[str]) -> Game:
    """Our ``Game`` from a BOARD body (stones in move order; the first stone is black)."""
    game = Game()
    for line in lines:
        x, y, _field = (int(v) for v in line.split(','))
        game.play(y, x)
    return game


def choose(game: Game, mode: str, random: Random) -> tuple[int, int]:
    if mode == 'forbidden' and game.to_play == BLACK:
        forbidden = sorted(our_forbidden_points(game))
        if forbidden:
            return forbidden[0]
    legal = game.legal_moves()
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if game.board[r][c]]
    near = [m for m in legal if any(max(abs(m[0] - r), abs(m[1] - c)) <= 1 for r, c in stones)]
    return random.choice(near or legal)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', choices=('legal', 'forbidden', 'silent', 'crash', 'garbage'), default='legal')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--forbid-lie', action='store_true')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args(argv)
    log = args.log.open('a', encoding='utf-8') if args.log else None
    body: list[str] | None = None
    last: list[str] = []
    for raw in sys.stdin:
        line = raw.strip()
        if log:
            log.write(line + '\n')
            log.flush()
        if body is not None:
            if line.upper() == 'DONE':
                last, think, body = body, think_after, None
                if not think:
                    continue
                if args.mode == 'silent':
                    continue
                if args.mode == 'crash':
                    return 3
                if args.mode == 'garbage':
                    say('hello')
                    continue
                game = position(last)
                say('MESSAGE fake engine thinking')
                row, col = choose(game, args.mode, Random(args.seed * 1000 + len(last)))
                say(f'{col},{row}')
            else:
                body.append(line)
            continue
        word = line.split(' ', 1)[0].upper()
        if word == 'START':
            say('OK' if line.split()[1:] == [str(SIZE)] else 'ERROR unsupported size')
        elif word == 'RESTART':
            say('OK')
        elif word == 'INFO':
            pass
        elif word == 'ABOUT':
            say('name="fake", version="1"')
        elif word in ('BOARD', 'YXBOARD'):
            body, think_after = [], word == 'BOARD'
        elif word == 'YXSHOWFORBID':
            game = position(last)
            points = [] if args.forbid_lie or game.to_play != BLACK else sorted(our_forbidden_points(game))
            say('FORBID ' + ''.join(f'{c:02d}{r:02d}' for r, c in points) + '.')
        elif word == 'END':
            return 0
        elif line:
            say(f'UNKNOWN {line}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
