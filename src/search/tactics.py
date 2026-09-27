"""Rule-level one-ply tactics for the Stage 7-B PUCT v2 teacher (torch-free).

Only facts that follow directly from the Renju rules are used; no heuristic scores,
threat planners or VCF search:

- a move that completes a five for the side to move wins now (BLACK: exactly five,
  WHITE: five or more — the same test as ``Game.play``);
- if the opponent has two or more winning points and the side to move cannot win now,
  the position is lost (one move blocks only one point);
- if the opponent has exactly one winning point, every other move loses at once, so
  the only non-losing move is that point (when it is legal for the side to move).

``tactical_filter`` turns these facts into (allowed moves, proven value). The search
keeps the full legal list for network input so encoder plane 5 never changes.
"""
from __future__ import annotations

from renju.rules import BLACK, DIRECTIONS, EMPTY, SIZE, inside, run_length

Move = tuple[int, int]


def _build_windows() -> tuple[tuple[Move, ...], ...]:
    windows = []
    for row in range(SIZE):
        for col in range(SIZE):
            for dr, dc in DIRECTIONS:
                cells = tuple((row + i * dr, col + i * dc) for i in range(5))
                if all(inside(r, c) for r, c in cells):
                    windows.append(cells)
    return tuple(windows)


WINDOWS = _build_windows()


def _makes_five(board, player: int, move: Move) -> bool:
    row, col = move
    board[row][col] = player
    try:
        lengths = (run_length(board, row, col, dr, dc) for dr, dc in DIRECTIONS)
        return any(n == 5 if player == BLACK else n >= 5 for n in lengths)
    finally:
        board[row][col] = EMPTY


def winning_points(board, player: int) -> list[Move]:
    """Empty points where ``player`` would complete a winning five (board restored).

    A five through a new stone lies in a five-cell window holding four of the
    player's stones and that empty point, so scanning windows is exhaustive; each
    candidate is then confirmed with the exact ``Game.play`` win test.
    """
    candidates = set()
    for window in WINDOWS:
        empty = None
        count = 0
        for r, c in window:
            value = board[r][c]
            if value == player:
                count += 1
            elif value == EMPTY and empty is None:
                empty = (r, c)
            else:
                break
        else:
            if count == 4 and empty is not None:
                candidates.add(empty)
    return sorted(move for move in candidates if _makes_five(board, player, move))


def tactical_filter(game, legal_moves: list[Move]) -> tuple[list[Move], float | None]:
    """Return (allowed moves, proven value from ``game.to_play``'s view or None).

    ``allowed`` is a non-empty subset of ``legal_moves`` in the same order. A proven
    value of +1 means the side to move wins now; -1 means every move loses at once.
    """
    board = game.board
    player = game.to_play
    legal = set(legal_moves)
    own = [move for move in winning_points(board, player) if move in legal]
    if own:
        return [move for move in legal_moves if move in set(own)], 1.0
    threats = winning_points(board, -player)
    if not threats:
        return list(legal_moves), None
    if len(threats) == 1 and threats[0] in legal:
        return [threats[0]], None
    return list(legal_moves), -1.0
