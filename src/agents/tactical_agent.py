"""One-ply wins, immediate threat blocking, then seeded random play."""
from copy import deepcopy
from random import Random

from renju import BLACK, EMPTY, Game, IllegalMove
from renju.rules import DIRECTIONS, run_length


def _wins(game: Game, move: tuple[int, int]) -> bool:
    """Check an already legal move on a private copy using engine primitives."""
    row, col = move
    game.board[row][col] = game.to_play
    try:
        lengths = (run_length(game.board, row, col, dr, dc) for dr, dc in DIRECTIONS)
        return any(n == 5 if game.to_play == BLACK else n >= 5 for n in lengths)
    finally:
        game.board[row][col] = EMPTY


class TacticalAgent:
    name = "Tactical"

    def __init__(self, seed: int = 42):
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        probe = deepcopy(game)
        moves = probe.legal_moves()
        if not moves:
            raise IllegalMove("No legal moves available")
        for move in moves:
            if _wins(probe, move):
                return move

        probe.to_play = -probe.to_play
        own_legal = set(moves)
        for move in probe.legal_moves():
            if move in own_legal and _wins(probe, move):
                return move
        return self._random.choice(moves)
