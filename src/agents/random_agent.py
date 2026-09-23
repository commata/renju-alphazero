"""Uniform sampling from the engine's legal moves."""
from random import Random

from renju import Game, IllegalMove


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int = 42):
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        moves = game.legal_moves()
        if not moves:
            raise IllegalMove("No legal moves available")
        return self._random.choice(moves)
