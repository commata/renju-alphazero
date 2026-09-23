"""Agent wrapper for pure Monte Carlo Tree Search."""
from math import sqrt
from random import Random

from renju import Game
from search import mcts_search


class MCTSAgent:
    name = "MCTS"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 10,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 8,
    ):
        if type(simulations) is not int or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if exploration <= 0:
            raise ValueError("exploration must be positive")
        if type(candidate_limit) is not int or candidate_limit <= 0:
            raise ValueError("candidate_limit must be a positive integer")
        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            random=self._random,
        )
