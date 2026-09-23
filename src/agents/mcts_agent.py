"""Agent wrappers for the frozen V2 and experimental V3 pure-MCTS revisions."""
from math import sqrt
from random import Random

from renju import Game
from search import mcts_search, mcts_search_v3


class MCTSV2Agent:
    """Frozen second revision for direct V2-vs-V3 evaluation."""

    name = "MCTS-v2"

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


class MCTSAgent(MCTSV2Agent):
    """Backward-compatible name for the second revision."""

    name = "MCTS"


class MCTSV3Agent:
    """Third revision with a wider pool, progressive widening, and local candidate generation."""

    name = "MCTS-v3"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 10,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 12,
        initial_width: int = 4,
        neighborhood_radius: int = 2,
    ):
        if type(simulations) is not int or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if exploration <= 0:
            raise ValueError("exploration must be positive")
        if type(candidate_limit) is not int or candidate_limit <= 0:
            raise ValueError("candidate_limit must be a positive integer")
        if type(initial_width) is not int or initial_width <= 0:
            raise ValueError("initial_width must be a positive integer")
        if initial_width > candidate_limit:
            raise ValueError("initial_width must not exceed candidate_limit")
        if type(neighborhood_radius) is not int or neighborhood_radius <= 0:
            raise ValueError("neighborhood_radius must be a positive integer")

        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self.initial_width = initial_width
        self.neighborhood_radius = neighborhood_radius
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v3(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.neighborhood_radius,
            random=self._random,
        )
