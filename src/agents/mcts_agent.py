"""Agent wrappers for the frozen V2 and experimental V3 pure-MCTS revisions."""
from math import sqrt
from random import Random

from renju import Game
from search import mcts_search, mcts_search_v3, mcts_search_v32, mcts_search_v321, mcts_search_v4


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

    name = "MCTS-v3.1"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 25,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 16,
        initial_width: int = 6,
        neighborhood_radius: int = 2,
        priority_top_k: int = 5,
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
        if type(priority_top_k) is not int or priority_top_k <= 0:
            raise ValueError("priority_top_k must be a positive integer")
        if priority_top_k > candidate_limit:
            raise ValueError("priority_top_k must not exceed candidate_limit")

        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self.initial_width = initial_width
        self.neighborhood_radius = neighborhood_radius
        self.priority_top_k = priority_top_k
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v3(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.neighborhood_radius,
            priority_top_k=self.priority_top_k,
            random=self._random,
        )


class MCTSV32Agent:
    """V3.2 with color-aware Renju tactical priorities."""

    name = "MCTS-v3.2"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 25,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 16,
        initial_width: int = 6,
        neighborhood_radius: int = 2,
        priority_top_k: int = 5,
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
        if type(priority_top_k) is not int or priority_top_k <= 0:
            raise ValueError("priority_top_k must be a positive integer")
        if priority_top_k > candidate_limit:
            raise ValueError("priority_top_k must not exceed candidate_limit")

        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self.initial_width = initial_width
        self.neighborhood_radius = neighborhood_radius
        self.priority_top_k = priority_top_k
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v32(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.neighborhood_radius,
            priority_top_k=self.priority_top_k,
            random=self._random,
        )


class MCTSV321Agent:
    """V3.2 policy with optimized non-recursive heuristic pattern scanning."""

    name = "MCTS-v3.2.1"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 25,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 16,
        initial_width: int = 6,
        neighborhood_radius: int = 2,
        priority_top_k: int = 5,
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
        if type(priority_top_k) is not int or priority_top_k <= 0:
            raise ValueError("priority_top_k must be a positive integer")
        if priority_top_k > candidate_limit:
            raise ValueError("priority_top_k must not exceed candidate_limit")

        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self.initial_width = initial_width
        self.neighborhood_radius = neighborhood_radius
        self.priority_top_k = priority_top_k
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v321(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.neighborhood_radius,
            priority_top_k=self.priority_top_k,
            random=self._random,
        )


class _MCTSV4Base:
    """Shared V4 wrapper with forced open-three defense."""

    name = "MCTS-v4"

    def __init__(
        self,
        *,
        seed: int,
        simulations: int,
        exploration: float,
        candidate_limit: int,
        initial_width: int,
        neighborhood_radius: int,
        priority_top_k: int,
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
        if type(priority_top_k) is not int or priority_top_k <= 0:
            raise ValueError("priority_top_k must be a positive integer")
        if priority_top_k > candidate_limit:
            raise ValueError("priority_top_k must not exceed candidate_limit")

        self.simulations = simulations
        self.exploration = exploration
        self.candidate_limit = candidate_limit
        self.initial_width = initial_width
        self.neighborhood_radius = neighborhood_radius
        self.priority_top_k = priority_top_k
        self._random = Random(seed)

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v4(
            game,
            simulations=self.simulations,
            exploration=self.exploration,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.neighborhood_radius,
            priority_top_k=self.priority_top_k,
            random=self._random,
        )


class MCTSV41Agent(_MCTSV4Base):
    """V4.1: 50 simulations with a moderate 20/8/8 candidate budget."""

    name = "MCTS-v4.1"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 50,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 20,
        initial_width: int = 8,
        neighborhood_radius: int = 2,
        priority_top_k: int = 8,
    ):
        super().__init__(
            seed=seed,
            simulations=simulations,
            exploration=exploration,
            candidate_limit=candidate_limit,
            initial_width=initial_width,
            neighborhood_radius=neighborhood_radius,
            priority_top_k=priority_top_k,
        )


class MCTSV42Agent(_MCTSV4Base):
    """V4.2: 50 simulations with a wider 24/10/10 candidate budget."""

    name = "MCTS-v4.2"

    def __init__(
        self,
        seed: int = 42,
        simulations: int = 50,
        exploration: float = sqrt(2.0),
        candidate_limit: int = 24,
        initial_width: int = 10,
        neighborhood_radius: int = 2,
        priority_top_k: int = 10,
    ):
        super().__init__(
            seed=seed,
            simulations=simulations,
            exploration=exploration,
            candidate_limit=candidate_limit,
            initial_width=initial_width,
            neighborhood_radius=neighborhood_radius,
            priority_top_k=priority_top_k,
        )
