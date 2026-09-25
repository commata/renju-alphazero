"""V6 agent with the fixed V5 FINAL search configuration."""
from renju import Game
from search.mcts_v6 import V5_FINAL, SearchDiagnostics, mcts_search_v6
from .mcts_agent import MCTSV5Agent


class MCTSV6Agent(MCTSV5Agent):
    def __init__(self, seed: int = 42, **overrides):
        config = {**V5_FINAL, **overrides}
        super().__init__(seed=seed, **config)
        self.name = 'MCTS-v6'
        self.diagnostics = SearchDiagnostics()

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v6(
            game, **{key: getattr(self, key) for key in V5_FINAL},
            random=self._random, diagnostics=self.diagnostics,
        )
