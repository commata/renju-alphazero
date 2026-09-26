"""MCTS-v7 agent using the fixed Stage 6.5 benchmark configuration."""
from renju import Game
from search.mcts_v6 import V5_FINAL
from search.mcts_v7 import V7_FINAL, SearchDiagnostics, _validate_v7_config, mcts_search_v7
from .mcts_v6_agent import MCTSV6Agent


_V7_KEYS = tuple(key for key in V7_FINAL if key not in V5_FINAL)


class MCTSV7Agent(MCTSV6Agent):
    def __init__(self, seed: int = 42, **overrides):
        unknown = set(overrides) - set(V7_FINAL)
        if unknown:
            raise TypeError(f"unknown V7 option(s): {', '.join(sorted(unknown))}")
        config = {**V7_FINAL, **overrides}
        _validate_v7_config(
            own_vcf_max_fours=config["own_vcf_max_fours"],
            own_vcf_node_limit=config["own_vcf_node_limit"],
            safety_vcf_max_fours=config["safety_vcf_max_fours"],
            safety_vcf_node_limit=config["safety_vcf_node_limit"],
            safety_precheck_node_limit=config["safety_precheck_node_limit"],
            safety_total_node_limit=config["safety_total_node_limit"],
            self_forbidden_min_white=config["self_forbidden_min_white"],
        )
        super().__init__(seed=seed, **{key: config[key] for key in V5_FINAL})
        for key in _V7_KEYS:
            setattr(self, key, config[key])
        self.name = "MCTS-v7"
        self.diagnostics = SearchDiagnostics()

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v7(
            game,
            **{key: getattr(self, key) for key in V7_FINAL},
            random=self._random,
            diagnostics=self.diagnostics,
        )
