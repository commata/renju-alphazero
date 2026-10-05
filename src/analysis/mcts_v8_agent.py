"""MCTS-v8 teacher agent (not exported from ``agents``; import it where V8 is requested)."""
from __future__ import annotations

from agents.mcts_v6_agent import MCTSV6Agent
from renju import Game
from search.mcts_v6 import V5_FINAL
from search.mcts_v7 import _validate_v7_config

from .mcts_v8 import _V8_KEYS, V8_DEFAULTS, SearchDiagnostics, _validate_v8_config, mcts_search_v8

_EXTRA_KEYS = tuple(key for key in V8_DEFAULTS if key not in V5_FINAL)


class MCTSV8Agent(MCTSV6Agent):
    def __init__(self, seed: int = 42, root_policy=None, **overrides):
        """``root_policy``: callable ``game -> {move: probability}`` for the H4 options
        (``root_policy_order`` / ``root_policy_extra``); required when either is on."""
        unknown = set(overrides) - set(V8_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown V8 option(s): {', '.join(sorted(unknown))}")
        config = {**V8_DEFAULTS, **overrides}
        _validate_v7_config(**{key: config[key] for key in _EXTRA_KEYS if key not in _V8_KEYS})
        _validate_v8_config(**{key: config[key] for key in _V8_KEYS})
        if (config["root_policy_order"] or config["root_policy_extra"]) and root_policy is None:
            raise ValueError("root_policy_order / root_policy_extra need a root_policy callable")
        self.root_policy = root_policy
        super().__init__(seed=seed, **{key: config[key] for key in V5_FINAL})
        for key in _EXTRA_KEYS:
            setattr(self, key, config[key])
        self.name = "MCTS-v8"
        self.diagnostics = SearchDiagnostics()

    def select_move(self, game: Game) -> tuple[int, int]:
        return mcts_search_v8(
            game,
            **{key: getattr(self, key) for key in V8_DEFAULTS},
            root_policy=self.root_policy,
            random=self._random,
            diagnostics=self.diagnostics,
        )
