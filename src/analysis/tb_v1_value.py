"""TB-V1-Value: the H6-4 candidate engine (docs/h6-value.md, H6-4; docs/mcts-v8-teacher.md §12.19).

TB-Engine-v1 with one change: the tree route's leaf is the H6-2a' value (v3 checkpoint, D4-averaged)
instead of the rollout (``tree_mode="puct_value"``, ``analysis.puct_value_v1``). λ = 1.0 (H6-3 gate:
ECE <= 0.10), so no rollout is mixed in. Everything else, the H3 policy prior included, is TB-Engine-v1's.

The value checkpoint is part of the engine (``VALUE_CHECKPOINT_SHA256``); ``config_hash`` covers it.
"""
from __future__ import annotations

from .tb_engine_v1 import TB_ENGINE_V1

NAME = 'TB-V1-Value'
BASE = 'TB-Engine-v1'
TB_V1_VALUE = {**TB_ENGINE_V1, 'tree_mode': 'puct_value'}
VALUE_CHECKPOINT_SHA256 = 'a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca'
VALUE_INFERENCE = 'd4_average'
VALUE_LAMBDA = 1.0


def config_hash() -> str:
    """SHA-256 of the canonical JSON of the configuration plus the value evaluator."""
    import hashlib
    import json

    payload = {'v8_config': TB_V1_VALUE, 'value_checkpoint_sha256': VALUE_CHECKPOINT_SHA256,
               'value_inference': VALUE_INFERENCE, 'value_lambda': VALUE_LAMBDA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, leaf_value, seed: int = 42):
    """``MCTSV8Agent`` with the candidate configuration; ``policy`` is the H3 ``RootPolicy`` and
    ``leaf_value`` the ``hybrid.h6_value_leaf.ValueLeaf`` of the gated checkpoint."""
    from .mcts_v8_agent import MCTSV8Agent

    agent = MCTSV8Agent(seed=seed, root_policy=policy, leaf_value=leaf_value, **TB_V1_VALUE)
    agent.name = NAME
    return agent
