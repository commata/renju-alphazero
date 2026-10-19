"""TB-Engine-v1-T: the H7-E2 candidate engine (docs/h6-value.md, H7-E2).

TB-Engine-v1 with one change: V8-C's root solver searches VCF with the call-local table of exact no-win
states (``analysis.vcf_tt_v1``, ``root_vcf_table``). V8-A, V8-B, the own VCF, ``threats.py``, the PUCT tree,
the H3 policy and every budget are TB-Engine-v1's. Adopted as TB-Engine-v2 only through the H7-E2 gate
(track-b-post-e1.md §2.3).

``root_vcf_table`` is not a ``V8_DEFAULTS`` option (TB-Engine-v1's configuration stays as frozen), so the
engine is the frozen configuration plus ``ENGINE_OPTIONS``; ``config_hash`` covers both.
"""
from __future__ import annotations

from .tb_engine_v1 import TB_ENGINE_V1

NAME = 'TB-Engine-v1-T'
BASE = 'TB-Engine-v1'
TB_ENGINE_V1_T = dict(TB_ENGINE_V1)
ENGINE_OPTIONS = {'root_vcf_table': True}
VCF_TABLE = 'analysis.vcf_tt_v1'


def config_hash() -> str:
    """SHA-256 of the canonical JSON of the configuration, the engine options and the table module."""
    import hashlib
    import json

    payload = {'v8_config': TB_ENGINE_V1_T, 'engine_options': ENGINE_OPTIONS, 'vcf_table': VCF_TABLE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42):
    """``MCTSV8Agent`` with the candidate configuration; ``policy`` is the H3 ``RootPolicy``."""
    from .mcts_v8_agent import MCTSV8Agent

    agent = MCTSV8Agent(seed=seed, root_policy=policy, **TB_ENGINE_V1_T, **ENGINE_OPTIONS)
    agent.name = NAME
    return agent
