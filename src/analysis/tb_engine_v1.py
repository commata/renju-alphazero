"""TB-Engine-v1: the frozen Track B engine (docs/mcts-v8-teacher.md §12.36; track-b-b1-b2.md §4).

E3-A, adopted in §12.33 and passed by the E1-holdout (§12.34, PASS on G1-G4): S3-VCT2-v1 with a
50k node budget per tree-route VCT2 check. The configuration is written out in full (not derived
from ``V8_DEFAULTS``, ``S3_VCT2_V1`` or ``E3_A``) so a later change elsewhere cannot change this
engine; ``tests/test_tb_engine_v1.py`` checks that it equals the ``v8_config`` of the E3 and
E1-holdout runs. The H3 policy checkpoint is part of the engine (``check_checkpoint``).

Changing anything here makes a new engine (TB-Engine-v2), adopted only at an H7 cycle boundary
through the gate of track-b-post-e1.md §2.
"""
from __future__ import annotations

from .s3_vct2_v1 import POLICY_CHECKPOINT_SHA256, POLICY_METADATA_SHA256, check_checkpoint  # noqa: F401

NAME = 'TB-Engine-v1'
ADOPTED_FROM = 'E3-A'
FREEZE_EVIDENCE = {
    'e3': 'docs/mcts-v8-results/e3/e3a_compare.json',           # MECHANISM_ESTABLISHED_BUT_MATCH_UNPROVEN, cost 1.15
    'e1_holdout': 'docs/mcts-v8-results/e1h/e1h_verdict.json',  # PASS (G1-G4, G5 N/A)
}
TB_ENGINE_V1 = {
    # V5 tree and V7 modules (V5_FINAL / V7_FINAL)
    'simulations': 50, 'tactical_simulations': 100, 'tactical_score_threshold': 1800,
    'exploration': 1.4142135623730951, 'candidate_limit': 20, 'initial_width': 8,
    'neighborhood_radius': 2, 'priority_top_k': 8, 'own_vcf_max_fours': 10, 'own_vcf_node_limit': 5000,
    'safety_vcf_max_fours': 10, 'safety_vcf_node_limit': 4000, 'safety_precheck_node_limit': 8000,
    'safety_total_node_limit': 8000, 'self_forbidden_min_white': 3,
    # V8-A / V8-B / V8-C
    'stage_vct_safety': True, 'vct_vcf_node_limit': 20000, 'vct_call_limit': 3000, 'vct_node_budget': 200000,
    'own_vct_attack': True, 'attack_vcf_node_limit': 20000, 'attack_call_limit': 10000,
    'attack_node_budget': 200000, 'root_vct_safety': True, 'root_vcf_node_limit': 20000,
    'root_call_limit': 10000, 'root_node_budget': 400000, 'root_max_children': 4,
    'root_vct_mode': 'aggressive', 'root_policy_order': False, 'root_policy_extra': 0,
    # H5 PUCT with the H3 policy prior
    'tree_mode': 'puct', 'puct_prior': 'policy', 'puct_c': 1.5,
    # Selective depth-2 veto after V8-C: E3-A budget
    'root_vct2_check': True, 'vct2_node_limit': 20000, 'vct2_call_limit': 20000,
    'vct2_node_budget': 50000, 'vct2_max_children': 4,
}


def config_hash() -> str:
    """SHA-256 of the canonical JSON of the configuration (``engine_config_hash`` of H6/H7 rows)."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(TB_ENGINE_V1, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42):
    """``MCTSV8Agent`` with the frozen configuration; ``policy`` is the H3 ``RootPolicy``."""
    from .mcts_v8_agent import MCTSV8Agent

    agent = MCTSV8Agent(seed=seed, root_policy=policy, **TB_ENGINE_V1)
    agent.name = NAME
    return agent
