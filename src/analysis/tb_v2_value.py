"""TB-V2-Value: the frozen Track B engine + net after H7-E3 (docs/h6-value.md, H7-E3).

TB-Engine-v2 (V8-C's root solver on the call-local VCF no-win table) with the H6-2a' value v3 as the tree
route's leaf (``tree_mode="puct_value"``, ``analysis.puct_value_v1``, D4-averaged, λ = 1.0: no rollout
mixed in). Adopted by the H7-E3 holdout (seeds 8407 + 8408, guard v2 unchanged: ADOPT).

The configuration is written out in full (not derived from ``TB_ENGINE_V2``, ``TB_V1_VALUE`` or
``V8_DEFAULTS``) so a later change elsewhere cannot change this engine; ``root_vcf_table`` is not a
``V8_DEFAULTS`` option and is kept in ``ENGINE_OPTIONS``. Part of the engine and of ``config_hash``: the table
implementation (``VCF_TABLE_SHA256`` = SHA-256 of ``src/analysis/vcf_tt_v1.py``), the value checkpoint
(``VALUE_CHECKPOINT_SHA256``, checked by ``hybrid.h6_value_leaf.ValueLeaf``), its inference (D4 average) and
λ, and the H3 policy checkpoint as for TB-Engine-v1 (``check_checkpoint``).

Changing anything here, or in a file of its identity, makes a new engine. Runtime optimizations are separate
candidates that must pass the same-move equivalence gate (docs/h6-value.md, after H7-E3) before a new freeze.
"""
from __future__ import annotations

from .s3_vct2_v1 import POLICY_CHECKPOINT_SHA256, POLICY_METADATA_SHA256, check_checkpoint  # noqa: F401

NAME = 'TB-V2-Value'
BASE = 'TB-Engine-v2'
FREEZE_EVIDENCE = {
    'e2': 'docs/h7/h7_e2_summary_8421_8422.json',   # TB-Engine-v2 ADOPT
    'e3': 'docs/h7/h7_e3_summary_8407_8408.json',   # TB-V2-Value ADOPT (guard v2)
}
TB_V2_VALUE = {
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
    # H5 PUCT with the H3 policy prior; H6-4 value leaf instead of the rollout
    'tree_mode': 'puct_value', 'puct_prior': 'policy', 'puct_c': 1.5,
    # Selective depth-2 veto after V8-C: E3-A budget
    'root_vct2_check': True, 'vct2_node_limit': 20000, 'vct2_call_limit': 20000,
    'vct2_node_budget': 50000, 'vct2_max_children': 4,
}
# H7-E2: V8-C's root solver on the call-local VCF no-win table (not a V8_DEFAULTS option)
ENGINE_OPTIONS = {'root_vcf_table': True}
VCF_TABLE = 'analysis.vcf_tt_v1'
VCF_TABLE_SHA256 = '68937a5f9792876e67d9f306624ea29b4ee74f4b238a5e44e146411d17555694'
VALUE_CHECKPOINT_SHA256 = 'a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca'
VALUE_INFERENCE = 'd4_average'
VALUE_LAMBDA = 1.0


def config_hash() -> str:
    """SHA-256 of the canonical JSON of everything above that makes the engine."""
    import hashlib
    import json

    payload = {'v8_config': TB_V2_VALUE, 'engine_options': ENGINE_OPTIONS, 'vcf_table': VCF_TABLE,
               'vcf_table_sha256': VCF_TABLE_SHA256, 'value_checkpoint_sha256': VALUE_CHECKPOINT_SHA256,
               'value_inference': VALUE_INFERENCE, 'value_lambda': VALUE_LAMBDA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, leaf_value, seed: int = 42):
    """``MCTSV8Agent`` with the frozen configuration; ``policy`` is the H3 ``RootPolicy`` and ``leaf_value``
    the ``hybrid.h6_value_leaf.ValueLeaf`` of the frozen checkpoint (refused if it reports another one)."""
    from .mcts_v8_agent import MCTSV8Agent

    sha = getattr(leaf_value, 'sha256', None)
    if sha is not None and sha != VALUE_CHECKPOINT_SHA256:
        raise ValueError(f'{NAME} needs the value checkpoint {VALUE_CHECKPOINT_SHA256[:8]}..., got {sha[:8]}...')
    agent = MCTSV8Agent(seed=seed, root_policy=policy, leaf_value=leaf_value, **TB_V2_VALUE, **ENGINE_OPTIONS)
    agent.name = NAME
    return agent
