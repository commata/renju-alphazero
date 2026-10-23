"""TB-Engine-v2: the frozen Track B engine after H7-E2 (docs/h6-value.md, H7-E2; track-b-post-e1.md §2.3).

TB-Engine-v1 with one change: V8-C's root solver searches VCF with the call-local table of exact no-win
states (``analysis.vcf_tt_v1``, ``root_vcf_table``). Adopted from TB-Engine-v1-T by the H7-E2 holdout
(seeds 8421 + 8422: soundness, strength, UNKNOWN frequency and cost all PASS).

The configuration is written out in full (not derived from ``TB_ENGINE_V1`` or ``V8_DEFAULTS``) so a later
change elsewhere cannot change this engine; ``root_vcf_table`` is not a ``V8_DEFAULTS`` option and is kept
in ``ENGINE_OPTIONS``. The table implementation is part of the engine: ``VCF_TABLE_SHA256`` is the SHA-256
of ``src/analysis/vcf_tt_v1.py`` (``tests/test_tb_engine_v2.py`` checks it) and ``config_hash`` covers it.
The H3 policy checkpoint is part of the engine as for TB-Engine-v1 (``check_checkpoint``).

Changing anything here, or in ``vcf_tt_v1.py``, makes a new engine, adopted only at an H7 cycle boundary
through the gate of track-b-post-e1.md §2.
"""
from __future__ import annotations

from .s3_vct2_v1 import POLICY_CHECKPOINT_SHA256, POLICY_METADATA_SHA256, check_checkpoint  # noqa: F401

NAME = 'TB-Engine-v2'
ADOPTED_FROM = 'TB-Engine-v1-T'
FREEZE_EVIDENCE = {
    'e1': 'docs/h7/h7_e1_vcf_table.json',               # go: 44.3% of standard UNKNOWN decided, cost 0.65
    'e2': 'docs/h7/h7_e2_summary_8421_8422.json',       # ADOPT (8421 + 8422)
}
TB_ENGINE_V2 = {
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
# H7-E2: V8-C's root solver on the call-local VCF no-win table (not a V8_DEFAULTS option)
ENGINE_OPTIONS = {'root_vcf_table': True}
VCF_TABLE = 'analysis.vcf_tt_v1'
VCF_TABLE_SHA256 = '68937a5f9792876e67d9f306624ea29b4ee74f4b238a5e44e146411d17555694'


def config_hash() -> str:
    """SHA-256 of the canonical JSON of the configuration, the engine options and the table implementation."""
    import hashlib
    import json

    payload = {'v8_config': TB_ENGINE_V2, 'engine_options': ENGINE_OPTIONS, 'vcf_table': VCF_TABLE,
               'vcf_table_sha256': VCF_TABLE_SHA256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42):
    """``MCTSV8Agent`` with the frozen configuration; ``policy`` is the H3 ``RootPolicy``."""
    from .mcts_v8_agent import MCTSV8Agent

    agent = MCTSV8Agent(seed=seed, root_policy=policy, **TB_ENGINE_V2, **ENGINE_OPTIONS)
    agent.name = NAME
    return agent
