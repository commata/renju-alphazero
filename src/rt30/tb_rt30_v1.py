"""TB-RT30-v1: the frozen real-time web engine (docs/web-ser5.md §9).

TB-Engine-v1's flow and budgets under one absolute deadline (``rt30.engine``), with the exact
speed-ups (``rt30.fast``), the V8-C root rescue and the V8-B cap chosen in A1 (8 s; §3.10, §9).
The configuration is written out in full, TB-Engine-v1's included, so a later change of a default
elsewhere cannot change this engine; ``tests/test_tb_rt30_v1.py`` checks it against the A1 runs
and TB-Engine-v1, and ``tests/frozen_tb_rt30_v1.sha256`` locks the files that define its moves.

Changing anything here, or in a locked file, makes a new engine (TB-RT30-v2).
"""
from __future__ import annotations

from analysis.tb_engine_v1 import POLICY_CHECKPOINT_SHA256, POLICY_METADATA_SHA256, check_checkpoint  # noqa: F401

NAME = 'TB-RT30-v1'
BASE_ENGINE = 'TB-Engine-v1'
FREEZE_EVIDENCE = {
    'equivalence': 'docs/rt30-results/h3_equivalence_fast2.json',          # frozen = fast = rt (no deadline), 8/8
    'cap8_s60': 'docs/rt30-results/cap8_h3_slow-stratified-v1.json',       # the adopted timing, H3
    'cap8_s60_recheck': 'docs/rt30-results/recheck_cap8_h3_slow-stratified-v1.json',
    'cap8_critical': 'docs/rt30-results/cap8_h3_critical-regressions-v1.json',
    'cap8_critical_recheck': 'docs/rt30-results/recheck_cap8_h3_critical-regressions-v1.json',
    'v8b_bench': ('docs/rt30-results/v8b_fast2_cap4.json', 'docs/rt30-results/v8b_fast2_cap6.json',
                  'docs/rt30-results/v8b_fast2_cap8.json'),
}
TB_RT30_V1 = {
    # TB-Engine-v1, written out (analysis.tb_engine_v1.TB_ENGINE_V1)
    'base_config': {
        'simulations': 50, 'tactical_simulations': 100, 'tactical_score_threshold': 1800,
        'exploration': 1.4142135623730951, 'candidate_limit': 20, 'initial_width': 8,
        'neighborhood_radius': 2, 'priority_top_k': 8, 'own_vcf_max_fours': 10, 'own_vcf_node_limit': 5000,
        'safety_vcf_max_fours': 10, 'safety_vcf_node_limit': 4000, 'safety_precheck_node_limit': 8000,
        'safety_total_node_limit': 8000, 'self_forbidden_min_white': 3,
        'stage_vct_safety': True, 'vct_vcf_node_limit': 20000, 'vct_call_limit': 3000, 'vct_node_budget': 200000,
        'own_vct_attack': True, 'attack_vcf_node_limit': 20000, 'attack_call_limit': 10000,
        'attack_node_budget': 200000, 'root_vct_safety': True, 'root_vcf_node_limit': 20000,
        'root_call_limit': 10000, 'root_node_budget': 400000, 'root_max_children': 4,
        'root_vct_mode': 'aggressive', 'root_policy_order': False, 'root_policy_extra': 0,
        'tree_mode': 'puct', 'puct_prior': 'policy', 'puct_c': 1.5,
        'root_vct2_check': True, 'vct2_node_limit': 20000, 'vct2_call_limit': 20000,
        'vct2_node_budget': 50000, 'vct2_max_children': 4,
    },
    # Real-time layer (rt30.engine)
    'deadline_seconds': 27.5,       # absolute, from the search start
    'watchdog_seconds': 29.0,       # rt30.supervisor
    'timing': {'final_margin': 1.5, 'attack_seconds': 8.0, 'puct_reserve': 6.0, 'vct2_reserve': 4.0,
               'vct2_min': 2.0},
    'attack_order': 'frozen',
    'attack_scheduler': 'fair',
    'root_rescue': True,
    'exact_speedups': ['threat_windows_codes', 'winning_extensions_in_direction'],
}


def config_hash() -> str:
    """SHA-256 of the canonical JSON of the configuration."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(TB_RT30_V1, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42, *, uniform_prior: bool = False):
    """``RT30Agent`` with the frozen configuration; installs the exact speed-ups in this process.

    ``policy`` is the H3 ``RootPolicy`` (hash-check the checkpoint with ``check_checkpoint``).
    ``uniform_prior=True`` (``policy=None``) runs the same engine with a uniform PUCT prior: a test
    engine for machines without the checkpoint, not TB-RT30-v1.
    """
    from . import fast
    from .engine import RT30Agent

    config = dict(TB_RT30_V1['base_config'])
    if uniform_prior:
        config['puct_prior'] = 'uniform'
        policy = None
    elif policy is None:
        raise ValueError('TB-RT30-v1 needs the H3 policy (or uniform_prior=True for a test engine)')
    fast.install_all()
    agent = RT30Agent(seed=seed, root_policy=policy, timing=dict(TB_RT30_V1['timing']),
                      attack_order=TB_RT30_V1['attack_order'], attack_scheduler=TB_RT30_V1['attack_scheduler'],
                      root_rescue=TB_RT30_V1['root_rescue'], **config)
    agent.name = NAME if not uniform_prior else f'{NAME} (uniform prior)'
    return agent
