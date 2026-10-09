"""S3-VCT2-v1: the frozen engine of docs/mcts-v8-teacher.md §12.25.

The configuration is written out in full (not derived from ``V8_DEFAULTS``) so a later
change of a default cannot change this engine; ``tests/test_s3_vct2_v1.py`` checks that it
equals the ``v8_config`` of the S3 benchmark results. The H3 policy checkpoint is part of
the engine: ``check_checkpoint`` compares its bytes with the ones the S3 runs used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

NAME = 'S3-VCT2-v1'
RESULTS_COMMIT = 'b06ecd6'  # §12.25; engine code unchanged since the S3 run commit b25aa3e
S3_VCT2_V1 = {
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
    # S3 selective depth-2 veto
    'root_vct2_check': True, 'vct2_node_limit': 20000, 'vct2_call_limit': 20000,
    'vct2_node_budget': 10000, 'vct2_max_children': 4,
}
POLICY_CHECKPOINT_SHA256 = 'efee483284dfdd87eb6caa59d0ea3fd015c9dd5862e5e26925b9a23095d0ce61'
POLICY_METADATA_SHA256 = '5648b7fbe71dfa4ab92f6c9bc505ee9ec0af68414f86b97ea22744c9fc022609'


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint(checkpoint) -> dict:
    """Hashes of the policy checkpoint and its metadata; ValueError unless both are the frozen ones."""
    checkpoint = Path(checkpoint)
    found = {'policy_checkpoint_sha256': _sha256(checkpoint),
             'policy_metadata_sha256': _sha256(checkpoint.with_suffix('.json'))}
    expected = {'policy_checkpoint_sha256': POLICY_CHECKPOINT_SHA256,
                'policy_metadata_sha256': POLICY_METADATA_SHA256}
    if found != expected:
        raise ValueError(f'{checkpoint} is not the {NAME} policy checkpoint: {found} != {expected}')
    return found


def make_agent(policy, seed: int = 42):
    """``MCTSV8Agent`` with the frozen configuration; ``policy`` is the H3 ``RootPolicy``."""
    from .mcts_v8_agent import MCTSV8Agent

    agent = MCTSV8Agent(seed=seed, root_policy=policy, **S3_VCT2_V1)
    agent.name = NAME
    return agent
