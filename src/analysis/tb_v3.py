"""TB-V3 Resolve: TB-V2-Value + the E4-v2 attack, adopted by the H7-E4-v2 arena (docs/h6-value.md, E4-v2).

The frozen candidate ``TB-V2-Value+E4v2`` (``analysis.tb_v2_value_e4v2``, ``150k/cd/killer``) under its engine
identity, with no behaviour change: the same base engine (``analysis.tb_v2_value``, unchanged), the same attack
step (``e4v2_attack.decide`` through the candidate's ``E4v2AttackAgent``, ``ATTACK`` below written out) and
the same runtime (fast4). Evidence: ``ADOPTION``. It becomes the canonical only once the promotion replay
(``scripts/tb_v3_promotion.py``: the 200 arena games of the candidate, every move and E4-v2 step equal) passes.

``ENGINE_HASH`` covers what earlier engine hashes left to run-time guards:
    configuration   the base engine hash and the attack configuration and sources
    models          the H3 policy checkpoint and metadata, the value checkpoint and its ``best.json``
    code            the SHA-256 of the code manifest ``tests/tb_v3_frozen.sha256``: every file of the import
                    closure of the TB-V3 factory (base engine, attack, solvers, policy / value inference, rules,
                    fast4 bindings, guards), except this file, which holds the manifest's hash
This file is pinned by its own SHA-256 inside the runtime hash (``tb_v3_runtime.runtime_hash``).
The factory (``tb_v3_runtime.make_tb_v3_agent``) checks all of it and fails closed: a file, checkpoint or
model it cannot verify is refused.
"""
from __future__ import annotations

from pathlib import Path

ENGINE_ID = 'TB-V3'
ENGINE_CODENAME = 'Resolve'
ENGINE_NAME = 'TB-V3 Resolve'
BASE_ENGINE = 'TB-V2-Value'
BASE_ENGINE_HASH = '256e5abe028c694891d33b9be1b9f58287fe18b6d4af278822b03cf09b7ec5c1'
ATTACK = {'name': '150k/cd/killer', 'budget': '150k', 'cooldown': True, 'killer': True,
          'budget_values': {'node_limit': 20_000, 'call_limit': 15_000, 'node_budget': 150_000},
          'gate_node_limit': 20_000, 'routes': ['stage4', 'stage5', 'tree']}
ATTACK_SOURCES = {
    'src/analysis/e4v2_attack.py': 'dd61f5af34d4ed26fa4e58fa6d43f3b09902136914610673c2a226cbca12de47',
    'src/analysis/e4_attack.py': '73b0902d9d479065b9279e43ca356509dfdec4daa1a759f881aa695009b7415f',
    'src/analysis/vct2_attack.py': 'ad263e3fc981f186f3196cdd914f646dc6d864038f412cb1e728c5a254515c15',
    'src/analysis/selective_vct.py': 'b0b3a0637ea91a6808a60521dfdda447958020f52cf2ec17ac29c10be369889b',
}
MODELS = {
    'policy_checkpoint_sha256': 'efee483284dfdd87eb6caa59d0ea3fd015c9dd5862e5e26925b9a23095d0ce61',
    'policy_metadata_sha256': '5648b7fbe71dfa4ab92f6c9bc505ee9ec0af68414f86b97ea22744c9fc022609',
    'value_checkpoint_sha256': 'a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca',
    'value_metadata_sha256': '51fd3fa4623314d415e1c75140c048a88daf9526098bd38ef771e11929d306ea',
}
CODE_MANIFEST = 'tests/tb_v3_frozen.sha256'
CODE_MANIFEST_SHA256 = 'eb15dedce46312511a2b0cc9aee666885d755170cc551147c2f48c324d4bc790'
ADOPTION = {
    'candidate': 'TB-V2-Value+E4v2',
    'candidate_hash': 'f5e43acab53bc3fa2169ea34ddd281ac3cb83be6200f956057a8a72daa2cd8b8',
    'arena': 'docs/h7/h7_e4v2_arena.json',
    'arena_sha256': '8fe17da9a4564df63948a61e126e53f7c1b8003ea52b062ff9c2a978609fa4e6',
    'precheck_sha256': '9694fcd8a8b85aa99e271d4464f547293df1e9e4f8ac2038d741c9a2224a28df',
    'reference_sha256': '9293b4908a9007aac6edcc20a5e8b99c777b4022b4b9ed2cc33f2d4962758f66',
}
ENGINE_HASH = 'd0ef0da6b8b922cf3020acf6ff6090424a3f321079b658abd88833dd4ce5f85b'
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash() -> str:
    """SHA-256 of the canonical JSON of everything above that makes the engine."""
    import hashlib
    import json

    payload = {'engine_id': ENGINE_ID, 'base_engine_hash': BASE_ENGINE_HASH, 'attack': ATTACK,
               'attack_sources': ATTACK_SOURCES, 'models': MODELS, 'code_manifest_sha256': CODE_MANIFEST_SHA256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_identity() -> None:
    """Raise unless the engine hash, the base, the candidate it was adopted as, the model constants and every
    file of the code manifest are the frozen ones."""
    from . import s3_vct2_v1, tb_v2_runtime, tb_v2_value, tb_v2_value_e4v2, track_b_guard

    if config_hash() != ENGINE_HASH:
        raise RuntimeError(f'{ENGINE_NAME}: config_hash {config_hash()[:12]} != ENGINE_HASH {ENGINE_HASH[:12]}')
    if tb_v2_value.config_hash() != BASE_ENGINE_HASH or tb_v2_runtime.ENGINE_HASH != BASE_ENGINE_HASH:
        raise RuntimeError(f'{ENGINE_NAME}: the base is not the frozen {BASE_ENGINE}')
    if (tb_v2_value_e4v2.ATTACK != ATTACK or tb_v2_value_e4v2.ATTACK_SOURCES != ATTACK_SOURCES
            or tb_v2_value_e4v2.candidate_hash() != ADOPTION['candidate_hash']):
        raise RuntimeError(f'{ENGINE_NAME}: the attack is not the adopted candidate {ADOPTION["candidate"]}')
    tb_v2_value_e4v2.check_sources()  # attack files, the TB-V2-Value manifest, the attack configuration
    known = {'policy_checkpoint_sha256': s3_vct2_v1.POLICY_CHECKPOINT_SHA256,
             'policy_metadata_sha256': s3_vct2_v1.POLICY_METADATA_SHA256,
             'value_checkpoint_sha256': tb_v2_value.VALUE_CHECKPOINT_SHA256,
             'value_metadata_sha256': track_b_guard.VALUE_METADATA_SHA256}
    if known != MODELS:
        raise RuntimeError(f'{ENGINE_NAME}: the model checks do not test the frozen models')
    manifest = _ROOT / CODE_MANIFEST
    if not manifest.is_file() or _sha(manifest) != CODE_MANIFEST_SHA256:
        raise RuntimeError(f'{ENGINE_NAME}: {CODE_MANIFEST} is not the frozen code manifest')
    bad = track_b_guard.check_code(manifest)
    if bad:
        raise RuntimeError(f'{ENGINE_NAME}: the code differs from {CODE_MANIFEST}: {bad}')
