"""TB-V3 Resolve + fast4: the runtime profile of TB-V3 (docs/h6-value.md, E4-v2 canonical promotion).

The fast4 profile of TB-V2-Value unchanged (native sources and bindings of ``analysis.tb_v2_runtime``) under
TB-V3's engine hash. ``runtime_hash`` also covers the SHA-256 of ``analysis/tb_v3.py``, the one engine file the
code manifest cannot hold. CUDA is the canonical value device; CPU needs its own certification (seeds 8451 /
8452) and the SER5 web profile its timing certification, neither of which changes the engine.

``make_tb_v3_agent`` is the only way to build TB-V3. It fails closed, in this order:
    identity   ``tb_v3.check_identity`` (engine hash, base, adopted candidate, model constants, code manifest)
    models     the real H3 ``RootPolicy`` and H6 ``ValueLeaf``, loaded from the given paths, both files of
               both checkpoints equal to ``tb_v3.MODELS`` (a missing file is refused, never skipped)
    runtime    the native and binding sources, and this process bound to fast4 with a fresh native build
then the frozen TB-V2-Value agent wrapped in the candidate's ``E4v2AttackAgent`` with ``tb_v3.ATTACK``: one
agent per game (its cooldown bit is the only state kept between moves).
"""
from __future__ import annotations

from pathlib import Path

from .tb_v2_runtime import BINDING_SOURCES, NATIVE_SOURCES, device_kind  # noqa: F401  (the fast4 profile)

NAME = 'TB-V3 Resolve/fast4'
ENGINE_ID = 'TB-V3'
FAST_MODE = 'fast4'
CANONICAL_DEVICE = 'cuda'
CERTIFICATION = {'cuda': 'canonical (the E4-v2 arena ran its value on CUDA)', 'cpu': None}
DEPLOYMENT = {'ser5': None}  # web timing certification (30 s deadline), separate from the engine
_IDENTITY_FILE = Path(__file__).resolve().parent / 'tb_v3.py'


def identity_sha256() -> str:
    import hashlib

    return hashlib.sha256(_IDENTITY_FILE.read_bytes()).hexdigest()


def runtime_hash(device: str) -> str:
    import hashlib
    import json

    from . import tb_v3

    payload = {'engine_id': ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH, 'identity_sha256': identity_sha256(),
               'fast_mode': FAST_MODE, 'native_sources': NATIVE_SOURCES, 'binding_sources': BINDING_SOURCES,
               'value_device': device_kind(device)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_bound() -> None:
    from . import tb_v2_runtime

    tb_v2_runtime.check_bound()


def install() -> dict:
    from . import tb_v2_runtime

    return tb_v2_runtime.install()


def _build(policy, leaf, seed: int):
    """The engine itself, after the checks (tests call it with stand-in models)."""
    from . import tb_v2_runtime, tb_v3
    from .tb_v2_value_e4v2 import E4v2AttackAgent

    agent = E4v2AttackAgent(tb_v2_runtime.make_agent(policy, leaf, seed=seed), attack=True, config=tb_v3.ATTACK)
    agent.name = NAME
    return agent


def make_tb_v3_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint):
    """TB-V3 Resolve + fast4 for one game, after every check (raises on the first failure)."""
    from . import tb_v3, track_b_guard

    tb_v3.check_identity()
    if policy_checkpoint is None or value_checkpoint is None:
        raise ValueError(f'{NAME} needs both checkpoint paths to verify its models')
    hashes = track_b_guard.check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    if hashes != tb_v3.MODELS:
        raise ValueError(f'{NAME}: the models are not the frozen ones: {hashes}')
    track_b_guard.check_runtime_sources()
    check_bound()
    return _build(policy, leaf, seed)


make_agent = make_tb_v3_agent


def provenance(device: str) -> dict:
    from rt30.native import build

    from . import tb_v3

    info = build.build_info() or {}
    return {'engine': tb_v3.ENGINE_NAME, 'engine_id': ENGINE_ID, 'engine_codename': tb_v3.ENGINE_CODENAME,
            'engine_hash': tb_v3.ENGINE_HASH, 'identity_sha256': identity_sha256(),
            'code_manifest_sha256': tb_v3.CODE_MANIFEST_SHA256, 'models': tb_v3.MODELS,
            'base_engine_hash': tb_v3.BASE_ENGINE_HASH, 'attack': tb_v3.ATTACK, 'runtime': NAME,
            'runtime_hash': runtime_hash(device), 'fast_mode': FAST_MODE, 'value_device': device_kind(device),
            'certified': CERTIFICATION.get(device_kind(device)), 'native_sources': NATIVE_SOURCES,
            'native_binaries': info.get('modules'), 'native_python': info.get('python'),
            'native_compiler': info.get('compiler_version')}
