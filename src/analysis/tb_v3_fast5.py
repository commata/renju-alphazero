"""TB-V3 Resolve + fast5: an exact runtime candidate of TB-V3 (X2-F; docs/h6-value.md, "TB-V3 실행 최적화").

Same engine (``analysis.tb_v3``: ``ENGINE_HASH`` unchanged, its 63-file manifest untouched), another runtime:
fast4 plus the whole VCF search in C (``rt30.fast5``, ``rt30/vcfcore/vcf_core.c``). Not the canonical runtime
until its gates pass (approved before running):
    G0  native vs Python differential (``tests/test_fast5.py``, ``scripts/fast5_g0.py`` shadow runs)
    G1  the E4-v2 arena's 200 games / 3,201 moves replayed on fast5 (``scripts/tb_v3_promotion.py --runtime
        fast5``): moves, decisions and every record field but times equal (charged VCF nodes, uncached VCF
        calls, cuts, statuses, V8-C table hits / stores, E4-v2 steps); low-level call counts may differ
    G2  speed on the desktop (fast4 vs fast5, same work) and on SER5 (two rounds, no move over 30 s, max <= 25 s)
A CPU certification of TB-V3 / fast4 does not carry over: fast5 on CPU is checked again before it is used there.

``runtime_hash`` covers the engine hash, ``tb_v3.py``, this file, fast5, the fast4 sources and bindings, the
native VCF core and its builder, and the value device. ``make_tb_v3_fast5_agent`` fails closed like the fast4
factory (identity, models, all runtime sources, both native builds and the bindings) and then builds the same
agent (``tb_v3_runtime._build``).
"""
from __future__ import annotations

from pathlib import Path

from .tb_v2_runtime import BINDING_SOURCES, NATIVE_SOURCES, device_kind  # noqa: F401  (fast4, unchanged)

NAME = 'TB-V3 Resolve/fast5'
ENGINE_ID = 'TB-V3'
FAST_MODE = 'fast5'
FAST5_SOURCES = {
    'src/rt30/vcfcore/vcf_core.c': 'c9fcfec0751b6f3b8286d694f925a854b6c5e307e2ab47d4ea1149ac46d474d5',
    'src/rt30/native/forbidden.c': '211fd89d65db9963b8d1c31f09c4d235f927ad6744010672b0ae6c70d3435822',
    'src/rt30/fast5.py': '891f3edb9ea2dff561da5412b78aa2f901733a877a3f3125ec557bcc44ea9289',
    'src/rt30/vcfcore/build.py': 'ffafbc5da1dc62c90cbbaa23fb00baa34e7679ecc233ad9d967f0cf128d07572',
    'src/rt30/vcfcore/__init__.py': 'cf781caf4f2579ec9e770836128c807f451a1ed4b1040b396c7e80d7c1ba13a7',
}
CERTIFICATION = {'cuda': None, 'cpu': None}  # set only by its own G1 / G2 records
_ROOT = Path(__file__).resolve().parents[2]
_FILE = Path(__file__).resolve()


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_file_sha256() -> str:
    return _sha(_FILE)


def check_sources() -> None:
    for name, digest in FAST5_SOURCES.items():
        path = _ROOT / name
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f'{NAME}: {name} is not the frozen fast5 source')


def runtime_hash(device: str) -> str:
    import hashlib
    import json

    from . import tb_v3, tb_v3_runtime

    payload = {'engine_id': ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH,
               'identity_sha256': tb_v3_runtime.identity_sha256(), 'runtime_file_sha256': runtime_file_sha256(),
               'fast_mode': FAST_MODE, 'native_sources': NATIVE_SOURCES, 'binding_sources': BINDING_SOURCES,
               'fast5_sources': FAST5_SOURCES, 'value_device': device_kind(device)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_bound() -> None:
    """fast4 bound (``tb_v2_runtime.check_bound``) and the native VCF core bound, fresh and of these sources."""
    from rt30 import fast5
    from rt30.vcfcore import build

    from . import tb_v2_runtime

    tb_v2_runtime.check_bound()
    if not fast5.installed():
        raise RuntimeError(f'{NAME} needs fast5 installed (rt30.fast5.install)')
    reason = build.stale()
    if reason is not None:
        raise RuntimeError(f'{NAME}: the native VCF core is stale ({reason}); python -m rt30.vcfcore.build --if-stale')
    sources = (build.build_info() or {}).get('sources', {})
    want = {'vcf_core.c': FAST5_SOURCES['src/rt30/vcfcore/vcf_core.c'],
            'forbidden.c': FAST5_SOURCES['src/rt30/native/forbidden.c']}
    if sources != want:
        raise RuntimeError(f'{NAME}: the native VCF core was built from other sources: {sources}')


def install() -> dict:
    from rt30 import fast5

    check_sources()
    bound = fast5.install()
    check_bound()
    return bound


def make_tb_v3_fast5_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint):
    """TB-V3 Resolve on fast5 for one game, after every check (raises on the first failure)."""
    from . import tb_v3, tb_v3_runtime, track_b_guard

    tb_v3.check_identity()
    if policy_checkpoint is None or value_checkpoint is None:
        raise ValueError(f'{NAME} needs both checkpoint paths to verify its models')
    hashes = track_b_guard.check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    if hashes != tb_v3.MODELS:
        raise ValueError(f'{NAME}: the models are not the frozen ones: {hashes}')
    track_b_guard.check_runtime_sources()
    check_sources()
    check_bound()
    agent = tb_v3_runtime._build(policy, leaf, seed)
    agent.name = NAME
    return agent


def provenance(device: str) -> dict:
    from rt30.vcfcore import build

    from . import tb_v3_runtime

    base = tb_v3_runtime.provenance(device)
    info = build.build_info() or {}
    return {**base, 'runtime': NAME, 'fast_mode': FAST_MODE, 'runtime_hash': runtime_hash(device),
            'runtime_file_sha256': runtime_file_sha256(), 'fast5_sources': FAST5_SOURCES,
            'fast4_runtime_hash': base['runtime_hash'], 'certified': CERTIFICATION.get(device_kind(device)),
            'vcf_core_binaries': info.get('modules'), 'vcf_core_python': info.get('python'),
            'vcf_core_compiler': info.get('compiler_version')}
