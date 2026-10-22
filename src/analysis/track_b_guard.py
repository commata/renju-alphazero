"""Run-time guards for the frozen Track B canonical (TB-V2-Value + fast4), outside the frozen files.

The frozen identity files cannot change without a new engine / runtime step (CLAUDE.md), and they leave
four checks to their callers (docs/h6-value.md, "동결 보증 점검"):
    policy     ``tb_v2_value.config_hash`` does not cover the H3 policy and ``make_agent`` does not check it:
               ``check_policy_checkpoint`` (both SHA-256, ``s3_vct2_v1.check_checkpoint``)
    value      ``tb_v2_value.make_agent`` accepts a leaf without a ``sha256``: ``check_leaf`` refuses it
    sources    ``tb_v2_runtime.check_bound`` checks the native build, not the Python binding files:
               ``check_runtime_sources`` hashes every native and binding source of the profile
    code       no engine hash covers the code: ``check_code`` compares every file of the canonical import
               closure with ``tests/track_b_frozen.sha256``
``make_canonical_agent`` runs them all, then ``tb_v2_runtime.make_agent`` (engine hash, fast4 binding).
The next engine freeze puts the policy SHA into its hash and fails closed in its own ``make_agent``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tests' / 'track_b_frozen.sha256'


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_policy_checkpoint(checkpoint, policy=None) -> dict:
    """Both SHA-256 of the H3 checkpoint must be the frozen ones; a ``RootPolicy`` must have loaded that file."""
    from .s3_vct2_v1 import check_checkpoint

    hashes = check_checkpoint(checkpoint)
    loaded = getattr(policy, 'checkpoint', None)
    if policy is not None and (loaded is None or Path(loaded).resolve() != Path(checkpoint).resolve()):
        raise ValueError(f'the policy was not loaded from {checkpoint} (it reports {loaded})')
    return hashes


def check_leaf(leaf) -> str:
    """Fail closed: the leaf must report exactly the frozen value checkpoint SHA-256."""
    from .tb_v2_value import VALUE_CHECKPOINT_SHA256

    sha = getattr(leaf, 'sha256', None)
    if sha != VALUE_CHECKPOINT_SHA256:
        raise ValueError(f'the value leaf reports {sha!r}, the frozen checkpoint is {VALUE_CHECKPOINT_SHA256[:12]}...')
    return sha


def check_runtime_sources() -> None:
    from . import tb_v2_runtime as rt

    for name, digest in rt.NATIVE_SOURCES.items():
        if _sha(ROOT / 'src' / 'rt30' / 'native' / name) != digest:
            raise RuntimeError(f'{rt.NAME}: src/rt30/native/{name} is not the frozen source')
    for name, digest in rt.BINDING_SOURCES.items():
        if _sha(ROOT / name) != digest:
            raise RuntimeError(f'{rt.NAME}: {name} is not the frozen binding source')


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            digest, name = line.split(maxsplit=1)
            out[name] = digest
    return out


def check_code(path: Path = MANIFEST) -> list[str]:
    """The files of the manifest that differ from it or are missing ([] when the code is the frozen one)."""
    bad = []
    for name, digest in read_manifest(path).items():
        file = ROOT / name
        if not file.exists() or _sha(file) != digest:
            bad.append(name)
    return bad


def make_canonical_agent(policy, leaf, seed: int, policy_checkpoint):
    """TB-V2-Value + fast4 after every check above (raises on the first failure)."""
    from . import tb_v2_runtime as rt

    check_policy_checkpoint(policy_checkpoint, policy)
    check_leaf(leaf)
    check_runtime_sources()
    bad = check_code()
    if bad:
        raise RuntimeError(f'the canonical code differs from {MANIFEST.name}: {bad}')
    return rt.make_agent(policy, leaf, seed=seed)
