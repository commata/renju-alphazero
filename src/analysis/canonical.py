"""Which engine is the Track B canonical: the one place that decides it.

Versioned factories never change meaning (``make_tb_v2_agent`` is TB-V2-Value for good, ``make_tb_v3_agent``
TB-V3 Resolve); only ``CANONICAL_ENGINE_ID`` moves. TB-V3 Resolve (adopted by the E4-v2 arena) is the canonical
since its promotion replay passed (``docs/h7/tb_v3_promotion.json``: 200 games, 3,201 / 3,201 moves, no decision
or record mismatch; ``check_track_b_frozen.py`` requires that record). Before it: TB-V2-Value + fast4.
Benchmark, web and self-play code that means "the canonical" builds it here.

The canonical runtime is fast5 (``analysis.tb_v3_fast5``: the same engine, the VCF search in C) since its gates
passed: G0 (4.59M VCF calls native = Python), G1 (the arena's 3,201 moves, every record field equal) and G2 (desktop,
and SER5_CERTIFIED). ``check_fast5_evidence`` re-checks those records against this checkout; fast4 stays the
rollback (``factory('TB-V3', 'fast4')``). The runtime files are not edited to record this: their SHA-256 is in the
runtime hash the evidence was measured on.
"""
from __future__ import annotations

from pathlib import Path

CANONICAL_ENGINE_ID = 'TB-V3'
ENGINES = ('TB-V2-Value', 'TB-V3')
CANONICAL_RUNTIME = 'fast5'
RUNTIMES = {'TB-V2-Value': ('fast4',), 'TB-V3': ('fast4', 'fast5')}
FAST5_EVIDENCE = {
    'g0_base': 'docs/h7/tb_v3_fast5_g0_base.json',
    'g0_e4': 'docs/h7/tb_v3_fast5_g0_e4.json',
    'g1': 'docs/h7/tb_v3_fast5_g1.json',
    'g2_desktop': 'docs/h7/tb_v3_fast5_g2_desktop.json',
    'g2_ser5': 'docs/h7/tb_v3_fast5_g2_ser5.json',
}
# What tb_v3_fast5.runtime_hash reads besides the device: equal files, equal runtime hash.
RUNTIME_HASH_FILES = ('src/analysis/tb_v3.py', 'src/analysis/tb_v3_fast5.py', 'src/analysis/tb_v2_runtime.py')
_ROOT = Path(__file__).resolve().parents[2]


def default_runtime(engine_id: str) -> str:
    """The canonical runtime for the canonical engine; an older engine keeps the runtime it was frozen on."""
    if engine_id not in ENGINES:
        raise ValueError(f'unknown engine {engine_id!r}; known: {ENGINES}')
    return CANONICAL_RUNTIME if engine_id == CANONICAL_ENGINE_ID else RUNTIMES[engine_id][0]


def factory(engine_id: str = CANONICAL_ENGINE_ID, runtime: str | None = None):
    """The versioned factory of ``engine_id`` on ``runtime`` (default: ``default_runtime``):
    (policy, leaf, seed, policy_checkpoint, value_checkpoint) -> agent."""
    runtime = runtime or default_runtime(engine_id)
    if engine_id not in ENGINES:
        raise ValueError(f'unknown engine {engine_id!r}; known: {ENGINES}')
    if runtime not in RUNTIMES[engine_id]:
        raise ValueError(f'{engine_id} has no runtime {runtime!r}; known: {RUNTIMES[engine_id]}')
    if engine_id == 'TB-V2-Value':
        from .track_b_guard import make_tb_v2_agent
        return make_tb_v2_agent
    if runtime == 'fast5':
        from .tb_v3_fast5 import make_tb_v3_fast5_agent
        return make_tb_v3_fast5_agent
    from .tb_v3_runtime import make_tb_v3_agent
    return make_tb_v3_agent


def make_canonical_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint):
    return factory()(policy, leaf, seed, policy_checkpoint, value_checkpoint)


def provenance(device: str, engine_id: str = CANONICAL_ENGINE_ID, runtime: str | None = None) -> dict:
    """The identities of ``engine_id`` on ``runtime`` (default: ``default_runtime``), as ``factory`` builds it."""
    runtime = runtime or default_runtime(engine_id)
    if engine_id not in ENGINES or runtime not in RUNTIMES[engine_id]:
        raise ValueError(f'unknown engine / runtime {engine_id!r} / {runtime!r}')
    canonical = engine_id == CANONICAL_ENGINE_ID and runtime == CANONICAL_RUNTIME
    if engine_id == 'TB-V3' and runtime == 'fast5':
        from .tb_v3_fast5 import provenance as v3_fast5
        return {**v3_fast5(device), 'canonical': canonical}
    if engine_id == 'TB-V3':
        from .tb_v3_runtime import provenance as v3
        return {**v3(device), 'canonical': canonical}
    from .tb_v2_runtime import provenance as v2
    return {**v2(device), 'canonical': canonical}


def _git_blob_sha256(commit: str, path: str) -> str:
    import hashlib
    import subprocess

    data = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=_ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def check_fast5_evidence() -> dict:
    """Raise unless the committed G0 / G1 / G2 records pass and were made on exactly this fast5 runtime."""
    import hashlib
    import json

    from . import tb_v3, tb_v3_fast5, tb_v3_runtime

    tb_v3_fast5.check_sources()
    load = {k: json.loads((_ROOT / v).read_text(encoding='utf-8')) for k, v in FAST5_EVIDENCE.items()}
    for name in ('g0_base', 'g0_e4'):
        r, fp = load[name], load[name]['fingerprint']
        if r.get('verdict') != 'G0_PASS' or r.get('git_dirty') is not False or r.get('mismatches') != 0:
            raise RuntimeError(f'fast5 evidence: {name} is not an official G0_PASS')
        if (fp['fast5_runtime_hash'] != tb_v3_fast5.runtime_hash(fp['device']) or fp['engine_hash'] != tb_v3.ENGINE_HASH
                or fp['identity_sha256'] != tb_v3_runtime.identity_sha256() or fp['stand_in']):
            raise RuntimeError(f'fast5 evidence: {name} was not measured on this fast5 runtime with the real models')
    if 'base' not in load['g0_base']['fingerprint']['parts'] or not load['g0_base']['calls']['table']:
        raise RuntimeError('fast5 evidence: the base G0 did not check the V8-C table path')
    g1 = load['g1']
    if (g1.get('format') != 'tb-v3-fast5-g1-v1' or g1.get('verdict') != 'G1_PASS' or g1.get('git_dirty') is not False
            or g1.get('runtime_hash') != tb_v3_fast5.runtime_hash('cuda') or g1['tb_v3']['record_mismatch_moves']):
        raise RuntimeError('fast5 evidence: G1 is not an official G1_PASS on this fast5 runtime')
    for name in ('g2_desktop', 'g2_ser5'):
        r = load[name]
        if r.get('official') is not True or r.get('same_decisions') is not True:
            raise RuntimeError(f'fast5 evidence: {name} is not an official G2 with the same decisions')
        for path in RUNTIME_HASH_FILES:  # the G2 tool checked the runtime hash then: the same files, the same hash
            if _git_blob_sha256(r['git_commit'], path) != hashlib.sha256((_ROOT / path).read_bytes()).hexdigest():
                raise RuntimeError(f'fast5 evidence: {path} changed since {name} ({r["git_commit"][:7]})')
    if load['g2_ser5'].get('verdict') != 'SER5_CERTIFIED':
        raise RuntimeError('fast5 evidence: SER5 is not certified')
    return {'runtime_hash_cuda': tb_v3_fast5.runtime_hash('cuda'), 'runtime_hash_cpu': tb_v3_fast5.runtime_hash('cpu'),
            'g0_calls': sum(load[k]['calls']['plain'] + load[k]['calls']['table'] for k in ('g0_base', 'g0_e4')),
            'g1_moves': g1['tb_v3']['moves_matched'], 'ser5': load['g2_ser5']['verdict']}
