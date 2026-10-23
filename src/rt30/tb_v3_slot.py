"""The web's latest-certified opponent: TB-V3 Resolve / fast5, frozen H3 policy / H6 value, value on CPU.

"Latest-certified" is not the newest checkpoint: it is the newest engine, runtime and model set that passed its
promotion gates and its deployment gates (docs/h6-value.md, H7-c1 design approval). Today that is the canonical of
``analysis.canonical`` (TB-V3 Resolve on fast5) with the frozen models of ``analysis.tb_v3.MODELS``:
    promotion   TB-V3 promotion replay (PROMOTION_PASS), fast5 G0 / G1 / G2 desktop
    deployment  fast5 G2 SER5 (SER5_CERTIFIED, worker 1), fast5 CPU certification (PASS, seeds 8451 / 8452)
An H7-c checkpoint never becomes this opponent by being trained: only a new promotion and deployment step that
changes ``LATEST_CERTIFIED`` does.

``check_deployment`` runs in the web process before any worker starts and fails closed: the canonical is still
TB-V3 / fast5, every evidence record holds for this checkout (``canonical.check_fast5_evidence``), TB-V3's
identity, both model files are the frozen ones and both native builds are fresh. ``make_agent`` runs in the engine
worker (``rt30.supervisor.EngineSupervisor``): fast4 then fast5 bound in that process, the models loaded once on the
CPU with one thread, and the canonical factory, which checks the identity, models and runtime again.

TB-V3 has no deadline (node budgets only; SER5: max 6.9 s per move over 827 positions). The web keeps a 30 s soft
monitor (logged, never cut) and no move fallback: a worker that does not answer is an error, not a played move.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

KEY = 'tbv3'
LATEST_CERTIFIED = {'engine_id': 'TB-V3', 'runtime': 'fast5', 'value_device': 'cpu', 'models': 'tb_v3.MODELS',
                    'ai_slots': 1, 'soft_limit_seconds': 30.0}
LABEL = '최신 인증 모델 - TB-V3 Resolve / fast5'
SOFT_LIMIT_SECONDS = LATEST_CERTIFIED['soft_limit_seconds']
HANG_SECONDS = 300.0  # a worker silent this long is hung: the move is an error (no fallback), the worker replaced


def check_deployment(policy_checkpoint, value_checkpoint) -> dict[str, Any]:
    """Raise unless this checkout and these model files are the latest-certified deployment; its provenance."""
    from analysis import canonical, tb_v3, tb_v3_fast5, track_b_guard
    from rt30.native import build as native_build
    from rt30.vcfcore import build as vcf_build

    if (canonical.CANONICAL_ENGINE_ID, canonical.CANONICAL_RUNTIME) != (LATEST_CERTIFIED['engine_id'],
                                                                        LATEST_CERTIFIED['runtime']):
        raise RuntimeError(f'the canonical is {canonical.CANONICAL_ENGINE_ID} / {canonical.CANONICAL_RUNTIME}, '
                           f'the latest-certified deployment is TB-V3 / fast5')
    evidence = canonical.check_fast5_evidence()
    tb_v3.check_identity()
    models = {**track_b_guard.check_policy_checkpoint(policy_checkpoint),
              **track_b_guard.check_value_checkpoint(value_checkpoint)}
    if models != tb_v3.MODELS:
        raise RuntimeError(f'the model files are not the certified ones: {models}')
    for name, build in (('rt30.native', native_build), ('rt30.vcfcore', vcf_build)):
        reason = build.stale()
        if reason is not None:
            raise RuntimeError(f'{name} is stale ({reason}): (cd src && python -m {name}.build --if-stale)')
    device = LATEST_CERTIFIED['value_device']
    return {'engine': f"{tb_v3.ENGINE_NAME}/{LATEST_CERTIFIED['runtime']}", 'engine_id': tb_v3.ENGINE_ID,
            'engine_config_hash': tb_v3.ENGINE_HASH, 'runtime': LATEST_CERTIFIED['runtime'],
            'runtime_hash': tb_v3_fast5.runtime_hash(device), 'fast_mode': tb_v3_fast5.FAST_MODE,
            'value_device': device, 'model': 'latest-certified', **models,
            'policy_checkpoint': str(policy_checkpoint), 'value_checkpoint': str(value_checkpoint),
            'evidence': {**canonical.FAST5_EVIDENCE, 'promotion': 'docs/h7/tb_v3_promotion.json'},
            'evidence_summary': evidence, 'native_build': native_build.build_info(),
            'vcf_core_build': vcf_build.build_info()}


_MODELS: dict[tuple[str, str], Any] = {}


def _models(policy_checkpoint: str, value_checkpoint: str):
    key = (str(policy_checkpoint), str(value_checkpoint))
    if key not in _MODELS:
        from hybrid.h4_policy import RootPolicy
        from hybrid.h6_value_leaf import ValueLeaf

        _MODELS[key] = (RootPolicy(policy_checkpoint, threads=1),
                        ValueLeaf(value_checkpoint, device=LATEST_CERTIFIED['value_device'], threads=1))
    return _MODELS[key]


def make_agent(seed: int = 42, *, policy_checkpoint: str, value_checkpoint: str):
    """The latest-certified TB-V3 agent for one web game, in the engine worker (fails closed)."""
    from analysis import canonical, tb_v2_runtime, tb_v3_fast5
    from rt30 import fast5

    if not fast5.installed():
        tb_v2_runtime.install()
        tb_v3_fast5.install()
    policy, leaf = _models(policy_checkpoint, value_checkpoint)
    factory = canonical.factory(LATEST_CERTIFIED['engine_id'], LATEST_CERTIFIED['runtime'])
    agent = factory(policy, leaf, seed, Path(policy_checkpoint), Path(value_checkpoint))
    agent.name = f"{agent.name} (CPU)"
    return agent
