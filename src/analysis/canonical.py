"""Which engine is the Track B canonical: the one place that decides it.

Versioned factories never change meaning (``make_tb_v2_agent`` is TB-V2-Value for good, ``make_tb_v3_agent``
TB-V3 Resolve); only ``CANONICAL_ENGINE_ID`` moves. TB-V3 Resolve (adopted by the E4-v2 arena) is the canonical
since its promotion replay passed (``docs/h7/tb_v3_promotion.json``: 200 games, 3,201 / 3,201 moves, no decision
or record mismatch; ``check_track_b_frozen.py`` requires that record). Before it: TB-V2-Value + fast4.
Benchmark, web and self-play code that means "the canonical" builds it here.
"""
from __future__ import annotations

CANONICAL_ENGINE_ID = 'TB-V3'
ENGINES = ('TB-V2-Value', 'TB-V3')
# The canonical engine's runtime: fast4. fast5 (exact, analysis.tb_v3_fast5) replaces it only after its G1 / G2.
CANONICAL_RUNTIME = 'fast4'
RUNTIMES = {'TB-V2-Value': ('fast4',), 'TB-V3': ('fast4', 'fast5')}


def factory(engine_id: str = CANONICAL_ENGINE_ID, runtime: str | None = None):
    """The versioned factory of ``engine_id`` on ``runtime`` (default: the canonical runtime):
    (policy, leaf, seed, policy_checkpoint, value_checkpoint) -> agent."""
    runtime = runtime or CANONICAL_RUNTIME
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


def provenance(device: str, engine_id: str = CANONICAL_ENGINE_ID) -> dict:
    if engine_id == 'TB-V3':
        from .tb_v3_runtime import provenance as v3
        return {**v3(device), 'canonical': engine_id == CANONICAL_ENGINE_ID}
    from .tb_v2_runtime import provenance as v2
    return {**v2(device), 'canonical': engine_id == CANONICAL_ENGINE_ID}
