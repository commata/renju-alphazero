"""Opening-suite generation seeds for the R2-S400-PRESERVE-1 development suites (docs/h7r/r2/R2-S400-PRESERVE-1-draft.md,
section 5), registered apart from ``hybrid.h7r_seeds``.

``hybrid.h7r_seeds`` (v2) is a source the R2-S400-TRAIN contract pins by SHA-256, so it is not edited: these two
generation seeds live here. They only draw openings (``hybrid.h7r_openings.build_suite``); the Arena game seeds of the
contract that plays these suites stay in the v2 Arena block and are checked by ``hybrid.h7r_eval_v2`` as before.

Registered 2026-11-19, before either suite was built.
"""
from __future__ import annotations

POLICY = 'h7r-suite-seeds-dev2'
REGISTERED = '2026-11-19'
SUITE_SEEDS = {
    202611191: 'docs/h7r/suites/O1-dev-c.json (R2-S400-PRESERVE-1 development suite c)',
    202611192: 'docs/h7r/suites/O1-dev-d.json (R2-S400-PRESERVE-1 development suite d)',
}


def conflicts() -> dict:
    """Every seed here that collides with something it must not (empty lists when the registry is sound): the v1
    reservations, the v2 Arena and suite seeds, the self-play training run seeds, their measurement seed and the S1
    probe base seed."""
    from hybrid import h7r_seeds as v2
    from hybrid.h6_labels import RESERVED_SEEDS

    mine = set(SUITE_SEEDS)
    return {'v1_reserved': sorted(mine & set(RESERVED_SEEDS)),
            'v2_arena_seeds': sorted(mine & set(v2.ARENA_SEEDS)),
            'v2_suite_seeds': sorted(mine & set(v2.SUITE_SEEDS)),
            'training_run_seeds': sorted(mine & set(v2.TRAINING_RUN_SEEDS)),
            'measure_seed': sorted(mine & {v2.MEASURE_SEED}),
            's1_probe_base_seed': sorted(mine & {v2.S1_PROBE_BASE_SEED})}
