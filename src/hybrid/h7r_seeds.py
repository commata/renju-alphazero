"""H7-R seed reservation v2 (docs/h7r/r2/R2-S400-TRAIN-draft.md, section 5).

The v1 H7-R block 8781-8799 (``hybrid.h7r_eval.SEED_BLOCK``, inside ``hybrid.h6_labels.RESERVED_SEEDS``) is used up.
Both files are sources that earlier experiment contracts pin by SHA-256, so v2 is a separate registry: it changes
neither file, and only the v2 contract path (``hybrid.h7r_eval_v2``) applies it. Arena records of v2 contracts never
become training data for the same reason v1 ones do not: the training label step reads only self-play files of a clean
generator with a training run seed (``scripts/h7c_labels.check_games``), and no v2 seed is a training run seed.

    ARENA_SEEDS  game seeds of v2 evaluation contracts (one contract per seed, pairings aside)
    SUITE_SEEDS  opening-suite generation seeds (``hybrid.h7r_openings.build_suite``), each used by one suite only

Registered 2026-11-13, before any v2 contract or suite.
"""
from __future__ import annotations

POLICY = 'h7r-seeds-v2'
REGISTERED = '2026-11-13'
ARENA_SEEDS = range(9101, 9141)
SUITE_SEEDS = {
    202611131: 'docs/h7r/suites/O1-dev-a.json (R2-S400-TRAIN development suite a)',
    202611132: 'docs/h7r/suites/O1-dev-b.json (R2-S400-TRAIN development suite b)',
}
# Seeds v2 must never touch (checked by ``conflicts`` and the tests): the v1 reservations, the self-play training run
# seeds and their measurement seed, the S1 probe base seed.
TRAINING_RUN_SEEDS = range(8801, 9000)  # scripts/h7c_selfplay.RUN_SEEDS
MEASURE_SEED = 8800                     # scripts/h7c_selfplay.MEASURE_SEED
S1_PROBE_BASE_SEED = 9201               # scripts/run_s1_probes.py --base-seed default


def require(ok, message) -> None:
    if not ok:
        raise ValueError(message)


def check_arena_seed(seed) -> None:
    require(type(seed) is int and seed in ARENA_SEEDS, f'seed {seed!r} is not a v2 Arena seed ({ARENA_SEEDS.start}-'
                                                       f'{ARENA_SEEDS.stop - 1}, {POLICY})')


def conflicts() -> dict:
    """Every v2 seed that collides with something it must not (empty lists when the registry is sound)."""
    from hybrid.h6_labels import RESERVED_SEEDS

    v2 = set(ARENA_SEEDS) | set(SUITE_SEEDS)
    return {'v1_reserved': sorted(v2 & set(RESERVED_SEEDS)),
            'training_run_seeds': sorted(v2 & set(TRAINING_RUN_SEEDS)),
            'measure_seed': sorted(v2 & {MEASURE_SEED}),
            's1_probe_base_seed': sorted(v2 & {S1_PROBE_BASE_SEED}),
            'arena_and_suite_seeds': sorted(set(ARENA_SEEDS) & set(SUITE_SEEDS))}
