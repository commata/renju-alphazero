# Working notes for Claude sessions

- Docs and replies are in Korean (the user's language); code, comments and commit messages in English.
- Long offline jobs: any script whose expected run time is an hour or more must print progress to stdout
  (`flush=True`, not stderr: Windows PowerShell `2>&1 | Tee-Object` turns stderr into error records):
  done / total, percent, elapsed and an ETA, about once a minute (`src/hybrid/progress.py`).
  With a process pool, collect results as they complete (`run_chunks`) or wait with a timeout and call
  `Progress.heartbeat`, so one slow task never silences the log; never rely on ordered `pool.map` alone.
- RenjuNet-derived data (rows, caches, models trained on it) stays under `data/` or `runs/` and is never committed.
- Do not edit frozen engine files (`src/analysis/tb_engine_v1.py`, `tb_engine_v2.py`, `tb_v2_value.py`, `vcf_tt_v1.py`,
  `mcts_v8.py`, `e3_arms.py`, `s3_vct2_v1.py`, `puct_v8.py`, `puct_value_v1.py`, `threats.py`;
  `src/hybrid/h6_value_leaf.py`) outside an explicit engine-version step.
- The frozen runtime profile `src/analysis/tb_v2_runtime.py` (TB-V2-Value + fast4) also locks `src/rt30/native/*.c`
  and `src/rt30/fast*.py` by SHA-256: change them only as a new runtime step (same-move equivalence gate first).
- The H7-E4 candidate `src/analysis/tb_v2_value_e4.py` locks `src/analysis/e4_attack.py`, `vct2_attack.py` and
  `selective_vct.py` by SHA-256 (`ATTACK_SOURCES`), and `src/analysis/e4_reference.py` is the arena's frozen event
  detector once its DEV check passes: do not edit them until the seed 8441 / 8442 arena has given ADOPT / REJECT.
  For E4-v2 the detector is frozen at its `'4m'` budget (`docs/h7/h7_e4v2_reference_4m.json`).
- `scripts/check_track_b_frozen.py` checks the frozen engines (code manifests, engine hashes, runtime sources and
  binding, policy / value checkpoints, TB-V3 identity and promotion): run it before benchmark, web or self-play work.
  Canonical runs go through `analysis.canonical.make_canonical_agent` (the E4 tools keep `track_b_guard`'s TB-V2 factory).
- The H7-E4-v2 candidate `src/analysis/tb_v2_value_e4v2.py` locks `src/analysis/e4v2_attack.py` (with the E4 attack
  files above) and the code manifest by SHA-256. Its arena verdict is ADOPT: it is frozen for good as part of TB-V3.
- TB-V3 Resolve (`src/analysis/tb_v3.py`, `tb_v3_runtime.py`) puts the SHA-256 of its code manifest
  `tests/tb_v3_frozen.sha256` (every file of its factory's import closure, including `track_b_guard.py`) in its engine
  hash, and `tb_v3.py` in its runtime hash: do not edit any of them, or `tests/track_b_frozen.sha256`, outside a new
  engine-version step (`check_track_b_frozen.py --write-manifest tb-v3` only there). `src/analysis/canonical.py` is the
  one place that says which engine is the canonical; TB-V3 becomes it only after its promotion replay
  (`docs/h7/tb_v3_promotion.json`, PROMOTION_PASS). Versioned factories (`make_tb_v2_agent`, `make_tb_v3_agent`) never
  change meaning.
- Exact runtime speed-ups of TB-V3 keep its engine hash and get their own runtime identity: fast5
  (`src/analysis/tb_v3_fast5.py`, `src/rt30/fast5.py`, `src/rt30/vcfcore/`) locks its sources by SHA-256
  (`FAST5_SOURCES`; the native VCF core includes the frozen `rt30/native/forbidden.c`). It passed G0 / G1 / G2
  (`docs/h7/tb_v3_fast5_*.json`) and is the canonical runtime (`canonical.CANONICAL_RUNTIME = 'fast5'`;
  `canonical.check_fast5_evidence`, run by the checker, re-checks those records against the checkout). Never edit the
  fast5 files or `tb_v3.py`: their SHA-256 is in the runtime hash the evidence was measured on. fast4 stays the
  rollback. CPU use of fast5 needs its own certification (seeds 8451 / 8452); a fast4 one does not carry over.
  Build it with `python -m rt30.vcfcore.build --if-stale` (in `src/`); `rt30/native/build.py` stays untouched.
- H7-c candidate models run only through `analysis.tb_v3_net.make_tb_v3_net_agent` (the frozen TB-V3 engine and fast5
  runtime, models recorded by SHA-256, never canonical). Only `scripts/h7c_*` tools may use it: the web, benchmarks and
  `analysis.canonical` keep the frozen H3 policy / H6 value until a separate promotion step.
- TB-RT30-v1 (the web engine) is frozen the same way: the files in `tests/frozen_tb_rt30_v1.sha256`
  (`src/rt30/engine.py`, `solver.py`, `deadline.py`, `vcf.py`, `fast*.py`, `tb_rt30_v1.py`); a change is TB-RT30-v2.
