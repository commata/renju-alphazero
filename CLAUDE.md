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
- `scripts/check_track_b_frozen.py` checks the canonical (code manifest `tests/track_b_frozen.sha256`, engine hash,
  runtime sources and binding, policy / value checkpoints): run it before benchmark, web or self-play work; regenerate
  the manifest (`--write-manifest`) only in an explicit engine or runtime freeze step. Canonical runs go through
  `analysis.track_b_guard.make_canonical_agent`.
- The H7-E4-v2 candidate `src/analysis/tb_v2_value_e4v2.py` locks `src/analysis/e4v2_attack.py` (with the E4 attack
  files above) and the code manifest by SHA-256: do not edit them until its COSTTEST / arena verdict.
