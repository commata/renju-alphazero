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
