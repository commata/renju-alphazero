# Working notes for Claude sessions

- Docs and replies are in Korean (the user's language); code, comments and commit messages in English.
- Long offline jobs: any script whose expected run time is an hour or more must print progress to stdout
  (`flush=True`, not stderr: Windows PowerShell `2>&1 | Tee-Object` turns stderr into error records):
  done / total, percent, elapsed and an ETA, about once a minute (`src/hybrid/progress.py`).
  With a process pool, collect results as they complete (`run_chunks`) or wait with a timeout and call
  `Progress.heartbeat`, so one slow task never silences the log; never rely on ordered `pool.map` alone.
- RenjuNet-derived data (rows, caches, models trained on it) stays under `data/` or `runs/` and is never committed.
- Do not edit frozen engine files (`src/analysis/tb_engine_v1.py`, `mcts_v8.py`, `e3_arms.py`, `s3_vct2_v1.py`,
  `puct_v8.py`, `threats.py`) outside an explicit engine-version step.
- TB-RT30-v1 (the web engine) is frozen the same way: the files in `tests/frozen_tb_rt30_v1.sha256`
  (`src/rt30/engine.py`, `solver.py`, `deadline.py`, `vcf.py`, `fast*.py`, `tb_rt30_v1.py`); a change is TB-RT30-v2.
