# H6-1 v1 review and v2 decision

Reviewed after `docs/h6/h6_1_report.json`.

## v1 result

- Input / kept games: `108,011 -> 73,944`.
- Whole-game probe exclusion dropped `34,067` games (`31.54%`).
- Rows / unmasked labelled rows: `2,409,192 / 1,986,865`.
- Label kinds: outcome `1,872,867`; proofs `113,998` (`vcf 68,781`, `unstoppable_four 19,509`, `immediate_win 14,408`, `forced_loss 11,300`).
- Count sign balance is healthy: `+1 1,013,905` vs `-1 972,960` (`51.0% / 49.0%`). Black/white side-to-move counts are also close.
- Proof signs are intentionally asymmetric: proof `+1 102,698` vs proof `-1 11,300` (`90.1% / 9.9%`).
- With proof weight `1.0` and outcome weight `0.1`, the effective weighted sign mass is approximately `+1 193,818.7` vs `-1 107,466.0`, i.e. `64.3% / 35.7%`. This is not by itself a gate failure, but H6-3 must report negative-side metrics separately before deciding whether `--prove-losses` is needed.
- The v1 label set is leakage-safe under the strict rule, but it is not the preferred B1 training set because the probe rule removed too much of the opening distribution.

## Probe finding

The large drop is caused by E1 / E1-holdout control positions that are standard 3-stone RIF openings. Whole-game exclusion on those positions removes every RenjuNet game using the same opening family.

The first proposed v2 threshold, `--probe-game-min-ply 9`, was reviewed and rejected. The repository contains genuine tactical probes below that threshold; for example `tests/fixtures/stage7_probes_defense_v1.json` contains a `must_defend_open3` probe with six stones in its `moves` history. With threshold 9, the exact probe row would be masked but its later game continuation could still enter training, defeating the purpose of whole-game probe exclusion.

## Final v2 policy

- Strict default remains `--probe-game-min-ply 0`: every probe drops the whole game.
- H6-v2 may exempt only probes with at most 3 stones (the standard opening controls).
- Therefore the current H6-v2 threshold is **4**, not 9.
- Any 4+ stone probe must drop its whole game. `build_h6_labels.py` now refuses a threshold that would make a 4+ stone probe row-only, so future probe additions cannot silently weaken the leakage rule.
- Run `h6_probe_audit.py` first and compare thresholds 0 and 4. The audit remains a count-only artifact and is safe to commit.

Recommended v2 build:

```powershell
python -u scripts\build_h6_labels.py --games $GAMES --workers 12 --probe-game-min-ply 4 `
  --output data\tactical\h6\renjunet_labels_v2.jsonl.gz `
  --report runs\h6\h6_1_report_v2.json 2>&1 | Tee-Object runs\h6\h6_1_v2.log
```

Do not start H6-2a from v1. Use the v2 report/cache paths already configured in `configs/hybrid/h6_value_head.yaml`, then run the H6-1 gate again on the v2 report before full value-head training.
