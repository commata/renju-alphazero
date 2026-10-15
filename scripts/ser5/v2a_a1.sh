#!/usr/bin/env bash
# v2-A on the A1 gate sets (as the cap 8 runs: workers 1, seed 42, the H3 fast baselines),
# the full-budget recheck of its changed moves, and the table beside TB-RT30-v1 (cap 8). About 1 h.
source "$(dirname "$0")/common.sh"
R=docs/rt30-results
step "$OUT/v2a_h3_s60.json" python scripts/run_rt30_replay.py --checkpoint "$CK" --arms rt2 --deadline 27.5 \
    --keys-file docs/rt30-eval/slow-stratified-v1.json --workers 1 --baseline "$R/fast_h3_s60.json" \
    --output "$OUT/v2a_h3_s60.json"
step "$OUT/v2a_h3_crit.json" python scripts/run_rt30_replay.py --checkpoint "$CK" --arms rt2 --deadline 27.5 \
    --keys-file docs/rt30-eval/critical-regressions-v1.json --workers 1 --baseline "$R/fast_h3_crit.json" \
    --output "$OUT/v2a_h3_crit.json"
step "$OUT/recheck_v2a_h3_s60.json" python scripts/check_rt30_changed_moves.py --run "$OUT/v2a_h3_s60.json" \
    --workers 4 --output "$OUT/recheck_v2a_h3_s60.json"
step "$OUT/recheck_v2a_h3_crit.json" python scripts/check_rt30_changed_moves.py --run "$OUT/v2a_h3_crit.json" \
    --workers 4 --output "$OUT/recheck_v2a_h3_crit.json"
python scripts/compare_rt30_runs.py \
    v1_cap8="$R/cap8_h3_slow-stratified-v1.json:$R/recheck_cap8_h3_slow-stratified-v1.json" \
    v2a="$OUT/v2a_h3_s60.json:$OUT/recheck_v2a_h3_s60.json" | tee "$OUT/v2a_compare_s60.md"
python scripts/compare_rt30_runs.py \
    v1_cap8="$R/cap8_h3_critical-regressions-v1.json:$R/recheck_cap8_h3_critical-regressions-v1.json" \
    v2a="$OUT/v2a_h3_crit.json:$OUT/recheck_v2a_h3_crit.json" | tee "$OUT/v2a_compare_crit.md"
finish
