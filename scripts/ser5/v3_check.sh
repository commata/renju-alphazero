#!/usr/bin/env bash
# TB-RT30-v3 check (docs/web-ser5.md §9.8): build the native scan, TB1 = TB1-fast3 = v3 without a
# deadline on the 8 H3 equivalence positions, TB1's own time with the native scan at the 34 A3
# divergence moves (the measured speed-up), and the recovery bound for that speed. About 40 min.
source "$(dirname "$0")/common.sh"
ensure_native
KEYS="puct_policy_vct2_50k_8415/2/14 puct_policy_vct2_50k_8415/9/18 puct_policy_vct2_50k_8415/9/26
puct_policy_vct2_50k_8415/10/18 puct_policy_vct2_50k_8415/10/20 puct_policy_vct2_50k_8415/11/9
puct_policy_vct2_50k_8415/11/11 puct_policy_vct2_50k_8415/11/13"
# shellcheck disable=SC2086
step "$OUT/v3_equivalence_h3_8.json" python scripts/run_rt30_replay.py --checkpoint "$CK" --arms frozen,fast3,rt3 \
    --keys $KEYS --workers 4 --output "$OUT/v3_equivalence_h3_8.json"
step "$OUT/a31_tb1f3_w4.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine tb1 --speedups fast3 \
    --workers 4 --games-jsonl "$OUT/a31_tb1f3_w4.jsonl" --output "$OUT/a31_tb1f3_w4.json"
python scripts/predict_rt30_recovery.py --divergence "$OUT/a31_tb1f3_w4.json" > "$OUT/predict_recovery_tb1f3_w4.json"
python scripts/check_v3_gate.py speedup --fast2 docs/rt30-results/a31_tb1_w4.json --fast3 "$OUT/a31_tb1f3_w4.json" \
    | tee "$OUT/v3_speedup.md"
finish
