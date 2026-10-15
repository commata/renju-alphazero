#!/usr/bin/env bash
# A3-v3: the E3-A 100 games with the TB-RT30-v3 candidate (workers 4), only when the gate fixed in
# docs/web-ser5.md §9.8 passes on the v3_check / v3_dev / v3_a1 results. About 2.5 h.
source "$(dirname "$0")/common.sh"
ensure_native
if ! python scripts/check_v3_gate.py gate --out "$OUT" | tee "$OUT/v3_gate.md"; then
    echo "[$(stamp)] v3 gate not met: A3-v3 not run (see $OUT/v3_gate.md)"; finish; exit 0
fi
step "$OUT/a3_v3_e3.json" python scripts/run_rt30_match.py --checkpoint "$CK" --engine v3 --workers 4 \
    --reference docs/mcts-v8-results/e3/puct_policy_vct2_50k_8415.jsonl \
                docs/mcts-v8-results/e3/puct_policy_vct2_50k_8416.jsonl \
    --games-jsonl "$OUT/a3_v3_e3.jsonl" --output "$OUT/a3_v3_e3.json"
finish
