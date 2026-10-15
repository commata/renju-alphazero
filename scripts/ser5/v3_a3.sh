#!/usr/bin/env bash
# A3-v3: the E3-A 100 games with the TB-RT30-v3 candidate (workers 4), only when the gate fixed in
# docs/web-ser5.md §9.8 passes on the v3_check / v3_dev / v3_a1 results. About 2.5 h.
# FAIL stops here. REVIEW (changed control moves, all cut by the deadline) stops too, until every
# listed move has been reviewed by hand and the job is started again with V3_REVIEWED=1.
source "$(dirname "$0")/common.sh"
ensure_native
python scripts/check_v3_gate.py gate --out "$OUT" | tee "$OUT/v3_gate.md"
gate=${PIPESTATUS[0]}
if [ "$gate" -eq 2 ] && [ "${V3_REVIEWED:-0}" = 1 ]; then
    echo "[$(stamp)] v3 gate REVIEW, reviewed by hand (V3_REVIEWED=1): running A3-v3"
elif [ "$gate" -eq 2 ]; then
    echo "[$(stamp)] v3 gate REVIEW: changed control moves need a review by hand (see $OUT/v3_gate.md); A3-v3 not run"
    finish; exit 0
elif [ "$gate" -ne 0 ]; then
    echo "[$(stamp)] v3 gate not met: A3-v3 not run (see $OUT/v3_gate.md)"; finish; exit 0
fi
step "$OUT/a3_v3_e3.json" python scripts/run_rt30_match.py --checkpoint "$CK" --engine v3 --workers 4 \
    --reference docs/mcts-v8-results/e3/puct_policy_vct2_50k_8415.jsonl \
                docs/mcts-v8-results/e3/puct_policy_vct2_50k_8416.jsonl \
    --games-jsonl "$OUT/a3_v3_e3.jsonl" --output "$OUT/a3_v3_e3.json"
finish
