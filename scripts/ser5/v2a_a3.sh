#!/usr/bin/env bash
# A3-v2A: the E3-A 100 games with the TB-RT30-v2 candidate, as A3 (workers 4). About 2.5-3 h.
source "$(dirname "$0")/common.sh"
step "$OUT/a3_v2a_e3.json" python scripts/run_rt30_match.py --checkpoint "$CK" --engine v2 --workers 4 \
    --reference docs/mcts-v8-results/e3/puct_policy_vct2_50k_8415.jsonl \
                docs/mcts-v8-results/e3/puct_policy_vct2_50k_8416.jsonl \
    --games-jsonl "$OUT/a3_v2a_e3.jsonl" --output "$OUT/a3_v2a_e3.json"
finish
