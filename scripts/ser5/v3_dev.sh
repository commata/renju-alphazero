#!/usr/bin/env bash
# TB-RT30-v3 at the 34 A3 divergence moves with 4, 2 and 1 searches at once (X2 runs 2 processes per
# search: the CPU check), and at the 26 root-control moves. About 2.5 h.
source "$(dirname "$0")/common.sh"
ensure_native
for w in 4 2 1; do
    step "$OUT/a31_v3_w$w.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine v3 \
        --workers "$w" --games-jsonl "$OUT/a31_v3_w$w.jsonl" --output "$OUT/a31_v3_w$w.json"
done
step "$OUT/ctrl_v3_w4.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine v3 \
    --manifest docs/rt30-eval/a3-root-control-v1.json --workers 4 --games-jsonl "$OUT/ctrl_v3_w4.jsonl" \
    --output "$OUT/ctrl_v3_w4.json"
finish
