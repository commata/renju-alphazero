#!/usr/bin/env bash
# The A3-v3 rt30_no_deadline move (8416/23 white, ply 21; docs/web-ser5.md §9.10), replayed from the
# game's opening with a trace of every own move: TB-Engine-v1 + fast3 (no deadline), v3 with no time
# pressure (deadline 100000 s) and v3 at 27.5 s, one search at a time. About 15-30 min.
source "$(dirname "$0")/common.sh"
ensure_native
M=docs/rt30-eval/a3v3-no-deadline-v1.json
step "$OUT/nd_tb1.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine tb1 --speedups fast3 \
    --manifest "$M" --trace --workers 1 --output "$OUT/nd_tb1.json"
step "$OUT/nd_v3_free.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine v3 --deadline 100000 \
    --manifest "$M" --trace --workers 1 --output "$OUT/nd_v3_free.json"
step "$OUT/nd_v3_rt.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" --engine v3 \
    --manifest "$M" --trace --workers 1 --output "$OUT/nd_v3_rt.json"
python scripts/compare_rt30_traces.py "$OUT/nd_tb1.json" "$OUT/nd_v3_free.json" "$OUT/nd_v3_rt.json" \
    | tee "$OUT/nd_compare.md"
finish
