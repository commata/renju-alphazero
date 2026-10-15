#!/usr/bin/env bash
# A3.1: the 34 A3 divergence moves in their game state, engine x workers (docs/web-ser5.md §9.5).
# workers 4 first (about 20 min each), then workers 1 (about 70 min each). About 4.5 h in all.
source "$(dirname "$0")/common.sh"
for w in 4 1; do
    for engine in tb1 v1 v2; do
        step "$OUT/a31_${engine}_w$w.json" python scripts/run_rt30_divergence.py --checkpoint "$CK" \
            --engine "$engine" --workers "$w" --games-jsonl "$OUT/a31_${engine}_w$w.jsonl" \
            --output "$OUT/a31_${engine}_w$w.json"
    done
done
finish
