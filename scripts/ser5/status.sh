#!/usr/bin/env bash
# Running / finished jobs and the last lines of their logs.
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/rt30}
for out in "$OUT"/bg/*.out; do
    [ -e "$out" ] || { echo "no jobs yet"; exit 0; }
    job=$(basename "$out" .out)
    if [ -f "$OUT/bg/$job.pid" ] && kill -0 "$(cat "$OUT/bg/$job.pid")" 2>/dev/null; then state="running (pid $(cat "$OUT/bg/$job.pid"))"
    elif [ -f "$OUT/bg/$job.done" ]; then state=finished
    else state="stopped or failed"; fi
    echo "== $job: $state"
    tail -n 3 "$out"
done
