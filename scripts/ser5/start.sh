#!/usr/bin/env bash
# Start one job, or several one after another, detached from the terminal: they keep running after
# the SSH session ends (new session, no terminal, output to files).
#   scripts/ser5/start.sh a31              # scripts/ser5/a31.sh
#   scripts/ser5/start.sh a31 v2a_a1       # a31, then v2a_a1 if a31 finished
# Each job writes $OUT/bg/JOB.out (log), JOB.pid while running, JOB.done when finished.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { echo "usage: scripts/ser5/start.sh JOB [JOB...]   (v3_check, v3_dev, v3_a1, v3_a3, v3_nd; a31, v2a_a1, v2a_a3)"; exit 1; }
OUT=${OUT:-runs/rt30}
export OUT
mkdir -p "$OUT/bg"
for job in "$@"; do
    [ -f "scripts/ser5/$job.sh" ] || { echo "no scripts/ser5/$job.sh"; exit 1; }
    if [ -f "$OUT/bg/$job.pid" ] && kill -0 "$(cat "$OUT/bg/$job.pid")" 2>/dev/null; then
        echo "$job is already running"; exit 1
    fi
    rm -f "$OUT/bg/$job.done"
done
setsid nohup bash -c 'for job in "$@"; do bash "scripts/ser5/$job.sh" >> "$OUT/bg/$job.out" 2>&1 || exit 1; done' \
    chain "$@" > /dev/null 2>&1 < /dev/null &
sleep 1
echo "started: $* (log $OUT/bg/<job>.out)"
echo "watch:   tail -f $OUT/bg/$1.out"
echo "status:  scripts/ser5/status.sh        stop: scripts/ser5/stop.sh $1"
