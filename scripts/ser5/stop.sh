#!/usr/bin/env bash
# Stop a running job, the jobs chained after it and every process they started (one process group).
#   scripts/ser5/stop.sh a31
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/rt30}
job=${1:?usage: scripts/ser5/stop.sh JOB}
pgid=$(cat "$OUT/bg/$job.pid" 2>/dev/null) || { echo "$job is not running"; exit 1; }
kill -- "-$pgid" && echo "stopped $job (process group $pgid); start it again to continue where it stopped"
rm -f "$OUT/bg/$job.pid"
