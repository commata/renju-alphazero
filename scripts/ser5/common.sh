# Sourced by the scripts/ser5/*.sh job scripts (docs/web-ser5.md §9.5).
# Each job is a list of `step OUTPUT.json command...` lines. A step whose output exists is skipped,
# so a job stopped by a reboot or `stop.sh` continues where it stopped when it is started again
# (the match and divergence runners also resume inside a step from their --games-jsonl).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [ -f .venv-ser5/bin/activate ]; then source .venv-ser5/bin/activate; fi
export PYTHONUNBUFFERED=1
command -v python > /dev/null || python() { python3 "$@"; }  # no venv: system python3
CK=${CK:-runs/h3_policy_64x4/best.pt}
OUT=${OUT:-runs/rt30}
JOB=$(basename "$0" .sh)
mkdir -p "$OUT/bg"
# The process group (start.sh makes a new session per chain): stop.sh kills the whole chain.
ps -o pgid= -p $$ | tr -d " " > "$OUT/bg/$JOB.pid"
trap 'rm -f "$OUT/bg/$JOB.pid"' EXIT

stamp() { date '+%F %T'; }

step() {
    local out=$1; shift
    if [ -s "$out" ]; then echo "[$(stamp)] skip $out (exists)"; return 0; fi
    local log="${out%.json}.log"
    echo "[$(stamp)] start $out (log $log)"
    if "$@" 2>&1 | tee "$log"; then
        echo "[$(stamp)] done $out"
    else
        echo "[$(stamp)] FAILED $out (see $log)"; exit 1
    fi
}

finish() { echo "[$(stamp)] $JOB finished"; touch "$OUT/bg/$JOB.done"; }
