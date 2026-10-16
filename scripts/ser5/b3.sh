#!/usr/bin/env bash
# B3: web load test of TB-RT30-v3 (docs/web-ser5.md §9.11). For --ai-slots 1 and 2, a test server on
# port 8100 (its own database under runs/b3/) and simulated browsers (scripts/web_load_test.py,
# v5 "humans") for 15 minutes each: slots 1 x sessions 1/2/4, slots 2 x sessions 2/4. About 1.5 h.
# SLOTS=2 runs only the 2-slot rows (B3.1: B3=runs/b3_1 SLOTS=2). Stop the production server's games first (it would share the CPU); its port 8000 is not touched.
source "$(dirname "$0")/common.sh"
ensure_native
B3=${B3:-runs/b3}
MINUTES=${MINUTES:-15}
PORT=${PORT:-8100}
mkdir -p "$B3"
for slots in ${SLOTS:-1 2}; do
    if [ "$slots" = 1 ]; then counts="1 2 4"; else counts="2 4"; fi
    todo=""
    for n in $counts; do [ -s "$B3/slots${slots}_n${n}.json" ] || todo="$todo $n"; done
    [ -n "$todo" ] || { echo "[$(stamp)] skip slots $slots (done)"; continue; }
    echo "[$(stamp)] server: --ai-slots $slots on port $PORT"
    python scripts/run_web_play.py --host 127.0.0.1 --port "$PORT" --rt30 --policy-checkpoint "$CK" \
        --db "$B3/web_slots${slots}.db" --ai-slots "$slots" --max-sessions 8 --max-queue 8 \
        > "$B3/server_slots${slots}.log" 2>&1 &
    server=$!
    for _ in $(seq 180); do curl -sf "http://127.0.0.1:$PORT/api/health" > /dev/null && break; sleep 1; done
    curl -sf "http://127.0.0.1:$PORT/api/health" > /dev/null || { echo "[$(stamp)] FAILED: server did not start (see $B3/server_slots${slots}.log)"; kill "$server"; exit 1; }
    for n in $todo; do
        step "$B3/slots${slots}_n${n}.json" python scripts/web_load_test.py --url "http://127.0.0.1:$PORT" \
            --sessions "$n" --minutes "$MINUTES" --db "$B3/web_slots${slots}.db" --label "slots${slots}_n${n}" \
            --output "$B3/slots${slots}_n${n}.json"
    done
    kill "$server"; wait "$server" 2>/dev/null
done
python scripts/web_load_report.py "$B3"/slots*_n*.json | tee "$B3/b3_report.md"
finish
