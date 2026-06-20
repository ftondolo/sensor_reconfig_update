#!/usr/bin/env bash
# Launch the rover demo in MOCK preview mode (no hardware), detached. For viewing the
# operator + audience frontends when the sensors/T265 are unplugged.
#
#   ./run_mock.sh     # detached; logs to $ROVER_UI_LOG
#   ./stop.sh         # stop it
cd "$(dirname "$0")"

PORT="${ROVER_UI_PORT:-8000}"
LOG="${ROVER_UI_LOG:-/tmp/rover_ui_v1.log}"

if pgrep -f "uvicorn backend.app" >/dev/null 2>&1 || pgrep -f "_mock_preview" >/dev/null 2>&1; then
    echo "Already running. Run ./stop.sh first."
    exit 1
fi

export ALLOW_MOCK=1
echo "Starting MOCK preview (no hardware) on port $PORT ..."
rm -f "$LOG"
setsid bash -c "exec ${PYTHON:-python3} _mock_preview.py" >"$LOG" 2>&1 &
disown 2>/dev/null || true

for _ in $(seq 1 40); do
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
        echo "Up (MOCK)."
        echo "  Audience : http://${ip:-<this-ip>}:$PORT/audience"
        echo "  Operator : http://${ip:-<this-ip>}:$PORT/"
        echo "  Log      : $LOG"
        exit 0
    fi
    sleep 0.5
done
echo "ERROR: did not bind :$PORT within 20s — check $LOG"
exit 1
