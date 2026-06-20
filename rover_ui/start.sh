#!/usr/bin/env bash
# Start the rover demo: dashboard + in-process multimodal detector +
# BEV-map navigation (rectilinear planner -> purely_control/T265) on hardware.
#
#   ./start.sh
#
# Runs detached (setsid) so it survives this shell / an SSH drop. Logs to
# $ROVER_UI_LOG. Open  http://<jetson-ip>:8000/audience  in a browser.
#
# Override any setting by exporting it first, e.g.
#   FLIR_DEVICE=/dev/video2 ALLOW_MOCK=1 ./start.sh
cd "$(dirname "$0")"

PORT="${ROVER_UI_PORT:-8000}"
LOG="${ROVER_UI_LOG:-/tmp/rover_ui_v1.log}"

# --- refuse to double-start: a stale instance holds the cameras + serial ports
if pgrep -f "uvicorn backend.app" >/dev/null 2>&1; then
    echo "Already running (uvicorn backend.app). Run ./stop.sh first."
    echo "(Only one instance may own the cameras/T265/serial ports at a time.)"
    exit 1
fi
if command -v lsof >/dev/null 2>&1 && lsof /dev/ttyUSB0 /dev/ttyUSB1 >/dev/null 2>&1; then
    echo "WARNING: /dev/ttyUSB0|1 are held by another process — mmWave will fall"
    echo "         back to mock. Run ./stop.sh (or kill the holder) first."
fi

# --- thermal relay (flirone) runs SEPARATELY and feeds /dev/video1.
if ! pgrep -x flirone >/dev/null 2>&1; then
    echo "WARNING: FLIR relay (flirone) is NOT running — thermal will be black."
    echo "         Start it with:  ./start_thermal.sh"
fi

# --- ROS 1 for /cmd_vel (purely_control). Best-effort; mock works without it.
if [ -z "${ROS_MASTER_URI:-}" ] && [ -f /opt/ros/noetic/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash 2>/dev/null || true
fi

# --- verified hardware config (override via env before calling)
export ALLOW_MOCK="${ALLOW_MOCK:-0}"          # 0 = real hardware only
export FLIR_DEVICE="${FLIR_DEVICE:-/dev/video1}"
export FLIR_WIDTH="${FLIR_WIDTH:-160}"
export FLIR_HEIGHT="${FLIR_HEIGHT:-120}"

echo "Starting rover demo (ALLOW_MOCK=$ALLOW_MOCK FLIR_DEVICE=$FLIR_DEVICE, port $PORT) ..."
echo "  map: $(cd .. && pwd)/map.json (6 m x 6 m, obstacles pre-marked)"
echo "  REMINDER: place the rover on its start cell (map.json rover_start),"
echo "  facing the top of the map, before relying on map positions."
rm -f "$LOG"
setsid bash run.sh >"$LOG" 2>&1 &
disown 2>/dev/null || true

# --- wait for the server to bind the port
for _ in $(seq 1 40); do
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
        echo "Up. (detector loads TRT engines + sensors stream in ~15-20s)"
        echo "  Audience view : http://${ip:-<jetson-ip>}:$PORT/audience"
        echo "  Operator      : http://${ip:-<jetson-ip>}:$PORT/"
        echo "  Status        : curl -s http://127.0.0.1:$PORT/api/status"
        echo "  Log           : $LOG"
        exit 0
    fi
    sleep 0.5
done
echo "ERROR: did not bind :$PORT within 20s — check $LOG"
exit 1
