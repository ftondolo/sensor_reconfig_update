#!/usr/bin/env bash
# Stop the rover demo and release the cameras + T265 + serial ports.
#
#   ./stop.sh
#
# Sends SIGTERM first (clean shutdown: the nav cancels its move and the rover
# is commanded to zero velocity), then SIGKILL if it lingers.
PORT="${ROVER_UI_PORT:-8000}"

echo "Stopping rover demo ..."
stopped=0
if pgrep -f "uvicorn backend.app" >/dev/null 2>&1; then
    pkill -f "uvicorn backend.app" 2>/dev/null || true
    echo "  sent SIGTERM to uvicorn backend.app"
    stopped=1
fi
if pgrep -f "_mock_preview" >/dev/null 2>&1; then
    pkill -f "_mock_preview" 2>/dev/null || true
    echo "  sent SIGTERM to mock preview"
    stopped=1
fi
[ "$stopped" = 0 ] && echo "  (nothing was running)"

# Give uvicorn time to run its shutdown handler and release the RealSense +
# serial ports CLEANLY (~10s). A clean exit means no replug is needed.
for _ in $(seq 1 20); do
    pgrep -f "uvicorn backend.app\|_mock_preview" >/dev/null 2>&1 || break
    sleep 0.5
done
if pgrep -f "uvicorn backend.app\|_mock_preview" >/dev/null 2>&1; then
    echo "  still alive — forcing (SIGKILL)"
    pkill -9 -f "uvicorn backend.app" 2>/dev/null || true
    pkill -9 -f "_mock_preview" 2>/dev/null || true
    sleep 1
fi

# --- report final state
if pgrep -f "uvicorn backend.app\|_mock_preview" >/dev/null 2>&1; then
    echo "  WARNING: uvicorn still running"
else
    echo "  uvicorn stopped"
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "  WARNING: port $PORT still bound"
else
    echo "  port $PORT free"
fi
if command -v lsof >/dev/null 2>&1; then
    if lsof /dev/ttyUSB0 /dev/ttyUSB1 >/dev/null 2>&1; then
        echo "  WARNING: /dev/ttyUSB0|1 still held"
    else
        echo "  serial ports /dev/ttyUSB0|1 free"
    fi
fi
# The FLIR relay (flirone) is a separate service feeding /dev/video1 — leave it
# running so thermal is ready immediately on the next ./start.sh.
if pgrep -x flirone >/dev/null 2>&1; then
    echo "  FLIR relay (flirone) left running -> /dev/video1 (not stopped)"
fi
echo "Done. Restart with ./start.sh (no replug needed)."
