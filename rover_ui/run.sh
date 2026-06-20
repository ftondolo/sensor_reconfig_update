#!/usr/bin/env bash
# Launch the rover UI backend (map navigation).
#
# Run from the rover_ui directory. Source your ROS 1 workspace first so
# purely_control can publish cmd_vel:
#   source /opt/ros/noetic/setup.bash
#   ./run.sh
#
# Set ALLOW_MOCK=0 to disable synthetic fallback (real hardware only).
set -e

cd "$(dirname "$0")"

HOST="${ROVER_UI_HOST:-0.0.0.0}"
PORT="${ROVER_UI_PORT:-8000}"

echo "Starting Rover Demo UI (map navigation) on http://${HOST}:${PORT}"
echo "  (open http://<jetson-ip>:${PORT}/audience from a browser on the same network)"

# Single worker on purpose: the sensor threads and the T265RoverService are
# process-global singletons. Multiple workers would open the devices twice.
PY="${PYTHON:-python3}"
exec "$PY" -m uvicorn backend.app:app --host "$HOST" --port "$PORT" --workers 1
