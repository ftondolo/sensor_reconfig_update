#!/usr/bin/env bash
# Start the FLIR One Pro relay (flirone -> v4l2loopback /dev/video1) in a
# detached screen named 'flir'. The rover demo's thermal panel reads /dev/video1,
# so this must be running for a live thermal feed.
#
#   ./start_thermal.sh
#
# Needs sudo (flirone touches USB + the loopback). If sudo prompts for a
# password the screen will pause — attach with `screen -r flir` to type it.
FLIR_DIR="${FLIR_DIR:-/home/icsl/workspace/flirone-v4l2}"
PALETTE="${FLIR_PALETTE:-palettes/Grayscale.raw}"

if pgrep -x flirone >/dev/null 2>&1; then
    echo "flirone already running -> /dev/video1 (nothing to do)."
    exit 0
fi
if [ ! -x "$FLIR_DIR/flirone" ]; then
    echo "ERROR: $FLIR_DIR/flirone not found. Set FLIR_DIR to the flirone-v4l2 dir."
    exit 1
fi
if ! command -v screen >/dev/null 2>&1; then
    echo "screen not installed. Run manually:"
    echo "  cd $FLIR_DIR && sudo ./flirone $PALETTE"
    exit 1
fi

echo "Starting flirone in screen 'flir' (may prompt for sudo password) ..."
screen -dmS flir bash -c "cd '$FLIR_DIR' && sudo ./flirone '$PALETTE'"
for _ in $(seq 1 10); do
    pgrep -x flirone >/dev/null 2>&1 && break
    sleep 0.5
done
if pgrep -x flirone >/dev/null 2>&1; then
    echo "flirone up. Thermal -> /dev/video1."
    echo "  view it:   screen -r flir     (detach again with Ctrl-A then D)"
else
    echo "flirone not up yet — it's likely waiting for the sudo password."
    echo "  attach and finish:   screen -r flir"
fi
