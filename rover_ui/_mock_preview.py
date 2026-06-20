"""MOCK preview launcher — run the full UI with NO hardware.

For viewing the operator + audience frontends when the sensors / T265 are
unplugged. Forces the T265 motion service and the FLIR + detector sensors
straight to their mock paths so startup never blocks probing absent USB
devices (a real start would hang opening a missing T265 and stall ~15 s/retry
on a producerless FLIR node).

    cd rover_ui && ALLOW_MOCK=1 python3 _mock_preview.py
    (or ./run_mock.sh to run it detached; ./stop.sh to stop)
"""
import os
import sys

os.environ.setdefault("ALLOW_MOCK", "1")

# Repo root (parent of rover_ui) holds the purely_control package.
_DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DEMO_ROOT not in sys.path:
    sys.path.insert(0, _DEMO_ROOT)

# T265 motion service -> mock (no USB probe, no hardware_reset waits).
import purely_control.t265_rover as _tr
_tr.rs = None

# FLIR + detector -> mock paths (no 15 s cv2 V4L2 stall, no TRT engine load).
from backend.sensors import flir as _flir
_flir.FlirThermal.open = lambda self: (_ for _ in ()).throw(RuntimeError("mock preview: no FLIR"))
from backend.detection import detector as _det
_det.DetectorThread.open = lambda self: (_ for _ in ()).throw(RuntimeError("mock preview: detector mock"))

if __name__ == "__main__":
    import uvicorn
    from backend import config
    uvicorn.run("backend.app:app", host=config.HOST, port=int(config.PORT),
                workers=1, log_level="info")
