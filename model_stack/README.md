# model_stack

Vendored detector/model code used by `rover_ui/backend/detection/detector.py`.
This is the minimal subset of the original `jetson_deploy` project needed at
runtime — imported by adding this directory to `sys.path` (see
`config.JETSON_DEPLOY_DIR`, which defaults to this folder).

Contents:

- `models/shared_gated_detector.py` — the gated-fusion multi-sensor detector
  (`SharedGatedDetector`), which produces per-sensor contribution weights.
- `front_end/adaptive_rl/{dqn_agent,environment}.py` — DQN resolution/FPS
  controller and its action/state definitions.
- `tracker.py` — IOU tracker that carries boxes (and contributions) between
  detection frames.
- `radar_tracker.py`, `radar_presence.py` — mmWave radar single-target tracking
  and per-frame presence clustering.
- `thermal_preprocess.py` — FLIR One Pro thermal domain-gap preprocessing.

## Weights (not in the repo)

The trained checkpoints are large binaries and are **not** committed (see the
top-level `.gitignore`). Provide them in one of two ways:

1. Drop the following files into `model_stack/weights/`:
   - `yolov8_rgb.pt`
   - `yolov8_thermal.pt`
   - `shared_gated.pt`
   - `dqn_agent.pt`

2. Or point the `JETSON_DEPLOY_DIR` environment variable at a directory that
   contains both this model code and a `weights/` subdirectory.

With `DETECT_ENABLE=0` the detector stack (and these weights) is not required.
