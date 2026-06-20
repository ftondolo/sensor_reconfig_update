# rover_ui — BEV-map navigation demo

A web dashboard + backend for the sensor rover on the Jetson Orin Nano.
Movement is *map-based*:

```
detection (bearing + radar range)
        │ project onto the pre-stored map
        ▼
6 m × 6 m BEV map (map.json, mm; obstacles pre-marked)
        │ goal = target (x, z) → (x, z + 2000)   [2 m on the rover side]
        │ if the rover footprint at the goal overlaps an obstacle → nudge to
        │ the nearest free spot (preferring positions farther from the target)
        ▼
rectilinear_mm.plan_rectilinear_path()   — obstacle-avoiding axis-aligned plan
        │ one (axis, mm) leg at a time
        ▼
purely_control.T265RoverService.move_axis(axis, mm)   — closed-loop on the T265
```

## Map / frames

- `../map.json` — the pre-stored map: `size` 6000×6000 mm, `obstacles` as
  `{x, z, w, h}` rectangles (mm), `rover_start` = the map cell the rover
  physically starts on. **Edit this file to mark your real obstacle layout.**
- `../config.json` — rover footprint (800×1200 mm), clearance (300 mm), turn
  penalty for the planner.
- Frame: `x` right, `z` down (toward the rover side). The rover noses "up" the
  map; forward motion decreases `z`. A target at `(x, z)` puts the standoff
  goal at `(x, z + 2000)` — 2 m below it on the map.
- The T265 world pose is anchored to `rover_start` at startup (and via the
  operator's *Re-anchor* button): place the rover on that cell facing the top
  of the map, then start the demo / re-anchor.

## mmWave policy

- **Vision first**: the detection target is the most-confident RGB/thermal box
  (≥ `DETECT_CONF_NORADAR`). The radar never selects/rejects vision boxes; it
  only supplies the *range* for the map projection.
- **mmWave = fallback only**: a radar-driven target exists only when RGB and
  thermal both see nothing.
- **Moving returns only**: the chirp cfg (`../iwr6843_moving_only.cfg`) has
  `clutterRemoval -1 1` — static returns are removed at the source — plus a
  software filter (`RADAR_MIN_POINT_V`).
- **Rover must be stationary**: radar frames captured while the rover itself
  is translating are dropped and the tracker is reset (ego-motion fakes
  Doppler on static clutter). A radar target requires the rover to be still
  AND a moving cluster to be acquired; the navigator additionally refuses to
  use a radar-sourced target while a move is in progress.

## Hardware ownership (important)

`purely_control.T265RoverService` is the **only** owner of the T265 and of the
`/cmd_vel` publisher in this process. Do not start a second `T265Pose` sensor
thread or another instance alongside it — they will fight over the cameras and
serial ports.

## Run

```bash
cd rover_ui
./start.sh           # detached; logs to /tmp/rover_ui_v1.log
./stop.sh
# or in the foreground:
source /opt/ros/noetic/setup.bash
./run.sh
```

- Audience view: `http://<jetson-ip>:8000/audience` — big BEV map (rover
  footprint, detected person, standoff goal, planned path) + sensor feeds.
- Operator: `http://<jetson-ip>:8000/` — same map, clickable (click = drive to
  a 2 m standoff below that point), plus *Navigate to detection*, *Cancel*,
  *Auto* toggle, pose re-anchor and 200 mm nudge buttons.

Without hardware/ROS everything falls back to **mock** (badges show MOCK,
purely_control integrates commanded velocity into a fake pose) so the whole
flow — click the map, watch the plan + the rover glide along it — is testable
on a laptop. `ALLOW_MOCK=0` requires real hardware.

## HTTP API

| Endpoint | Meaning |
|---|---|
| `GET /api/map` | static map payload (size, obstacles, car, rover_start) |
| `GET /api/status` | sensors + nav snapshot |
| `POST /api/nav/go` | navigate to the current detection target |
| `POST /api/nav/goto {x, z}` | navigate to a manual map target (mm) |
| `POST /api/nav/cancel` | cancel + stop the rover |
| `POST /api/nav/auto {enabled}` | auto-navigate on stable detections |
| `POST /api/nav/reset_pose {x?, z?}` | anchor the current pose to a map cell |
| `POST /api/nav/reload_map` | hot-reload an edited map.json / config.json |
| `POST /api/nav/nudge {axis, mm}` | small open-map setup move (≤1000 mm) |
| `WS /ws/telemetry` | `{mmwave, detection, nav, status}` ~15 Hz |

All tunables live in `backend/config.py` (`NAV_*` for navigation), each
overridable by an env var of the same name.
