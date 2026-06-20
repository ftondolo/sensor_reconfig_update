# purely_control — T265-only rover motion service

Drive the rover an **exact signed distance along a T265 axis and stop**, using
nothing but the Intel RealSense T265 for feedback. No UI, no dependency on the
`rover_ui` stack — it opens the T265 and the ROS `cmd_vel` publisher directly and
exposes a small service you call from your own code.

```python
from purely_control import T265RoverService

with T265RoverService() as rover:
    rover.move_axis("x", 500, units="mm")    # +x  -> right   500 mm
    rover.move_axis("z", -200, units="mm")   # -z  -> forward 200 mm
```

## Axes (T265 native frame)

Right-handed device frame, exactly as the T265 reports it:

```
        +y (up)
         |
         |____ +x (right)
        /
      +z (back)            forward = -z
```

| You ask for      | Rover does            |
|------------------|-----------------------|
| `move_axis("x",  0.5)`  | strafe **right** 0.5 m |
| `move_axis("x", -0.5)`  | strafe **left** 0.5 m  |
| `move_axis("z", -0.2)`  | drive **forward** 0.2 m |
| `move_axis("z",  0.2)`  | drive **backward** 0.2 m |
| `move_axis("y",  …)`    | **rejected** — that's vertical |

Friendly aliases also work: `"right"`, `"left"`, `"forward"`, `"back"`.
Units: `"m"` (default), `"cm"`, `"mm"`.

The base is assumed **mecanum/holonomic** — lateral moves go straight to
`Twist.linear.y`, no rotate-drive-rotate. A heading-hold loop keeps the rover at
its starting yaw so a strafe tracks a straight line.

## API

| Call | What it does |
|------|--------------|
| `T265RoverService(allow_mock=True, **cfg)` | construct; pass `Config` field overrides as kwargs |
| `.start(wait_for_pose=True, timeout=5.0)` | open camera + ROS, spin up threads (context-manager `__enter__` does this) |
| `.move_axis(axis, distance, units="m", blocking=True, timeout=None)` | the headline call — one T265-native axis |
| `.move(right=, forward=, units="m", blocking=True)` | rover-frame convenience; diagonals allowed |
| `.is_busy()` / `.wait(goal=None)` | poll / block on a non-blocking move |
| `.stop()` | cancel the active move, command zero now |
| `.get_pose()` | latest `{right, forward, yaw, confidence, mock}` |
| `.shutdown()` | stop the rover, join threads, close handles (`__exit__`) |

Each move is **relative to wherever the rover is now** (the current pose is
snapshotted as the origin on entry). `move_axis` blocks by default and returns a
`MoveResult` — truthy on success, with `.reason` in
`{"arrived","timeout","cancelled","tracking_lost"}`, plus `.target`, `.reached`,
`.error` (m), and `.elapsed` (s).

Moves are **one at a time**: calling `move()` while one is running raises. Call
`stop()` first, or `wait()` for it.

## How it controls

1. Snapshot the start pose; each tick re-express the live pose in that
   start-heading body frame → `(right, forward)` travelled since the move began.
2. Drive one speed along the remaining-error vector: proportional to distance (so
   it eases in), floored at `MIN_LINEAR` (so the last cm runs at a real speed,
   not an asymptotic crawl), capped at `MAX_LINEAR`.
3. Hold the start yaw every tick (`linear.x`/`linear.y` translate, `angular.z`
   corrects yaw) — see below.
4. Stop once inside `POS_TOL` **and** `YAW_TOL` for `SETTLE_TIME`; `timeout` is
   the safety backstop.

### No rotation during a move (heading-hold)

The rover keeps the heading it had when the move began — it translates without
turning, and any rotation that creeps in is driven back out:

- A **PI** controller on `angular.z` corrects the yaw error against the start
  heading on every control tick. The P term reacts to disturbances (e.g. a wheel
  slip nudges the rover); the I term nulls a *constant* bias (e.g. a mecanum base
  that drifts slightly while strafing) so heading returns **fully** to the start,
  not just close to it.
- A move only completes when heading is back within `YAW_TOL` (~1.1°) as well as
  position within `POS_TOL` — **a move can never finish with the rover rotated.**
  If it can't get heading back (e.g. a large continuous external torque) it times
  out rather than reporting success.
- `MoveResult.yaw_error` reports the residual heading error (rad; ~0 on success),
  so callers can verify orientation was preserved.

Verified in mock against a mid-move 15° kick (corrected back to <1°) and a steady
8°/s yaw drift (held under ~3° and restored to <1° at the end). Tune with
`YAW_GAIN`/`YAW_IGAIN`/`YAW_TOL` in `Config`.

Arrival means *within `POS_TOL`* (default 20 mm), so moves finish a hair short
(~18 mm on a 500 mm move) — consistent and deterministic. Tighten `POS_TOL` for a
closer stop (at the cost of a slower final approach). All knobs live in `Config`
in `t265_rover.py`; override per-instance, e.g.
`T265RoverService(MAX_LINEAR=0.1, POS_TOL=0.01)`.

## CLI (manual testing)

```bash
# right 500 mm, then forward 200 mm:
python -m purely_control --move x:500mm --move z:-200mm

# dry-run the control logic with no hardware/ROS (mock integrates the commands):
python -m purely_control --mock --move forward:0.3
```

## Running on the real rover

- **ROS:** `source /opt/ros/noetic/setup.bash` first so `rospy` is importable and
  a master is reachable. Publishes `geometry_msgs/Twist` to `/cmd_vel` (the
  verified WHEELTEC topic). Override with `T265RoverService(CMD_VEL_TOPIC=...)`.
- **T265 driver:** needs `pyrealsense2 <= 2.53.x` — T265 pose support was removed
  in 2.54. **Verified installed here: 2.49.0**, which supports T265 pose
  (`rs.stream.pose` present). Keep it ≤ 2.53; don't upgrade to 2.54+.
  (No T265 was connected when this was checked — enumerate the device with
  `rs.context().query_devices()` and do one live pose read before driving.)
- **Mock fallback:** if the SDK/camera or ROS is missing it prints a notice and
  runs in MOCK mode (integrates the commanded velocity into a fake pose) so the
  control logic is testable off-hardware. `allow_mock=False` makes a missing
  camera raise instead. **Mock never moves a real rover.**
- **Sign flips:** if your base strafes the wrong way set `Y_SIGN=-1`; if
  forward/back is reversed set `X_SIGN=-1`; if a heading correction turns the
  **wrong way** (the rover spins up instead of straightening) set `YAW_SIGN=-1`.

### Frame alignment — verify before trusting it (NOT hardware-checked)

The T265↔Twist mapping (`+x=right`, `forward=-z`, `linear.y=-right`,
`angular.z=yaw`) is derived from the documented T265 convention and matches the
sibling `rover_ui` project, but it assumes the camera is **mounted level, lens
facing forward, +x to the rover's right**, and it has **not been tested on real
hardware**. Bring it up carefully, at low speed, with an e-stop ready:

1. **Pose sanity (no motion).** With ROS off (or the base unpowered), read
   `get_pose()` while you push the rover by hand:
   - roll it **forward** → `forward` should increase;
   - roll it **right** → `right` should increase;
   - rotate it **left/CCW** (viewed from above) → note which way `yaw` moves.
   If forward/right are swapped or inverted, the camera isn't mounted as assumed
   — fix the mount or set `X_SIGN`/`Y_SIGN`.
2. **Translation, low speed** (`MAX_LINEAR=0.08`): `move_axis("z", -100, units="mm")`
   should go **forward** ~10 cm; `move_axis("x", 100, units="mm")` should go
   **right**. Flip `X_SIGN`/`Y_SIGN` if reversed.
3. **Yaw sign — the dangerous one.** Start a slow forward move and, by hand,
   gently rotate the rover a few degrees off heading. The controller must turn it
   **back** toward the start heading. If instead it turns *further away* (or
   `move_axis` ends with a growing `yaw_error` / starts spinning), the yaw
   feedback is inverted: set `YAW_SIGN=-1`. **Test this with a hand on the
   rover** — an inverted yaw loop is positive feedback and will spin up.
- **Tracking confidence:** if the T265 reports lost tracking
  (`confidence == 0`) mid-move the rover holds still until tracking recovers or
  the move times out (`reason="tracking_lost"`).

> Safety: validate `X_SIGN`/`Y_SIGN` and start with a small move (e.g.
> `move_axis("z", -100, units="mm")`) and a low `MAX_LINEAR` before trusting
> larger commands.
