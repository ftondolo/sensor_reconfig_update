# Rover — map-based autonomous navigation with fiducial localization

An autonomous ground rover that tracks a person with a multimodal sensor suite
and drives to a standoff position beside them, inside a mapped arena, avoiding
obstacles.

The **detection pipeline** (RGB + thermal + mmWave fusion, gated ResNet, DQN
resolution control) is prior work and is documented in
[`model_stack/README.md`](model_stack/README.md). **This README covers the rover:**
localization, path planning, motion control, safety, and the fiducial
correction system built on top.

For day-to-day operation, the UI, every readout, and troubleshooting, see
**[`docs/OPERATOR_MANUAL.md`](docs/OPERATOR_MANUAL.md)**.

---

## 1. Hardware

| Part | Role |
|---|---|
| NVIDIA Jetson Orin Nano | all compute |
| WHEELTEC mecanum base | omnidirectional drive, ROS 1 `/cmd_vel` |
| Intel RealSense **T265** | visual-inertial odometry (pose) |
| Intel RealSense **D435i** | RGB (detection + ArUco) and aligned depth (range) |
| FLIR One Pro | thermal |
| TI **IWR6843ISK** mmWave | Doppler-gated radar presence and range |

The rover is **holonomic and never rotates deliberately.** It strafes and
reverses; the sensor heading is fixed in the world frame. Any yaw that appears
is drift, and the controller's job is to null it. This assumption is load-bearing
throughout — it is what makes an absolute heading hold viable and what lets
several safety checks reject physically impossible states.

---

## 2. Coordinate frames

Everything is **millimetres**, in a single 2-D map frame.

```
        x →  (right)
   ┌──────────────────────┐  z = 0      far end
   │                      │
   │      ARENA           │             rover drives toward −z
   │      5000 × 6000     │
   │                      │
   └──────────────────────┘  z = 6000   rover_start / home
```

* **x** increases to the right.
* **z** increases *backward*, toward the start. Driving forward **decreases z**.
* **yaw 0** means the rover faces −z. The camera therefore sees the **+z faces**
  of the panels, which is where all ArUco tags are mounted.

The T265 reports its own local frame; the map frame is defined by an *anchor*
pair `(_anchor_map, _anchor_pose)` captured at `reset_pose()`. That pair is
effectively a `map → odom` transform, and it is the hook every absolute
correction uses (§7).

---

## 3. Pose: T265 with lever-arm compensation

`T265RoverService._make_pose()` does not report the sensor's position — it
reports the **rover's turn centre**:

```python
ox, oz = self.cfg.SENSOR_OFFSET_RIGHT, self.cfg.SENSOR_OFFSET_FWD
if ox or oz:
    c, sn = math.cos(yaw), math.sin(yaw)
    rx -= ox * c + oz * sn
    rf -= -ox * sn + oz * c
```

Without this, a sensor mounted `d` off the turn centre traces an arc of
`2·d·sin(Δθ/2)` when the rover rotates in place, and reports it as real
translation. At the measured 60 mm offset a 30° correction moved the believed
position 31 mm — larger than the 25 mm arrival tolerance, so heading corrections
alone could fail a leg. The same correction is applied to the **camera** when
projecting a detection (`Navigator._camera_map_xz`), or a stationary target
appears to swing every time the rover turns.

Pose frames are filtered (`_filter_pose`) against an impossible-speed gate. One
subtlety: a large step **while commanded stationary** is *accepted*, not
rejected — a real teleport is physically impossible when stopped, so the step is
the T265 relocalising against its own map, and discarding it would re-apply
drift the device had just removed.

---

## 4. Obstacle representation

Panels are thin walls standing in stabiliser feet (an "H" in plan). Those are
two different geometries and the code keeps them apart:

```jsonc
// config.json
"obstacle_size": { "w": 1350, "h": 40 },   // the WALL — what the camera sees
"obstacle_pad":  { "x": 0,    "z": 155 }   // the FEET — what the rover hits
```

```jsonc
// map.json — obstacles carry no size; (x, z) is the −x edge on the z MIDLINE
{ "x": 182, "z": 4398, "tags": [ { "id": 0, "dx": -550, "dy": 1500, "dz": 25 } ] }
```

`rectilinear_mm.obstacle_boxes(map, cfg, kind)` resolves both:

* `kind="collision"` → wall **+ pad**: used by the planner, inflation and e-stop.
* `kind="panel"` → the bare wall: used for tag geometry, because a tag is stuck
  to the panel surface, not to the feet.

The pad exists so physical extent stays separate from safety *policy*. Folding
the feet into `clearance` would also inflate every panel **end** in x, where the
feet do not reach, needlessly narrowing every corridor — and it would make the
reported clearance figure meaningless by silently including foot depth.

A legacy min-corner schema (per-obstacle `w`/`h`) is still read. Detection is by
**content, not a version field**, because misreading one as the other shifts
every panel by half its depth, silently, in the direction that causes collisions.
The interpretation is logged at startup:

```
[nav] MIDLINE schema, 5 obstacles; #0 keep-out x 182..1532 z 4223..4573
```

---

## 5. Planning: widest rectilinear path

`rectilinear_mm.py` plans **axis-aligned** routes on a Hanan grid built from the
inflated obstacle edges, A* with a `turn_penalty` so it prefers few long legs.

The non-obvious part is clearance. Rather than planning once at the configured
`clearance`, it **binary-searches the largest clearance that still admits a
path**:

> Feasibility is *monotonic* in clearance — inflating obstacles can only shrink
> free space — so the largest feasible clearance **is** the max-min (bottleneck)
> clearance.

The base clearance is always tried first and kept as a fallback, so the search
never fails where a single-shot plan would have succeeded. Measured on this
arena: **+120 mm mean clearance for +3.2% path length.**

Two refinements matter in practice:

* **Footprint bounds, not centre bounds.** Testing only the rover's centre
  against the map rectangle let routes hug an edge with half the car outside the
  room — one measured route ran the full length of the arena 252 mm through a
  wall. `_excess()` tests the footprint, with slack for a start pose that
  already protrudes so a corner-parked rover is never stranded.
* **A tight start must not cap the whole route.** When the rover is parked close
  to a panel, raising the trial clearance would swallow its own start cell and
  abort the search. Obstacles already containing the start keep their *base*
  inflation while everything else scales up — so the panel stays solid, but
  standing near it no longer forces the rest of the route to be narrow. The
  reported clearance is then capped at the start pose's own gap so the figure
  never overstates what the rover actually gets.

---

## 6. Motion execution

`Navigator._drive()` walks the plan **leg by leg**, and each leg is re-derived
from the **live pose** rather than from the plan's coordinates:

```python
dx = wx - p["x"]
dz = wz - p["z"]
```

so per-leg arrival error does not accumulate across a path.

`T265RoverService` runs a **20 Hz** closed loop:

* **Translation** — P on distance (`LINEAR_GAIN`), clamped between `MIN_LINEAR`
  and `MAX_LINEAR`, slew-limited by `RAMP_STEP`. Below `MIN_LINEAR/LINEAR_GAIN`
  of remaining distance the P term is floored, so the last stretch is crossed at
  constant speed — which is why the approach speed and the arrival tolerance
  must be chosen together.
* **Stiction integral** — pure P cannot break static friction. `LIN_IGAIN`
  integrates **time spent stuck** (commanded to move but closing slower than
  `LIN_STICTION_EPS`), not distance error, so it contributes nothing during
  normal cruise and cannot wind up and overshoot.
* **Heading hold** — PI on an **absolute** target (`_anchor_pose["yaw"]`), so
  every leg drives heading back to true map-forward rather than merely holding
  wherever it started. Inside `YAW_DEADBAND` the command is **zeroed and the
  integral bled**; applying the integral there while zeroing only the P term
  made the base coast through centre and oscillate left-right in place.
* **Cross-axis stiffening** — a strafing mecanum base creeps on the minor axis,
  so that axis gets its own full P correction capped at `MAX_CROSS`.

`ARRIVE_SETTLE_TIMEOUT = 0` by default: legs complete on **position alone** and
the next leg's heading hold removes the residual while actually making progress.

> **Effective values differ from the class defaults.** `_Cfg` in
> `t265_rover.py` is overridden by `config.py` at construction. Live values are
> `MAX_LINEAR 0.25`, `MIN_LINEAR 0.06`, `RAMP_STEP 0.05`, `POS_TOL 0.025 m`,
> `STALL_TIMEOUT 8.0`. Read `config.py`, not the class body.

---

## 7. ArUco localization — the only thing that *bounds* drift

VIO drifts without bound because nothing in the loop ever observes absolute
position. Everything else in this system slows drift; fiducials remove it.

**20 tags** (`DICT_4X4_50`, 190 mm), 4 per panel in a 2×2 on the home-facing
side. IDs are the 20 with maximum minimum Hamming separation over all rotations
(`0–15, 17, 18, 22, 35`), which buys real error-correction margin over an
arbitrary `0..19`.

`tag_localizer.py` pools **every** recognised tag into one `solvePnP`
(`SQPNP`) in map coordinates, so one code path serves 1 tag or 8 and simply
improves as more appear. Tag offsets are measured from each panel's
**ground-centre**: `dx`/`dz` from the panel centre, `dy` as absolute height
above the floor.

Detection is tuned for this arena, not left at defaults:

* `adaptiveThreshWinSizeMin = 7`. OpenCV's default starts at 3 px, which is the
  perforation pitch of the pegboard panels — binarisation locks onto hole
  texture. On a real captured frame the default found **1 of 4** tags; 7 found
  3, stably across every step value.
* **Unsharp mask** before detection. With the window above it recovered the 4th
  tag in 12 of 12 parameter combinations, at ~6 ms and **no** loss of corner
  precision (SUBPIX re-finds the true edge afterwards).
* `CORNER_REFINE_SUBPIX`. The OpenCV default is `NONE`, which locates corners
  only to contour-vertex precision; SUBPIX roughly halves corner error and cut
  pose error 2.7 mm → 0.4 mm on a 4-tag panel at 2.5 m.

**Gates before a fix is applied** (`Navigator.apply_tag_fix`):

| Gate | Why |
|---|---|
| rover commanded **stationary** | motion blur wrecks corners; a mid-leg fix moves the goal under the planner |
| ≥ 2 tags | a single tag has 4 corners against 6 pose DOF — it fits *exactly* however wrong the map is, so it cannot self-check |
| horizontal spread ≥ 400 mm | tags in one vertical column leave sideways position undetermined; the solver returns a confident **mirrored** pose that reprojects at ~0.3 px |
| reprojection RMS ≤ limit (scales with tag count) | RMS is a direct proxy for how wrong the fix would be |
| heading residual ≤ limit | the rover never turns deliberately, so a large tag-derived yaw is evidence the *map* is wrong |
| >300 mm corrections need 3 agreeing fixes | challenge-counter, as used by the radar ghost guard |

Accepted fixes move the **anchor**, never the pose:

```python
self._anchor_map = (self._anchor_map[0] + sx, self._anchor_map[1] + sz)
```

eased at `α = 0.35` and capped at 120 mm per application, so the 20 Hz control
loop never sees a discontinuity. Deliberately **not** `reset_pose()`, which
re-snapshots `_anchor_pose` and would zero the map yaw.

Fixes run between legs **and** from the monitor loop whenever the rover is
stopped — a parked rover that could only correct between legs drifted
uncorrected however long it sat there, while standing still in front of a panel
is the best possible moment to take a fix.

---

## 8. Safety

Layered, and deliberately independent:

1. **Planner** — routes keep the footprint inside the arena and off obstacles,
   maximising clearance.
2. **Obstacle e-stop** (20 Hz) — cancels the move if the footprint comes within
   `NAV_OBSTACLE_STOP_MARGIN_MM` of a raw obstacle.
3. **Bounds e-stop** (20 Hz) — cancels if the footprint leaves the arena. Unlike
   the obstacle stop this is **not** suppressed by *ignore obstacles*: driving
   out of a keep-out is a legitimate recovery, leaving the arena never is.
4. **Stall watchdog** — fails a leg after `STALL_TIMEOUT` without progress. It
   holds its timer while position is inside tolerance, since rotating to settle
   heading makes no distance progress *by design*.
5. **Drift margin** — accumulates 8 mm per metre driven and 1.5 mm per degree
   turned since the last fix, added to the planner's clearance floor and capped
   at 120 mm, so routes get conservative as the pose ages. Reset by every
   accepted tag fix.
6. **Jog dead-man** — hold-to-move stops within 0.5 s if the UI stops refreshing.

> The e-stops are pose-derived. Under large drift they fail *silently and
> simultaneously* with the planner, because all three consult the same map and
> the same wrong pose. That is precisely why §8 exists.

---

## 9. Running it

```bash
cd rover_ui
./start_thermal.sh     # FLIR relay → /dev/video1 (separate service)
./start.sh             # detached; logs to /tmp/rover_ui_v1.log
./stop.sh              # clean SIGTERM, then SIGKILL + zero-velocity failsafe
```

* Operator console — `http://<jetson>:8000/`
* Audience display — `http://<jetson>:8000/audience`

Mock mode needs no hardware: `ALLOW_MOCK=1 ./run_mock.sh`.

**Before trusting map positions**, place the rover physically on its start cell,
squared up, and press *Re-anchor pose to start cell*. Map yaw is defined by that
anchor: anchoring while skewed rotates the whole map frame.

Every value in `config.py` is environment-overridable:

```bash
NAV_MAX_LINEAR=0.15 TAGS_ENABLED=0 ./start.sh
```

---

## 10. Authoring an arena

`tools/map_config_editor.html` is a standalone browser tool (no build step) that
reads and writes `map.json` and `config.json` in the current schema, imports the
legacy one, edits tags per panel, and validates duplicate IDs, tags overhanging
a panel face, overlapping panels, and an illegal start cell.


1. Ensure center emergency stop button is depressed and rightmost is instead engaged - then press leftmost metal latching switch to turn on power, a green ring should illuminate 
2. Press & hold Power button on battery pack until screen illuminates then press “Switch” button once to turn on AC power distribution- the power draw should quickly get up to approx. 26W
3. Connect to “ICSL-Exp” WiFi network- password “Icslicsl”
4. ssh icsl@192.168.0.100 - password “Icslicsl”
5. cd /home/icsl/workspace/flirone-v4l2/ && sudo scripts/load_v4l2loopback [PASSWORD]
6. screen [ENTER]
7. sudo ./flirone palettes/Grayscale.raw - [CTRL + A + D] to disconnect process
8. screen [ENTER]
9. roslaunch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch  - [CTRL + A + D] to disconnect process
10. Two buttons on the usb hub on the gimbal arm should be deactivated- turn them on now with short presses to connect the T265 and D435i to the Orin (remember these will have to be manually turned off by holding the respective buttons before the next shutdown/reboot)
11. Start the Web UI with the command: /home/icsl/claude_related/demo_mk4/rover_ui/start.sh - it should return a series of IP addresses for the newly running services upon successful execution. The stop.sh script in the selfsame directory (/home/icsl/claude_related/demo_mk4/rover_ui/stop.sh) halts the system (though a reboot in between UI activations is HIGHLY recommended).
12. The Operator interface can be reached by navigating in a browser to 192.168.0.100:8000 with the audience interface being accessible at 192.168.0.100:8000/audience
13. There are three metal latches on the gimbal which are currently engaged to lock the axes for transport which need to be released - while holding the sensor array in place, gently push each latch away from its respective motor stator until you hear a distinctive click and the rotating axis unlocks and begins to spin freely freeing the gimbal motors.
14. While continuing to hold the sensor array - turn on the DJI gimbal by pressing and holding the side-mounted power button until the gimbal activates and centers the sensor array. 
15. Release motor brake by rotating rightmost button clockwise until it automatically depresses 
16. Rover is now ready - invert procedure to turn off
