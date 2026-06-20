"""Map-based navigation.

Owns the movement logic of the demo:

  * loads the pre-stored BEV map (map.json, mm units, x right / z down) with
    its marked obstacle regions and the rover's start cell;
  * tracks the rover's live position on that map by anchoring the T265 world
    pose (via purely_control.T265RoverService) to the configured start cell;
  * projects the detector's fused target (bearing deg + radar range m) into
    map coordinates — the "detection marker" the audience map shows;
  * computes the rover goal: the target shifted NAV_STANDOFF_MM toward the
    bottom of the map (the rover side), i.e. target (x, z) -> goal
    (x, z + 2000). If the rover footprint at that point overlaps an obstacle
    region (inflated by car/2 + clearance) the goal is nudged to the nearest
    non-overlapping point;
  * plans an obstacle-avoiding rectilinear path with
    rectilinear_mm.plan_rectilinear_path and executes it one axis-aligned leg
    at a time through T265RoverService.move_axis(axis, mm) — re-deriving each
    leg from the LIVE pose so per-move arrival error (~POS_TOL) does not
    accumulate along the path.

The T265 has exactly one owner in this process: the T265RoverService here.
(Do not start a second T265 pose sensor thread alongside this.)
"""
import json
import math
import os
import sys
import threading
import time

from .. import config

# Repo root holds rectilinear_mm.py and the purely_control package.
if config.DEMO_ROOT not in sys.path:
    sys.path.insert(0, config.DEMO_ROOT)

from rectilinear_mm import plan_rectilinear_path  # noqa: E402
from purely_control import T265RoverService       # noqa: E402


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


class Navigator:
    """Singleton owner of the map, the rover pose anchor, and move execution."""

    def __init__(self, detector=None):
        self._detector = detector
        self._lock = threading.Lock()

        self._anchor_map = None
        self._load_files(anchor_to_start=True)

        # Rover-motion service (sole T265 + cmd_vel owner in this process).
        self.rover = T265RoverService(
            allow_mock=config.ALLOW_MOCK,
            CMD_VEL_TOPIC=config.NAV_CMD_VEL_TOPIC,
            MAX_LINEAR=float(config.NAV_MAX_LINEAR),
            POS_TOL=float(config.NAV_POS_TOL),
            RAMP_STEP=float(config.NAV_RAMP_STEP),     # snappier start (clears base stiction)
            MIN_LINEAR=float(config.NAV_MIN_LINEAR),   # floor above the base's stiction speed
        )
        self._anchor_pose = None     # T265 pose snapshot mapped to _anchor_map

        # Navigation state (everything the UI shows lives here).
        self._status = "idle"        # idle|planning|moving|arrived|no_path|blocked|cancelled|error
        self._message = ""
        self._target = None          # {x, z, t, range_m, az_deg, source}
        self._target_win = []        # [(t, x, z)] raw projections for the median window
        self._track = None           # {x, z, t} continuity-gated target (anti-teleport)
        self._track_reject_since = None  # when a far-jump detection started being held
        self._goal = None            # {x, z, adjusted: bool}
        self._path = None            # [[x, z], ...] map waypoints of the active plan
        self._leg = 0                # index of the waypoint being driven to
        self._nav_thread = None
        self._cancel = threading.Event()

        self._jog_speed = float(config.NAV_JOG_SPEED)   # operator-settable
        self._auto = bool(config.NAV_AUTO)
        self._auto_hist = []         # [(t, x, z)] recent projected targets
        self._auto_last_nav = 0.0
        # FOLLOW mode: bounded-time continuous navigate-to-detection loop
        # (mutually exclusive with AUTO). Driven by its own _follow_loop thread.
        self._follow = bool(config.NAV_FOLLOW)
        self._stop = threading.Event()
        self._monitor = None
        self._follow_thread = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self.rover.start(wait_for_pose=True, timeout=8.0)
        self.reset_pose()            # current physical spot == map rover_start
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()
        self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._follow_thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.cancel()
        try:
            self.rover.shutdown()
        except Exception:
            pass

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_files(self, anchor_to_start=False):
        """(Re)read map.json + config.json. map/config are EDITABLE between
        runs — sizes, obstacle regions, car footprint all come from the files,
        nothing is hard-coded. Called at init and by reload_files()."""
        m = self._load_json(config.MAP_FILE)
        pc = self._load_json(config.PLAN_CONFIG_FILE)
        size = m.get("size", {})
        map_w = float(size.get("width", 6000))
        map_d = float(size.get("depth", 6000))
        car = pc.get("car", {})
        car_w = float(car.get("width", 800))
        car_l = float(car.get("length", 1200))
        clearance = float(pc.get("clearance", 300))
        # Inflated obstacle boxes (x1, z1, x2, z2): a rover CENTER inside one
        # of these means the rover footprint overlaps the raw obstacle region.
        mx = car_w / 2.0 + clearance
        mz = car_l / 2.0 + clearance
        inflated = [(o["x"] - mx, o["z"] - mz,
                     o["x"] + o["w"] + mx, o["z"] + o["h"] + mz)
                    for o in m.get("obstacles", [])]
        start = m.get("rover_start", {})
        # Default start: bottom-edge middle, rear bumper flush with the map's
        # bottom edge (center is half a car length up from it).
        map_start = (float(start.get("x", map_w / 2)),
                     float(start.get("z", map_d - car_l / 2)))
        with self._lock:
            self.map, self.plan_cfg = m, pc
            self.map_w, self.map_d = map_w, map_d
            self.car_w, self.car_l, self.clearance = car_w, car_l, clearance
            self._inflated = inflated
            self._map_start = map_start
            if anchor_to_start or self._anchor_map is None:
                self._anchor_map = map_start

    def reload_files(self):
        """Hot-reload an edited map.json / config.json without restarting.
        Refused mid-navigation. The live pose anchor is kept (the rover is
        where it is); use reset_pose() to re-anchor to the new rover_start."""
        with self._lock:
            if self._status in ("planning", "moving"):
                return False, "cannot reload while navigating (cancel first)"
        try:
            self._load_files(anchor_to_start=False)
        except Exception as exc:
            return False, f"reload failed: {exc}"
        with self._lock:
            self._goal, self._path, self._leg = None, None, 0
            self._status, self._message = "idle", "map reloaded"
        return True, "map + config reloaded"

    # ------------------------------------------------------------ pose
    def reset_pose(self, x=None, z=None):
        """Anchor the CURRENT physical pose to map cell (x, z) (default: the
        CURRENT map's rover_start — follows a reloaded map). Do this with the
        rover physically standing there."""
        pose = self.rover.get_pose()
        if pose is None:
            return False
        with self._lock:
            if x is not None or z is not None:
                self._anchor_map = (float(x if x is not None else self._anchor_map[0]),
                                    float(z if z is not None else self._anchor_map[1]))
            else:
                self._anchor_map = self._map_start
            self._anchor_pose = pose
        return True

    def pose(self):
        """Rover center in map mm: {x, z, yaw_deg} (None until anchored).

        The T265 world displacement is rotated by the ANCHOR yaw so the map
        frame is defined by how the rover was facing when it was anchored —
        re-anchoring with the rover squared up on the start cell corrects both
        position AND heading alignment of the whole map."""
        pose = self.rover.get_pose()
        with self._lock:
            anchor_pose, anchor_map = self._anchor_pose, self._anchor_map
        if pose is None or anchor_pose is None:
            return None
        dxr = (pose["right"] - anchor_pose["right"]) * 1000.0
        dxf = (pose["forward"] - anchor_pose["forward"]) * 1000.0
        c, s = math.cos(anchor_pose["yaw"]), math.sin(anchor_pose["yaw"])
        right_map = dxr * c + dxf * s      # displacement along map +x
        fwd_map = -dxr * s + dxf * c       # displacement up the map (-z)
        x = anchor_map[0] + right_map
        z = anchor_map[1] - fwd_map
        yaw = _wrap(pose["yaw"] - anchor_pose["yaw"])
        return {"x": x, "z": z, "yaw_deg": math.degrees(yaw)}

    # ------------------------------------------------------------ geometry
    def _center_free(self, x, z):
        """True when a rover CENTERED at (x, z) overlaps no obstacle region and
        its footprint stays inside the map."""
        hx = self.car_w / 2.0
        hz = self.car_l / 2.0
        if not (hx <= x <= self.map_w - hx and hz <= z <= self.map_d - hz):
            return False
        for (x1, z1, x2, z2) in self._inflated:
            if x1 < x < x2 and z1 < z < z2:
                return False
        return True

    def _footprint_blocked(self, x, z):
        """Emergency-stop test: True when a rover centered at (x, z) has its
        footprint (car half-extent + NAV_OBSTACLE_STOP_MARGIN_MM) overlapping a
        RAW obstacle. The planner keeps the center a full clearance away from
        this, so it never trips on a valid path — only when the plan/target was
        wrong or the rover drifted into a keep-out."""
        m = float(config.NAV_OBSTACLE_STOP_MARGIN_MM)
        hx = self.car_w / 2.0 + m
        hz = self.car_l / 2.0 + m
        for o in self.map.get("obstacles", []):
            if (o["x"] - hx < x < o["x"] + o["w"] + hx
                    and o["z"] - hz < z < o["z"] + o["h"] + hz):
                return True
        return False

    def compute_goal(self, tx, tz):
        """Standoff goal for a target at map (tx, tz). The NAV_STANDOFF_MM
        (2 m) is a SOFT preference — obstacle avoidance always wins:

          1. prefer (tx, tz + STANDOFF), straight below the target;
          2. if that overlaps an obstacle keep-out / leaves the map, first
             VARY THE STANDOFF along the approach line (stay directly below
             the target; longer slightly preferred over shorter, never closer
             than NAV_STANDOFF_MIN_MM to the person);
          3. lateral shifts are the last resort (cost-weighted 3x).

        Returns {x, z, adjusted} or None when nothing within the search
        radius keeps the footprint clear."""
        standoff = float(config.NAV_STANDOFF_MM)
        smin = float(config.NAV_STANDOFF_MIN_MM)
        gx, gz = tx, tz + standoff
        if self._center_free(gx, gz):
            return {"x": gx, "z": gz, "adjusted": False}
        step = float(config.NAV_ADJUST_STEP_MM)
        max_d = float(config.NAV_ADJUST_MAX_MM)
        best, best_cost = None, None
        n = int(max_d / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                dx, dz = i * step, j * step
                if dx == 0 and dz == 0:
                    continue
                cx, cz = gx + dx, gz + dz
                if cz < tz + smin:     # would crowd the person
                    continue
                if not self._center_free(cx, cz):
                    continue
                # Soft threshold: changing the standoff is cheap (shrinking a
                # touch dearer than growing), moving sideways is expensive.
                cost = 3.0 * abs(dx) + (1.5 * -dz if dz < 0 else 1.0 * dz)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (cx, cz), cost
        if best is None:
            return None
        return {"x": best[0], "z": best[1], "adjusted": True}

    # ------------------------------------------------------------ target projection
    def project_detection(self):
        """Project the detector's live fused target (bearing + radar range)
        into map mm and remember it. Returns the target dict or None."""
        if self._detector is None:
            return None
        tel, _ = self._detector.telemetry.get()
        if not tel:
            return None
        tgt = tel.get("target")
        if not tgt or tgt.get("range") is None or tgt.get("az") is None:
            return None
        # A radar-only target is only trusted while the rover itself is
        # stationary: ego-motion fakes Doppler on static clutter. The detector
        # already drops radar frames while moving (and resets its tracker);
        # this guard also rejects a smoothed/coasting radar target captured
        # just before the rover started to move.
        if tgt.get("source") == "radar" and self.rover.is_busy():
            return None
        p = self.pose()
        if p is None:
            return None
        rng_mm = float(tgt["range"]) * 1000.0
        bearing = math.radians(p["yaw_deg"] + float(tgt["az"]))
        tx = p["x"] + rng_mm * math.sin(bearing)
        tz = p["z"] - rng_mm * math.cos(bearing)
        # Clamp into the map so a noisy range can't paint the marker off-canvas.
        tx = max(0.0, min(self.map_w, tx))
        tz = max(0.0, min(self.map_d, tz))
        now = time.time()
        win = float(config.NAV_TARGET_WINDOW_S)
        with self._lock:
            # Median over a short window: a transient wrong detection is an
            # outlier and gets out-voted, so the target never jumps short-term;
            # a sustained move shifts the median once it dominates the window.
            self._target_win = [(ts, x, z) for (ts, x, z) in self._target_win
                                if now - ts <= win] + [(now, tx, tz)]
            fx = _median([x for _, x, _ in self._target_win])
            fz = _median([z for _, _, z in self._target_win])
            # Continuity gate (anti-teleport). Once locked, a detection that lands
            # more than NAV_TARGET_JUMP_MM from the current track is treated as a
            # glitch and DROPPED — the track holds its position, so a brief wrong
            # detection (or a target lost and re-found somewhere else) cannot make
            # the marker jump. A far detection only RE-LOCKS after it persists for
            # NAV_TARGET_RELOCK_S (the person genuinely walked there) or after the
            # track has gone stale (no accepted update for NAV_TARGET_FORGET_S,
            # i.e. a long loss). This is the same idea as the T265 pose-jump gate,
            # applied to the projected target in the map frame.
            jump = float(config.NAV_TARGET_JUMP_MM)
            relock = float(config.NAV_TARGET_RELOCK_S)
            forget = float(config.NAV_TARGET_FORGET_S)
            tr = self._track
            if tr is None or (now - tr["t"]) > forget:
                self._track = {"x": fx, "z": fz, "t": now}     # fresh lock
                self._track_reject_since = None
            elif math.hypot(fx - tr["x"], fz - tr["z"]) <= jump:
                a = 0.5                                        # near: accept (light EMA)
                self._track = {"x": a * fx + (1 - a) * tr["x"],
                               "z": a * fz + (1 - a) * tr["z"], "t": now}
                self._track_reject_since = None
            else:
                if self._track_reject_since is None:
                    self._track_reject_since = now
                if now - self._track_reject_since >= relock:
                    self._track = {"x": fx, "z": fz, "t": now}  # sustained -> re-lock
                    self._track_reject_since = None
                else:
                    tr["t"] = now                               # hold the last position
            fx, fz = self._track["x"], self._track["z"]
            t = {"x": fx, "z": fz, "t": now,
                 "range_m": round(float(tgt["range"]), 2),
                 "az_deg": float(tgt["az"]), "source": tgt.get("source")}
            self._target = t
        return t

    # ------------------------------------------------------------ navigation
    def navigate_to_detection(self):
        """Plan + drive to the standoff goal of the CURRENT detection target."""
        t = self.project_detection()
        if t is None:
            return False, "no detection target with range available"
        return self.navigate_to_target(t["x"], t["z"])

    def navigate_to_target(self, tx, tz):
        """Plan + drive to the standoff goal of a target at map (tx, tz)."""
        goal = self.compute_goal(float(tx), float(tz))
        with self._lock:
            self._target = (self._target
                            if self._target and self._target.get("x") == float(tx)
                            else {"x": float(tx), "z": float(tz), "t": time.time(),
                                  "range_m": None, "az_deg": None, "source": "manual"})
        if goal is None:
            with self._lock:
                self._status, self._message, self._goal, self._path = \
                    "blocked", "no free standoff position near the target", None, None
            return False, self._message
        return self._dispatch(goal)

    def return_to_start(self):
        """Drive straight back to the configured start cell (map.json
        rover_start). Disables FOLLOW/AUTO and cancels any in-flight navigation
        first, then plans + drives to the start cell itself (no standoff)."""
        if self._map_start is None:
            return False, "no start cell defined"
        with self._lock:
            self._follow = False
            self._auto = False
        self.cancel()                       # stop + join any running nav
        gx, gz = float(self._map_start[0]), float(self._map_start[1])
        with self._lock:
            self._target = {"x": gx, "z": gz, "t": time.time(),
                            "range_m": None, "az_deg": None, "source": "home"}
        return self._dispatch({"x": gx, "z": gz, "adjusted": False})

    def _dispatch(self, goal):
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return False, "a navigation is already running (cancel it first)"
        p = self.pose()
        if p is None:
            return False, "no rover pose yet"
        if math.hypot(goal["x"] - p["x"], goal["z"] - p["z"]) <= float(config.NAV_GOAL_TOLERANCE_MM):
            with self._lock:
                self._goal, self._status, self._message = goal, "arrived", "already at the goal"
            return True, "already at the goal"
        start = (round(p["x"]), round(p["z"]))
        end = (round(goal["x"]), round(goal["z"]))
        with self._lock:
            self._goal = goal
            self._status, self._message = "planning", ""
            self._path, self._leg = None, 0
        segs = plan_rectilinear_path(self.map, self.plan_cfg, start, end)
        if segs is None:
            with self._lock:
                self._status = "no_path"
                self._message = "planner found no obstacle-free rectilinear path"
            return False, self._message
        # Segments -> absolute waypoints (so each leg can be re-derived live).
        wps, cx, cz = [], float(start[0]), float(start[1])
        for axis, d in segs:
            if axis == "x":
                cx += d
            else:
                cz += d
            wps.append((cx, cz))
        with self._lock:
            self._path = [[round(start[0]), round(start[1])]] + [[round(x), round(z)] for x, z in wps]
            self._status = "moving"
        self._cancel.clear()
        self._nav_thread = threading.Thread(target=self._drive, args=(wps,), daemon=True)
        self._nav_thread.start()
        return True, f"navigating: {len(wps)} leg(s)"

    def _drive(self, waypoints):
        try:
            for i, (wx, wz) in enumerate(waypoints):
                with self._lock:
                    self._leg = i + 1
                # Re-derive this leg from the LIVE pose so per-move arrival
                # error does not accumulate across legs.
                p = self.pose()
                if p is None:
                    raise RuntimeError("lost rover pose")
                dx = wx - p["x"]
                dz = wz - p["z"]
                if math.hypot(dx, dz) < 5.0:  # sub-tolerance residual: skip
                    continue
                if self._cancel.is_set():
                    return
                # Map-frame delta -> the move's own start-heading frame: rotate
                # by -(yaw relative to anchor) so the leg lands on the map
                # waypoint even when the rover's heading has drifted. One
                # mecanum move per waypoint (handles both axes + residuals).
                g = math.radians(p["yaw_deg"])
                mr, mf = dx, -dz              # map delta as (right, forward)
                cr = mr * math.cos(g) + mf * math.sin(g)
                cf = -mr * math.sin(g) + mf * math.cos(g)
                # Hold true map-forward (the T265 yaw that == anchor heading)
                # so the rover corrects accumulated yaw drift WHILE driving this
                # leg, instead of locking in whatever heading it started with.
                with self._lock:
                    hold_yaw = self._anchor_pose["yaw"] if self._anchor_pose else None
                # Run the leg non-blocking so we can poll an obstacle safety check
                # at ~20 Hz while it drives, and e-stop if the footprint enters a
                # keep-out (a wrong plan/target or pose drift would otherwise let
                # the rover plough into an obstacle).
                goal = self.rover.move(right=cr, forward=cf, units="mm",
                                       hold_yaw=hold_yaw, blocking=False)
                while self.rover.is_busy():
                    if self._cancel.is_set() or self._stop.is_set():
                        self.rover.stop()
                        return
                    lp = self.pose()
                    if lp is not None and self._footprint_blocked(lp["x"], lp["z"]):
                        self.rover.stop()
                        self._cancel.set()
                        with self._lock:
                            self._status = "blocked"
                            self._message = ("E-STOP: rover footprint entered an obstacle "
                                             "zone at (%.0f, %.0f) mm" % (lp["x"], lp["z"]))
                        return
                    self._stop.wait(0.05)   # ~20 Hz safety poll
                res = self.rover.wait(goal)
                if self._cancel.is_set():
                    return
                if res is None or not res:
                    reason = res.reason if res is not None else "no result"
                    with self._lock:
                        self._status = "error"
                        self._message = (f"leg {i + 1} ({dx:+.0f}, {dz:+.0f})mm "
                                         f"failed ({reason})")
                    return
            with self._lock:
                self._status, self._message = "arrived", ""
        except Exception as exc:
            with self._lock:
                self._status, self._message = "error", str(exc)

    def cancel(self):
        self._cancel.set()
        try:
            self.rover.stop()
        except Exception:
            pass
        t = self._nav_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            if self._status in ("moving", "planning"):
                self._status, self._message = "cancelled", ""

    def is_moving(self):
        """For the detector's radar tracker: is the rover translating?
        True during goal moves AND manual jogs (both fake radar Doppler)."""
        return self.rover.is_busy()

    # ------------------------------------------------------------ speed
    def set_speed(self, mps):
        """Set the navigation translation-speed cap (MAX_LINEAR), clamped to
        [NAV_SPEED_MIN, NAV_SPEED_MAX]. The control loop reads cfg.MAX_LINEAR
        live, so this takes effect on the next tick — even mid-move. Returns
        the clamped value actually applied."""
        lo, hi = float(config.NAV_SPEED_MIN), float(config.NAV_SPEED_MAX)
        v = max(lo, min(hi, float(mps)))
        self.rover.cfg.MAX_LINEAR = v
        return v

    def set_jog_speed(self, mps):
        """Set the hold-to-move (Drive pad) speed, clamped to
        [NAV_JOG_SPEED_MIN, NAV_JOG_SPEED_MAX]. The rover service additionally
        hard-caps jog at MAX_LINEAR, so the effective speed is
        min(this, the nav speed cap). Returns the clamped value."""
        lo, hi = float(config.NAV_JOG_SPEED_MIN), float(config.NAV_JOG_SPEED_MAX)
        self._jog_speed = max(lo, min(hi, float(mps)))
        return self._jog_speed

    # ------------------------------------------------------------ manual jog
    def jog(self, dx, dz):
        """Hold-to-move at NAV_JOG_SPEED. dx/dz are map directions in {-1,0,1}
        (x: +right, z: +down/back; forward = -z). The UI must keep calling this
        every ~200 ms while the button is held — the service's dead-man
        (NAV_JOG_DEADMAN_S) stops the rover otherwise. Refused mid-navigation."""
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return False, "navigation in progress"
        speed = self._jog_speed
        ok = self.rover.jog(forward=-dz * speed, right=dx * speed,
                            duration=float(config.NAV_JOG_DEADMAN_S))
        return (True, "jogging") if ok else (False, "rover busy")

    def jog_stop(self):
        self.rover.jog_stop()

    # ------------------------------------------------------------ auto / follow mode
    def set_auto(self, enabled):
        with self._lock:
            self._auto = bool(enabled)
            self._auto_hist = []
            if self._auto:
                self._follow = False     # AUTO and FOLLOW are mutually exclusive

    def set_follow(self, enabled):
        """Toggle FOLLOW: a continuous navigate-to-detection loop with a per-cycle
        time cap (see _follow_loop). Turning it on disables AUTO."""
        with self._lock:
            self._follow = bool(enabled)
            if self._follow:
                self._auto = False
                # Fresh lock: drop any stale track so follow locks onto wherever
                # the person is right now (no carry-over teleport from before).
                self._track = None
                self._track_reject_since = None

    def _follow_loop(self):
        """FOLLOW = a strict SENSE-while-stationary -> COMMIT -> MOVE -> re-stop
        -> SENSE cycle.

        Because mmWave reflection strength depends heavily on the occluding
        material (so an absolute SNR threshold is unreliable), the moving-point
        radar target is trusted ONLY under a hard precondition: the rover is
        confirmed STATIONARY. Each cycle:

          1. SENSE (rover stationary): take the fused detection. RGB/thermal come
             FIRST — a vision target is used immediately. A radar (moving-point)
             target is accepted ONLY after the rover has been confirmed stationary
             for NAV_FOLLOW_STILL_CONFIRM_S, long enough that the detector
             re-acquired a MOVING cluster while still (not a value carried over
             from before the last move).
          2. COMMIT + MOVE: plan + drive to the standoff goal. A radar-committed
             move runs to completion (radar is invalid while moving); a
             vision-committed move may re-plan early if the person walks
             > NAV_FOLLOW_REPLAN_MM. NAV_FOLLOW_CYCLE_S is a stuck-backstop.
          3. Arrive -> stop -> a fresh stationary window opens -> back to SENSE."""
        idle = float(config.NAV_FOLLOW_MIN_INTERVAL_S)
        replan_mm = float(config.NAV_FOLLOW_REPLAN_MM)
        cap = float(config.NAV_FOLLOW_CYCLE_S)
        still_confirm = float(config.NAV_FOLLOW_STILL_CONFIRM_S)
        still_since = time.time()
        was_following = False
        while not self._stop.is_set():
            with self._lock:
                follow = self._follow
            if not follow:
                was_following = False
                self._stop.wait(0.1)
                continue
            if not was_following:                  # just enabled -> fresh dwell
                was_following = True
                still_since = time.time()
            # ---------------- SENSE PHASE (must be stationary) ----------------
            if self.rover.is_busy():
                still_since = time.time()          # still moving -> reset the dwell
                self._stop.wait(0.1)
                continue
            t = self.project_detection()
            if t is None:
                self._stop.wait(idle)
                continue
            # HARD INVARIANT: a radar/moving-point target is accepted ONLY once the
            # rover has been confirmed stationary long enough to have re-acquired a
            # moving cluster while still. Vision (RGB/thermal) is unconstrained.
            if t.get("source") == "radar" and (time.time() - still_since) < still_confirm:
                with self._lock:
                    if self._status not in ("moving", "planning"):
                        self._status = "idle"
                        self._message = "follow: confirming stationary mmWave lock…"
                self._stop.wait(0.1)
                continue
            # ---------------- COMMIT + MOVE PHASE ----------------
            committed_src = t.get("source")
            ok, _msg = self.navigate_to_target(t["x"], t["z"])
            with self._lock:
                navving = self._nav_thread is not None and self._nav_thread.is_alive()
            if not ok or not navving:
                # Nothing to drive: already at the goal, blocked, or no path.
                self._stop.wait(idle)
                continue
            gx, gz = t["x"], t["z"]
            deadline = time.time() + cap
            while not self._stop.is_set():
                with self._lock:
                    still = self._follow
                    alive = self._nav_thread is not None and self._nav_thread.is_alive()
                if not alive:
                    break                          # arrived/error/blocked -> cycle done
                if not still:
                    break                          # follow turned off -> let it finish
                # Mid-move re-plan ONLY for a VISION target that moved. A radar move
                # is committed and runs to completion — radar is invalid while
                # moving, so we never re-target it until the rover stops again.
                if committed_src == "vision":
                    nt = self.project_detection()
                    if (nt is not None and nt.get("source") == "vision"
                            and math.hypot(nt["x"] - gx, nt["z"] - gz) >= replan_mm):
                        self.cancel()
                        with self._lock:
                            self._status = "moving"
                            self._message = "follow: target moved, re-targeting"
                        break
                if time.time() >= deadline:
                    self.cancel()                  # stuck-backstop -> treat as done
                    with self._lock:
                        self._status = "arrived"
                        self._message = "follow: cycle time cap reached, re-targeting"
                    break
                self._stop.wait(0.1)
            # Move ended -> rover is stopping; open a fresh stationary window so the
            # next radar lock must be re-confirmed while still.
            still_since = time.time()

    def _monitor_loop(self):
        """Continuously project the detection onto the map (for the UI), and in
        AUTO mode dispatch a navigation once the target holds still. (FOLLOW is
        driven by its own _follow_loop.)"""
        while not self._stop.is_set():
            t = self.project_detection()
            with self._lock:
                auto = self._auto
                busy = self._status in ("planning", "moving")
            if auto and t is not None and not busy:
                now = t["t"]
                self._auto_hist = [(ts, x, z) for ts, x, z in self._auto_hist
                                   if now - ts <= float(config.NAV_AUTO_STABLE_S)] \
                    + [(now, t["x"], t["z"])]
                if (now - self._auto_last_nav >= float(config.NAV_AUTO_COOLDOWN_S)
                        and self._auto_hist
                        and now - self._auto_hist[0][0] >= float(config.NAV_AUTO_STABLE_S) * 0.9):
                    xs = [x for _, x, _ in self._auto_hist]
                    zs = [z for _, _, z in self._auto_hist]
                    if (max(xs) - min(xs) <= float(config.NAV_AUTO_STABLE_MM)
                            and max(zs) - min(zs) <= float(config.NAV_AUTO_STABLE_MM)):
                        goal = self.compute_goal(t["x"], t["z"])
                        p = self.pose()
                        if goal is not None and p is not None and math.hypot(
                                goal["x"] - p["x"], goal["z"] - p["z"]) > float(config.NAV_GOAL_TOLERANCE_MM):
                            self._auto_last_nav = now
                            self.navigate_to_target(t["x"], t["z"])
            self._stop.wait(0.2)

    # ------------------------------------------------------------ state for the UI
    def map_payload(self):
        return {
            "size": {"width": self.map_w, "depth": self.map_d},
            "obstacles": self.map.get("obstacles", []),
            "car": {"width": self.car_w, "length": self.car_l},
            "clearance": self.clearance,
            # The CONFIGURED start cell (map.json rover_start) — i.e. where the
            # operator should physically place the rover, which is what the
            # "Re-anchor to start cell" button anchors to. Not the live anchor,
            # which only matches after a re-anchor.
            "rover_start": {"x": self._map_start[0], "z": self._map_start[1]},
            "standoff_mm": float(config.NAV_STANDOFF_MM),
            "speed": round(float(self.rover.cfg.MAX_LINEAR), 3),
            "speed_min": float(config.NAV_SPEED_MIN),
            "speed_max": float(config.NAV_SPEED_MAX),
            "jog_speed": round(float(self._jog_speed), 3),
            "jog_speed_min": float(config.NAV_JOG_SPEED_MIN),
            "jog_speed_max": float(config.NAV_JOG_SPEED_MAX),
        }

    def state(self):
        p = self.pose()
        with self._lock:
            target = dict(self._target) if self._target else None
            goal = dict(self._goal) if self._goal else None
            path = [list(w) for w in self._path] if self._path else None
            status, message, leg = self._status, self._message, self._leg
            auto, follow = self._auto, self._follow
        if target is not None:
            age = time.time() - target.pop("t", 0)
            target["age_s"] = round(age, 1)
            if age > float(config.NAV_TARGET_HOLD_S):
                target["stale"] = True
        return {
            "rover": ({"x": round(p["x"]), "z": round(p["z"]),
                       "yaw_deg": round(p["yaw_deg"], 1)} if p else None),
            "target": target,
            "goal": goal,
            "path": path,
            "leg": leg,
            "status": status,
            "message": message,
            "auto": auto,
            "follow": follow,
            "speed": round(float(self.rover.cfg.MAX_LINEAR), 3),
            "jog_speed": round(float(self._jog_speed), 3),
            "moving": self.rover.is_busy(),
            "mock": self.rover.using_mock,
            "ros": self.rover.ros_connected,
            "pose_jumps": getattr(self.rover, "pose_jumps", 0),
        }
