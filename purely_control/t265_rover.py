"""Closed-loop rover motion driven purely by the Intel RealSense T265.

This is a small *service*: start it once, then call :meth:`T265RoverService.move_axis`
(or :meth:`move`) from anywhere to drive the rover a precise signed distance and
have it stop itself. There is no UI and no dependency on the ``rover_ui`` stack —
it talks to the T265 (pose feedback) and the ROS ``cmd_vel`` publisher directly.

    from purely_control.t265_rover import T265RoverService

    with T265RoverService() as rover:
        rover.move_axis("x", 500, units="mm")    # +x (right) 500 mm
        rover.move_axis("z", -200, units="mm")   # -z (forward) 200 mm

Frames
------
Axes are the **T265 native device frame** (right-handed):

    +x = right,  +y = up,  +z = back   =>   forward = -z

So ``move_axis("x", 0.5)`` strafes 0.5 m to the right and ``move_axis("z", -0.2)``
drives 0.2 m forward. ``y`` (vertical) is rejected — a ground rover can't move along it.

The base is assumed **holonomic/mecanum**: lateral motion is commanded directly
via ``Twist.linear.y`` (no rotate-drive-rotate). A heading-hold loop keeps the
rover pointing the way it started so strafing tracks a straight line.

How a move works
----------------
Each move is *relative to wherever the rover is now*. On entry we snapshot the
current pose as the origin; every tick we re-express the live pose in that
start-heading body frame, so ``(right, forward)`` is the displacement since the
move began. A proportional controller with an ease-in near the target drives the
displacement to the goal, holds heading, then stops once it has settled inside
tolerance. A per-move timeout is the safety backstop.
"""
import math
import threading
import time

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - allow running/testing without the SDK
    rs = None

try:
    import rospy
    from geometry_msgs.msg import Twist
except Exception:  # pragma: no cover - allow running/testing without ROS
    rospy = None
    Twist = None


# --------------------------------------------------------------------------- #
# Tunables. Every one can be overridden per-instance via the constructor.
# Conservative speeds: this is a "go an exact distance and stop" controller,
# not a teleop, so precision matters more than being fast.
# --------------------------------------------------------------------------- #
class Config:
    CONTROL_HZ = 20            # Twist publish / control rate
    POSE_HZ = 100              # how fast we drain T265 pose frames (mock only)

    # Linear position control (applies to both right and forward axes).
    # Speed is proportional to remaining error (so it eases in near the goal),
    # but floored at MIN_LINEAR so it never crawls below what the base can
    # actually execute — the last centimetres are covered at a real speed and
    # motion stops cleanly at the tolerance instead of asymptotically.
    LINEAR_GAIN = 1.5          # (m/s) per m of error
    MAX_LINEAR = 0.18          # m/s cap on translation speed
    MIN_LINEAR = 0.045         # m/s floor while outside tolerance (base stiction)
    RAMP_STEP = 0.03           # m/s max change in commanded speed per tick (anti-jerk)
    # Cross-axis stiffening: mecanum bases drift on the axis they are NOT
    # driving (e.g. forward creep during a strafe). The minor axis of a move —
    # and the held axis of a jog — is corrected at full LINEAR_GAIN, capped
    # here, instead of waiting for its share of the error-direction vector.
    MAX_CROSS = 0.10           # m/s cap on minor-axis drift correction

    # When are we "there"?
    POS_TOL = 0.02             # m: within 20 mm of the goal counts as arrived
    SETTLE_TIME = 0.30         # s: must hold inside POS_TOL this long before stopping
    # End-stage yaw correction is BOUNDED: heading is corrected continuously
    # while translating, so on arrival it is usually already settled. If it is
    # not, the rover rotates in place to fix it — but only for this long, then
    # the move finishes as "arrived" (position is good; we never spin forever
    # chasing the last fraction of a degree). 0 disables the in-place phase.
    # Max time spent rotating in place once position is reached. 0 = never do it:
    # accept arrival on position alone and let the next leg's heading-hold
    # correct the residual while actually making progress. The rover never turns
    # deliberately (omni base, fixed sensor heading), so all yaw is drift, and
    # correcting it standing still is where the oscillation lived.
    ARRIVE_SETTLE_TIMEOUT = 0.0

    # Heading-hold: actively drive yaw back to the start heading every tick so
    # the rover translates without rotating (a strafe stays straight, and any
    # wheel-slip yaw is corrected continuously). A move is not "arrived" until
    # heading is back within YAW_TOL as well as position within POS_TOL, so it
    # can never finish leaving the rover rotated.
    YAW_GAIN = 1.5             # P: rad/s per rad of heading error
    YAW_IGAIN = 0.9            # I: rejects a *constant* yaw bias (e.g. a base that
                               #    drifts slightly while strafing) so heading
                               #    returns fully to zero, not just close
    YAW_I_MAX = 0.35           # rad/s cap on the integral contribution (anti-windup)
    # Deadband MUST be >= YAW_TOL. When the band the controller stops correcting
    # in is narrower than the band it is judged against, it keeps nudging inside
    # the "arrived" window and hunts. Was 0.012 (0.7 deg) against a 0.020 tol.
    YAW_DEADBAND = 0.020       # rad (~1.1 deg): below this, leave yaw alone
    YAW_TOL = 0.020            # rad (~1.1 deg): heading must be within this to arrive
    YAW_I_BLEED = 2.0          # 1/s: how fast the integral decays once settled
    MAX_YAW = 0.6              # rad/s cap
    RAMP_STEP_YAW = 0.5        # rad/s max change per tick

    # Safety.
    # Stiction integral for TRANSLATION. Pure P cannot push through static
    # friction; this accrues while the rover is commanded to move but is closing
    # on the goal slower than LIN_STICTION_EPS, and bleeds off once it moves
    # properly. Integrating time-stuck rather than distance-error keeps it from
    # winding up during normal cruise and overshooting at the end of a leg.
    LIN_IGAIN = 0.6            # m/s of boost per second spent stuck
    LIN_I_MAX = 0.10           # m/s cap on the boost (anti-windup)
    LIN_STICTION_EPS = 0.02    # m/s closing speed below which "stuck" accrues
    LIN_I_BLEED = 3.0          # bleed-off multiplier once moving properly
    MOVE_TIMEOUT_BASE = 9.0    # s of slack added to every move's deadline
    MOVE_MIN_SPEED = 0.04      # m/s assumed worst-case progress for the deadline calc
    LOST_CONF = 0              # T265 tracker_confidence == this => tracking failed

    # Stall watchdog. MOVE_TIMEOUT_BASE/MOVE_MIN_SPEED above sizes a per-move
    # deadline for the WORST-CASE real speed, which can be tens of seconds to
    # minutes for a normal leg — fine as a hard backstop, but far too slow to
    # notice "commanded to move but genuinely isn't" (stuck wheel, e-stop,
    # motor fault) or a T265 tracking loss that never recovers. Instead, track
    # the best (smallest) remaining distance-to-goal seen so far; if it hasn't
    # improved by STALL_MIN_PROGRESS_M within STALL_TIMEOUT seconds, fail the
    # move immediately with reason "stalled" so the caller (Navigator) can
    # clear the failed move and accept a new command right away instead of
    # being blocked behind it until the full deadline elapses.
    STALL_TIMEOUT = 5.0           # s of no meaningful progress before failing fast
    STALL_MIN_PROGRESS_M = 0.005  # m: minimum shrink in remaining error that counts as progress

    # ROS wiring (matches the verified WHEELTEC base in the sibling project).
    ROS_NODE_NAME = "purely_control"
    CMD_VEL_TOPIC = "/cmd_vel"
    T265_SERIAL = ""           # "" => first T265 found
    # The T265 often enumerates but refuses to start from a stale USB state
    # ("No device connected"); a hardware reset + retry clears it on open().
    T265_OPEN_RETRIES = 2
    T265_RESET_WAIT = 5.0      # s to wait after a hardware_reset re-enumerates

    # Lever arm: where the T265 sits relative to the rover's TURN CENTRE, in the
    # rover body frame, METRES (+fwd = toward the front, +right = to the right).
    # Rotating in place swings a sensor mounted off centre through an arc, which
    # it reports as real translation; without this the believed position slides
    # sideways on every heading correction. Compensated in _make_pose so that
    # EVERYTHING downstream — the control loop's own distance-to-goal included —
    # works in rover-centre coordinates.
    SENSOR_OFFSET_FWD = 0.060
    SENSOR_OFFSET_RIGHT = 0.0
    # Sign flips to reconcile the T265 mounting/convention with the base wiring.
    # Right is +x (T265); ROS body +y is LEFT, so right maps to -linear.y by
    # default. Forward is +linear.x. YAW_SIGN is the important one: if the T265's
    # yaw increases in the OPPOSITE sense to a positive angular.z on your base,
    # heading-hold becomes positive feedback and the rover spins up — set
    # YAW_SIGN=-1. Verify all three with the bring-up test (see README) at low
    # speed before trusting larger moves.
    X_SIGN = 1                 # flip if forward/back is reversed on your base
    Y_SIGN = 1                 # flip if strafe goes the wrong way
    YAW_SIGN = 1               # flip if a heading correction turns the WRONG way

    # Pose jump rejection. On repetitive-texture scenes the T265 can relocalise
    # to a wrong place and TELEPORT the pose in a single frame. A wheeled rover
    # can't move that fast, so any frame-to-frame delta implying a speed above
    # POSE_JUMP_MAX_SPEED (or a yaw rate above POSE_JUMP_MAX_YAW_RATE) is treated
    # as a glitch and NOT propagated into the reported pose (the raw baseline is
    # still advanced, so normal motion after a relocalisation step keeps being
    # tracked from the new position). The *_BASE terms allow per-frame noise
    # regardless of frame interval. Set POSE_JUMP_MAX_SPEED huge to disable.
    # Accept a pose STEP while the rover is commanded stationary. The T265
    # relocalises against its own continuously-built map; those corrections
    # arrive as one large step, which the filter below otherwise discards
    # permanently, re-applying drift the device had just removed. A real
    # teleport is impossible while stopped, so a step there is either a
    # relocalisation (wanted) or a tracking fault. Steps DURING motion are still
    # rejected. Set False to restore the old always-reject behaviour.
    POSE_JUMP_ACCEPT_WHEN_STILL = True
    POSE_STILL_EPS = 0.005         # m/s — commanded speed below which "stationary"
    POSE_JUMP_MAX_SPEED = 1.5      # m/s — well above the rover's real top speed
    POSE_JUMP_BASE_M = 0.05        # m — per-frame position allowance (noise)
    POSE_JUMP_MAX_YAW_RATE = 4.0   # rad/s — ~230 deg/s, above any real turn here
    POSE_JUMP_BASE_RAD = 0.10      # rad — per-frame yaw allowance (noise)

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(Config, k):
                raise AttributeError(f"unknown config key: {k}")
            setattr(self, k, v)


class MoveResult:
    """Outcome of a single move. Truthy when the goal was reached."""

    __slots__ = ("ok", "reason", "axis", "target", "reached", "error",
                 "yaw_error", "elapsed")

    def __init__(self, ok, reason, axis, target, reached, error, yaw_error, elapsed):
        self.ok = ok              # bool
        self.reason = reason      # "arrived" | "timeout" | "cancelled" | "tracking_lost" | "stalled"
        self.axis = axis          # e.g. "x" / "z" / "right,forward"
        self.target = target      # (right_m, forward_m) goal in the start-body frame
        self.reached = reached    # (right_m, forward_m) actually reached
        self.error = error        # remaining distance to goal (m)
        self.yaw_error = yaw_error  # residual heading error vs start (rad); ~0 on success
        self.elapsed = elapsed    # seconds the move took

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return (f"MoveResult(ok={self.ok}, reason={self.reason!r}, axis={self.axis!r}, "
                f"target={_r2(self.target)}, reached={_r2(self.reached)}, "
                f"error={self.error*1000:.0f}mm, "
                f"yaw_error={math.degrees(self.yaw_error):.1f}deg, "
                f"elapsed={self.elapsed:.1f}s)")


class _Goal:
    """A live move target plus the pose snapshot it is measured against."""

    __slots__ = ("right", "forward", "axis", "rx0", "rf0", "yaw0", "hold_yaw",
                 "t_start", "deadline", "done", "result", "in_tol_since", "pos_since",
                 "best_dist", "best_dist_t", "lin_i", "prev_dist", "prev_t")

    def __init__(self, right, forward, axis, pose, timeout, hold_yaw=None):
        self.right = right            # goal displacement, start-body frame
        self.forward = forward
        self.axis = axis
        self.rx0 = pose["rx"]         # origin snapshot (T265 right/forward/yaw)
        self.rf0 = pose["rf"]
        self.yaw0 = pose["yaw"]       # frame the displacement is measured in
        # Heading the rover should FACE while moving. Defaults to the start
        # heading (hold whatever we had); a caller can pass an ABSOLUTE
        # reference (e.g. true map-forward) so accumulated yaw drift is
        # corrected toward it continuously, not just held at the start value.
        self.hold_yaw = pose["yaw"] if hold_yaw is None else hold_yaw
        self.t_start = pose["_now"]
        self.deadline = self.t_start + timeout
        self.done = threading.Event()
        self.result = None
        self.in_tol_since = None
        self.pos_since = None         # when position first entered POS_TOL
        self.best_dist = None         # smallest remaining distance-to-goal seen (stall watchdog)
        self.best_dist_t = self.t_start  # when best_dist last improved
        self.lin_i = 0.0              # stiction integral, seconds-stuck (per leg)
        self.prev_dist = None         # previous tick's distance-to-goal, for closing speed
        self.prev_t = self.t_start


class T265RoverService:
    """Drive a mecanum rover exact distances using T265 pose feedback.

    Lifecycle: :meth:`start` (opens the camera + ROS and spins up the control
    thread), then any number of :meth:`move_axis` / :meth:`move` calls, then
    :meth:`shutdown`. Usable as a context manager, which does start/shutdown.
    """

    def __init__(self, config=None, allow_mock=True, **config_overrides):
        if config is not None and config_overrides:
            raise ValueError("pass either a Config or keyword overrides, not both")
        self.cfg = config or Config(**config_overrides)
        self.allow_mock = allow_mock

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pose = None             # latest pose dict (see _make_pose)
        self._pose_event = threading.Event()
        self._goal = None             # current _Goal or None
        self._jog = None              # manual hold-to-move: {"f","r","until","yaw0"}
        self._out = (0.0, 0.0, 0.0)   # commanded (forward, right, yaw) body velocity
        self._yaw_i = 0.0             # heading-hold integral accumulator (per move)
        # Pose jump-rejection state (real T265 only).
        self._filt = None             # [rx, rf, yaw] reported (de-glitched) pose
        self._raw_last = None         # [rx, rf, yaw] previous RAW T265 pose
        self._filt_last_t = 0.0
        self._pose_jumps = 0          # count of rejected teleports (diagnostics)
        self._reloc_accepted = 0      # count of relocalisation steps ACCEPTED while still
        self._cmd_still = True        # is the commanded body velocity ~zero right now?
                                      # (set by the control loop, read by _filter_pose)
        self._last_jump_log = 0.0

        self._pipe = None
        self._mock = False
        self._pub = None
        self._ros_ok = False

        self._pose_thread = None
        self._ctrl_thread = None
        self._started = False

    # ------------------------------------------------------------------ API #
    def start(self, wait_for_pose=True, timeout=5.0):
        """Open the T265 + ROS and start the background threads.

        Blocks until the first pose arrives (so a move right after start has
        real feedback). Raises if no pose appears within ``timeout`` and mock
        is disabled.
        """
        if self._started:
            return self
        self._open_t265()
        self._init_ros()
        self._pose_thread = threading.Thread(target=self._pose_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._pose_thread.start()
        self._ctrl_thread.start()
        self._started = True
        if wait_for_pose and not self._pose_event.wait(timeout):
            raise RuntimeError("no T265 pose within %.1fs (camera not streaming?)" % timeout)
        return self

    def move_axis(self, axis, distance, units="m", blocking=True, timeout=None):
        """Move ``distance`` along a single **T265-native** axis, then stop.

        ``axis`` is ``"x"`` (right), ``"z"`` (back; forward = -z), or a friendly
        alias: ``"right"/"left"/"forward"/"back"``. ``distance`` is signed.
        ``units`` is ``"m"`` (default) or ``"mm"``.

        Examples::

            move_axis("x", 500, units="mm")   # right 500 mm
            move_axis("z", -200, units="mm")  # forward 200 mm  (-z is forward)
            move_axis("forward", 0.3)         # forward 0.3 m
        """
        d = _to_meters(distance, units)
        a = axis.strip().lower().lstrip("+")
        if a in ("x", "right"):
            right, forward = d, 0.0
        elif a == "left":
            right, forward = -d, 0.0
        elif a == "z":                     # native z: +z is back, so forward = -z
            right, forward = 0.0, -d
        elif a in ("forward", "fwd"):
            right, forward = 0.0, d
        elif a in ("back", "backward"):
            right, forward = 0.0, -d
        elif a == "y":
            raise ValueError("axis 'y' is vertical (up); a ground rover can't move along it")
        else:
            raise ValueError(f"unknown axis {axis!r} (use x/z/right/left/forward/back)")
        return self.move(right=right, forward=forward, blocking=blocking,
                          timeout=timeout, _axis=a)

    def move(self, right=0.0, forward=0.0, units="m", blocking=True, timeout=None,
             hold_yaw=None, _axis=None):
        """Move to ``(right, forward)`` displacement (start-body frame) and stop.

        ``right`` and ``forward`` are in the rover's intuitive frame: +right
        strafes right, +forward drives ahead. A diagonal goal is driven directly
        (mecanum). Set ``blocking=False`` to return immediately; poll with
        :meth:`is_busy` / :meth:`wait`.

        ``hold_yaw`` (rad, T265 world frame) is the heading the rover should
        FACE during the move. Default (None) holds the heading it had at the
        start. Pass an absolute reference to actively correct accumulated yaw
        drift toward it while translating (e.g. keep facing true map-forward).
        """
        if not self._started:
            raise RuntimeError("call start() before move()")
        right = _to_meters(right, units)
        forward = _to_meters(forward, units)
        axis = _axis if _axis is not None else _r2((right, forward))

        # Snapshot the current pose as this move's origin.
        pose = self._latest_pose()
        if pose is None:
            raise RuntimeError("no T265 pose available")
        dist = math.hypot(right, forward)
        if timeout is None:
            timeout = self.cfg.MOVE_TIMEOUT_BASE + dist / max(self.cfg.MOVE_MIN_SPEED, 1e-3)
        goal = _Goal(right, forward, axis, pose, timeout, hold_yaw=hold_yaw)

        with self._lock:
            if self._goal is not None and not self._goal.done.is_set():
                raise RuntimeError("a move is already in progress (call stop() first)")
            self._jog = None       # a goal move takes over from any manual jog
            self._goal = goal
            self._yaw_i = 0.0      # fresh heading-hold integrator for this move

        if not blocking:
            return goal
        return self.wait(goal)

    def wait(self, goal=None, timeout=None):
        """Block until the active (or given) move finishes; return its MoveResult."""
        with self._lock:
            goal = goal or self._goal
        if goal is None:
            return None
        goal.done.wait(timeout)
        return goal.result

    def jog(self, forward=0.0, right=0.0, duration=0.5):
        """Continuous manual velocity (hold-to-move), in m/s, dead-man guarded.

        Keeps driving at (forward, right) until either jog_stop() is called or
        ``duration`` seconds pass without a refreshing jog() call — so a UI that
        sends jog() every ~200 ms while a button is held gets press-and-hold
        motion that can never run away if the connection drops. Heading-hold
        keeps the yaw the rover had when the jog began. Refused (returns False)
        while a goal move is active.
        """
        if not self._started:
            raise RuntimeError("call start() before jog()")
        f = _clamp(float(forward), -self.cfg.MAX_LINEAR, self.cfg.MAX_LINEAR)
        r = _clamp(float(right), -self.cfg.MAX_LINEAR, self.cfg.MAX_LINEAR)
        pose = self._latest_pose()
        with self._lock:
            if self._goal is not None and not self._goal.done.is_set():
                return False
            if self._jog is None:
                # Fresh jog: snapshot heading AND position so the un-commanded
                # axis can be position-held against parasitic drift.
                self._yaw_i = 0.0
                yaw0 = pose["yaw"] if pose else 0.0
                rx0 = pose["rx"] if pose else 0.0
                rf0 = pose["rf"] if pose else 0.0
            else:
                yaw0 = self._jog["yaw0"]
                rx0 = self._jog["rx0"]
                rf0 = self._jog["rf0"]
            self._jog = {"f": f, "r": r, "yaw0": yaw0, "rx0": rx0, "rf0": rf0,
                         "until": time.monotonic() + max(0.1, float(duration))}
        return True

    def jog_stop(self):
        """End a manual jog; the control loop ramps the rover to a stop."""
        with self._lock:
            self._jog = None

    def is_busy(self):
        with self._lock:
            return ((self._goal is not None and not self._goal.done.is_set())
                    or self._jog is not None)

    def stop(self):
        """Cancel any active move/jog and command zero velocity immediately."""
        with self._lock:
            g = self._goal
            if g is not None and not g.done.is_set():
                self._finish(g, False, "cancelled")
            self._jog = None
            self._out = (0.0, 0.0, 0.0)
        self._publish_now((0.0, 0.0, 0.0))

    def get_pose(self):
        """Latest raw pose: dict with right/forward/yaw(rad)/confidence, or None.

        These are in the T265 world frame (not relative to a move origin) — handy
        for diagnostics. ``right`` = T265 +x, ``forward`` = T265 -z.
        """
        p = self._latest_pose()
        if p is None:
            return None
        return {"right": p["rx"], "forward": p["rf"], "yaw": p["yaw"],
                "confidence": p["confidence"], "mock": self._mock}

    @property
    def ros_connected(self):
        return self._ros_ok

    @property
    def using_mock(self):
        return self._mock

    def shutdown(self):
        """Stop the rover, tear down threads, and close the camera/ROS handles."""
        self._stop.set()
        try:
            self.stop()
        except Exception:
            pass
        for t in (self._ctrl_thread, self._pose_thread):
            if t is not None:
                t.join(timeout=2.0)
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.shutdown()
        return False

    # ------------------------------------------------------------- internals #
    def _open_t265(self):
        if rs is None:
            self._fallback_mock("pyrealsense2 not installed")
            return
        last = None
        for attempt in range(self.cfg.T265_OPEN_RETRIES + 1):
            try:
                pipe = rs.pipeline()
                cfg = rs.config()
                if self.cfg.T265_SERIAL:
                    cfg.enable_device(self.cfg.T265_SERIAL)
                cfg.enable_stream(rs.stream.pose)
                pipe.start(cfg)
                self._pipe = pipe
                self._mock = False
                return
            except Exception as exc:
                last = exc
                # Enumerated-but-won't-start: hardware-reset and retry before
                # giving up (and only then falling back to mock).
                if attempt < self.cfg.T265_OPEN_RETRIES:
                    print(f"[T265RoverService] open failed ({exc}); hardware-resetting T265 "
                          f"and retrying ({attempt + 1}/{self.cfg.T265_OPEN_RETRIES})...")
                    self._reset_t265()
        self._fallback_mock(f"T265 open failed after {self.cfg.T265_OPEN_RETRIES + 1} tries: {last}")

    def _reset_t265(self):
        """Power-cycle the T265 in firmware and wait for it to re-enumerate."""
        try:
            for d in rs.context().query_devices():
                if "T265" not in d.get_info(rs.camera_info.name):
                    continue
                if (not self.cfg.T265_SERIAL
                        or d.get_info(rs.camera_info.serial_number) == self.cfg.T265_SERIAL):
                    d.hardware_reset()
                    break
        except Exception:
            pass
        time.sleep(self.cfg.T265_RESET_WAIT)

    def _fallback_mock(self, why):
        if not self.allow_mock:
            raise RuntimeError(why)
        self._mock = True
        # Mock pose state, integrated from commanded velocity (yaw starts at 0,
        # held there, so body ~= world and we can integrate body velocity directly).
        self._mock_state = {"rx": 0.0, "rf": 0.0, "yaw": 0.0}
        # Test hook: an external yaw push (rad/s) and a one-shot yaw kick (rad)
        # injected into the mock so the heading-hold can be exercised off-hardware.
        self._mock_yaw_drift = 0.0
        self._mock_yaw_kick = 0.0
        print(f"[T265RoverService] {why}; running in MOCK mode (no real motion)")

    def _init_ros(self):
        if rospy is None or self._mock:
            self._ros_ok = False
            return
        try:
            try:
                import rosgraph
                if not rosgraph.is_master_online():
                    self._ros_ok = False
                    print("[T265RoverService] ROS master offline; commands won't move the rover")
                    return
            except Exception:
                pass
            rospy.init_node(self.cfg.ROS_NODE_NAME, anonymous=True, disable_signals=True)
            self._pub = rospy.Publisher(self.cfg.CMD_VEL_TOPIC, Twist, queue_size=5)
            self._ros_ok = True
        except Exception as exc:
            self._ros_ok = False
            print(f"[T265RoverService] ROS init failed: {exc}")

    # ---- pose acquisition ------------------------------------------------ #
    def _pose_loop(self):
        if self._mock:
            self._mock_pose_loop()
            return
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(2000)
                pf = frames.get_pose_frame()
                if not pf:
                    continue
                d = pf.get_pose_data()
                tr, rot = d.translation, d.rotation
                yaw = _quat_to_yaw(rot.x, rot.y, rot.z, rot.w)
                # T265: +x right, +y up, +z back => forward = -z.
                rx, rf, ry = self._filter_pose(tr.x, -tr.z, yaw, time.monotonic())
                self._set_pose(self._make_pose(rx, rf, ry, d.tracker_confidence))
            except Exception as exc:
                if not self._stop.is_set():
                    print(f"[T265RoverService] pose read error: {exc}")
                    time.sleep(0.2)

    def _mock_pose_loop(self):
        dt = 1.0 / max(self.cfg.POSE_HZ, 1)
        while not self._stop.is_set():
            with self._lock:
                fwd, right, yaw_rate = self._out
                s = self._mock_state
                # Commanded rotation + any injected external disturbance/kick.
                s["yaw"] += (yaw_rate + self._mock_yaw_drift) * dt + self._mock_yaw_kick
                self._mock_yaw_kick = 0.0
                # Integrate body velocity into world (small-angle: yaw held ~0).
                c, sn = math.cos(s["yaw"]), math.sin(s["yaw"])
                s["rx"] += (right * c - fwd * sn) * dt
                s["rf"] += (right * sn + fwd * c) * dt
                pose = self._make_pose(s["rx"], s["rf"], s["yaw"], 3)
            self._set_pose(pose)
            self._stop.wait(dt)

    def _filter_pose(self, rx, rf, yaw, now):
        """De-glitch the raw T265 pose: integrate frame-to-frame deltas but DROP
        any single delta that implies an impossible speed/yaw-rate (a repetitive-
        texture relocalisation teleport). Returns the reported (rx, rf, yaw).

        Transient spike -> both the out- and back-jump deltas are dropped, so the
        reported pose stays put. Persistent relocalisation step -> the step delta
        is dropped once, then normal small deltas resume from the new raw
        baseline, so the reported pose continues smoothly from where it was.
        """
        if self._filt is None:
            self._filt = [rx, rf, yaw]
            self._raw_last = [rx, rf, yaw]
            self._filt_last_t = now
            return rx, rf, yaw
        dt = max(now - self._filt_last_t, 1e-3)
        self._filt_last_t = now
        dpx = rx - self._raw_last[0]
        dpf = rf - self._raw_last[1]
        dyaw = _wrap(yaw - self._raw_last[2])
        self._raw_last = [rx, rf, yaw]      # always advance the raw baseline
        max_pos = self.cfg.POSE_JUMP_MAX_SPEED * dt + self.cfg.POSE_JUMP_BASE_M
        max_yaw = self.cfg.POSE_JUMP_MAX_YAW_RATE * dt + self.cfg.POSE_JUMP_BASE_RAD
        jumped = False
        if math.hypot(dpx, dpf) <= max_pos:
            self._filt[0] += dpx
            self._filt[1] += dpf
        else:
            jumped = True
        if abs(dyaw) <= max_yaw:
            self._filt[2] = _wrap(self._filt[2] + dyaw)
        else:
            jumped = True
        if jumped and self._cmd_still and self.cfg.POSE_JUMP_ACCEPT_WHEN_STILL:
            # Commanded stationary: the rover cannot physically have moved, so
            # this step is the T265 relocalising (or a tracking fault, which the
            # confidence gate catches). ACCEPT it — discarding it would re-apply
            # the very drift the device just removed. Snap the filtered pose to
            # the raw one rather than integrating the delta.
            self._filt = [rx, rf, yaw]
            self._reloc_accepted += 1
            if now - self._last_jump_log > 2.0:
                self._last_jump_log = now
                print("[T265RoverService] accepted relocalisation #%d while still "
                      "(dpos=%.2fm dyaw=%.0fdeg)"
                      % (self._reloc_accepted, math.hypot(dpx, dpf),
                         math.degrees(abs(dyaw))))
            return self._filt[0], self._filt[1], self._filt[2]
        if jumped:
            self._pose_jumps += 1
            if now - self._last_jump_log > 2.0:
                self._last_jump_log = now
                print("[T265RoverService] rejected pose jump #%d "
                      "(dpos=%.2fm dyaw=%.0fdeg in %.0fms) — T265 relocalisation?"
                      % (self._pose_jumps, math.hypot(dpx, dpf),
                         math.degrees(dyaw), dt * 1000))
        return self._filt[0], self._filt[1], self._filt[2]

    @property
    def reloc_accepted(self):
        """Count of T265 relocalisation steps ACCEPTED while commanded still.
        A healthy non-zero value means the device is correcting its own drift
        and the correction is now being kept rather than discarded."""
        return self._reloc_accepted

    @property
    def pose_jumps(self):
        """Number of T265 teleport frames rejected so far (diagnostics)."""
        return self._pose_jumps

    def _make_pose(self, rx, rf, yaw, confidence):
        """Build a pose dict in ROVER-CENTRE coordinates.

        The T265 reports ITS OWN position. Mounted off the turn centre it traces
        an arc when the rover rotates in place and reports that as translation,
        so the believed rover position (and every target projected from it)
        swings on each heading correction. Subtract the lever arm, rotated into
        the world frame by the current heading, and pure rotation leaves the
        reported position unchanged — which is what physically happens.
        """
        ox, oz = self.cfg.SENSOR_OFFSET_RIGHT, self.cfg.SENSOR_OFFSET_FWD
        if ox or oz:
            c, sn = math.cos(yaw), math.sin(yaw)
            # same (right, forward) rotation convention the navigator uses
            rx -= ox * c + oz * sn
            rf -= -ox * sn + oz * c
        return {"_now": time.monotonic(), "rx": rx, "rf": rf,
                "yaw": yaw, "confidence": confidence}

    def _set_pose(self, pose):
        with self._lock:
            self._pose = pose
        self._pose_event.set()

    def _latest_pose(self):
        with self._lock:
            return self._pose

    # ---- control --------------------------------------------------------- #
    def _control_loop(self):
        period = 1.0 / max(self.cfg.CONTROL_HZ, 1)
        while not self._stop.is_set():
            with self._lock:
                out = self._tick(time.monotonic())
                self._out = out
                # "Commanded stationary" = we are publishing (near) zero body
                # velocity. _filter_pose uses this to decide whether a large
                # pose step can possibly be real motion (it cannot, while still)
                # or must be a T265 relocalisation worth accepting.
                self._cmd_still = (abs(out[0]) < self.cfg.POSE_STILL_EPS
                                   and abs(out[1]) < self.cfg.POSE_STILL_EPS
                                   and abs(out[2]) < self.cfg.POSE_STILL_EPS)
            self._publish_now(out)
            self._stop.wait(period)
        self._publish_now((0.0, 0.0, 0.0))   # final safety stop

    def _tick(self, now):
        """One control step. Must be called with ``self._lock`` held. Returns the
        ramped (forward, right, yaw) body velocity to publish."""
        goal = self._goal
        pose = self._pose
        if goal is None or goal.done.is_set():
            return self._jog_tick(now, pose)
        if pose is None:
            return self._ramp(self._out, (0.0, 0.0, 0.0))

        # Displacement since the move began, in the start-heading body frame.
        right, forward = self._displacement(pose, goal)
        err_r = goal.right - right
        err_f = goal.forward - forward
        dist = math.hypot(err_r, err_f)

        # Stall watchdog: fail fast if the remaining distance-to-goal hasn't
        # meaningfully improved in a while, whether that's because the rover
        # isn't actually moving despite being commanded to (stuck, e-stop,
        # motor fault) or because tracking is lost and never recovers. This
        # runs well ahead of the worst-case deadline below so a dead move
        # clears quickly instead of blocking new commands behind it.
        # NOTE: once position is inside tolerance the rover deliberately stops
        # translating and rotates in place to settle heading, so distance stops
        # improving BY DESIGN. best_dist is the minimum ever seen, so that phase
        # can never improve it and would otherwise burn the whole stall budget —
        # a leg that arrived correctly then failed as "stalled" while doing
        # exactly what it was told. Hold the timer while position is good.
        if dist <= self.cfg.POS_TOL:
            goal.best_dist_t = now
        if goal.best_dist is None or dist <= goal.best_dist - self.cfg.STALL_MIN_PROGRESS_M:
            goal.best_dist = dist
            goal.best_dist_t = now
        elif now - goal.best_dist_t >= self.cfg.STALL_TIMEOUT:
            self._finish(goal, False, "stalled", right, forward)
            return self._ramp(self._out, (0.0, 0.0, 0.0))

        # Tracking lost -> hold still (the deadline still applies as a backstop).
        if not self._mock and pose["confidence"] <= self.cfg.LOST_CONF:
            if now >= goal.deadline:
                self._finish(goal, False, "tracking_lost", right, forward)
            return self._ramp(self._out, (0.0, 0.0, 0.0))

        # Heading error vs the HOLD heading, corrected every tick so the rover
        # is driven toward (and kept at) the desired heading WHILE translating —
        # not just held at the start value. With an absolute hold_yaw this
        # actively removes accumulated yaw drift during the move.
        yaw_err = _wrap(goal.hold_yaw - pose["yaw"])
        wz = self._yaw_cmd(yaw_err, 1.0 / max(self.cfg.CONTROL_HZ, 1))

        # Track when position first entered tolerance, to BOUND the end-stage
        # in-place yaw correction (below).
        pos_ok = dist <= self.cfg.POS_TOL
        yaw_ok = abs(yaw_err) <= self.cfg.YAW_TOL
        if pos_ok and goal.pos_since is None:
            goal.pos_since = now
        elif not pos_ok:
            goal.pos_since = None

        # Arrival requires BOTH position (POS_TOL) and heading (YAW_TOL) settled,
        # held for SETTLE_TIME — a move normally won't complete while rotated.
        if pos_ok and yaw_ok:
            if goal.in_tol_since is None:
                goal.in_tol_since = now
            elif now - goal.in_tol_since >= self.cfg.SETTLE_TIME:
                self._finish(goal, True, "arrived", right, forward)
            return self._ramp(self._out, (0.0, 0.0, wz))
        goal.in_tol_since = None

        if now >= goal.deadline:
            self._finish(goal, False, "timeout", right, forward)
            return self._ramp(self._out, (0.0, 0.0, 0.0))

        # Position reached but heading not yet settled: rotate in place to fix
        # it — but only up to ARRIVE_SETTLE_TIMEOUT, then accept arrival anyway
        # (position is good; never spin indefinitely on the last fraction of a
        # degree). A base that just can't hit YAW_TOL still completes the path.
        if pos_ok:
            if (self.cfg.ARRIVE_SETTLE_TIMEOUT <= 0.0
                    or now - goal.pos_since >= self.cfg.ARRIVE_SETTLE_TIMEOUT):
                self._finish(goal, True, "arrived", right, forward)
                return self._ramp(self._out, (0.0, 0.0, 0.0))
            # Rotate to settle heading, but KEEP nudging position at the same
            # time. Commanding pure rotation here used to create a mode flip:
            # any drift during the spin pushed position back outside tolerance,
            # the controller switched to translating, which disturbed heading,
            # and it oscillated between the two. Use the bare proportional term
            # with NO stiction floor or boost, so this is a gentle trim (a 10 mm
            # error asks for 0.015 m/s) rather than a fresh full-speed run.
            trim = _clamp(self.cfg.LINEAR_GAIN * dist, 0.0, self.cfg.MAX_LINEAR)
            if dist > 1e-6:
                tf, tr = (err_f / dist) * trim, (err_r / dist) * trim
            else:
                tf = tr = 0.0
            return self._ramp(self._out, (tf, tr, wz))

        # Translate straight toward the goal: one speed along the error vector
        # (so a diagonal and a straight move behave identically). Speed is
        # proportional to remaining distance — easing in near the goal — but
        # floored at MIN_LINEAR so the final stretch runs at an executable speed
        # rather than an asymptotic crawl, and capped at MAX_LINEAR.
        # Stiction integral: pure P has no authority to break static friction —
        # if the floor speed doesn't move the base, it never rises. Accrue time
        # spent commanded-but-not-closing, and add a bounded boost. Integrating
        # TIME-STUCK rather than distance-error means it contributes nothing
        # during normal cruise, so it cannot wind up and overshoot at the end.
        dt_i = max(now - goal.prev_t, 1e-3)
        closing = ((goal.prev_dist - dist) / dt_i) if goal.prev_dist is not None else 0.0
        goal.prev_dist, goal.prev_t = dist, now
        if closing < self.cfg.LIN_STICTION_EPS:
            goal.lin_i += dt_i                                   # stuck: build
        else:
            goal.lin_i = max(0.0, goal.lin_i - self.cfg.LIN_I_BLEED * dt_i)   # moving: bleed
        boost = min(self.cfg.LIN_IGAIN * goal.lin_i, self.cfg.LIN_I_MAX)
        speed = _clamp(self.cfg.LINEAR_GAIN * dist + boost,
                       self.cfg.MIN_LINEAR, self.cfg.MAX_LINEAR)
        vr = speed * err_r / dist
        vf = speed * err_f / dist
        # Cross-axis stiffening: the minor axis only gets its error-share of
        # the speed above, which under-corrects parasitic drift (a strafing
        # mecanum base creeps forward/back). Give the minor axis its own full
        # P correction, capped at MAX_CROSS, so drift is pulled out promptly.
        if abs(err_r) >= abs(err_f):
            vf = _clamp(self.cfg.LINEAR_GAIN * err_f, -self.cfg.MAX_CROSS, self.cfg.MAX_CROSS)
        else:
            vr = _clamp(self.cfg.LINEAR_GAIN * err_r, -self.cfg.MAX_CROSS, self.cfg.MAX_CROSS)
        return self._ramp(self._out, (vf, vr, wz))

    def _jog_tick(self, now, pose):
        """Control step while no goal move is active: drive any live jog.
        Must be called with ``self._lock`` held. Returns the ramped output."""
        jog = self._jog
        if jog is None or pose is None:
            return self._ramp(self._out, (0.0, 0.0, 0.0))
        # Dead-man expired or tracking lost -> stop.
        if now >= jog["until"] or (not self._mock and pose["confidence"] <= self.cfg.LOST_CONF):
            self._jog = None
            return self._ramp(self._out, (0.0, 0.0, 0.0))
        yaw_err = _wrap(jog["yaw0"] - pose["yaw"])
        wz = self._yaw_cmd(yaw_err, 1.0 / max(self.cfg.CONTROL_HZ, 1))
        # Position-hold the axis the user is NOT driving: displacement since
        # the jog began (start-heading frame), corrected at full P gain capped
        # at MAX_CROSS — a strafe stays on its line, a forward run stays
        # centered, despite mecanum parasitic drift.
        dxr = pose["rx"] - jog["rx0"]
        dxf = pose["rf"] - jog["rf0"]
        cy, sy = math.cos(jog["yaw0"]), math.sin(jog["yaw0"])
        right = dxr * cy + dxf * sy
        fwd = -dxr * sy + dxf * cy
        vr, vf = jog["r"], jog["f"]
        if vr == 0.0:
            vr = _clamp(-self.cfg.LINEAR_GAIN * right, -self.cfg.MAX_CROSS, self.cfg.MAX_CROSS)
        if vf == 0.0:
            vf = _clamp(-self.cfg.LINEAR_GAIN * fwd, -self.cfg.MAX_CROSS, self.cfg.MAX_CROSS)
        return self._ramp(self._out, (vf, vr, wz))

    def _displacement(self, pose, goal):
        """Live pose re-expressed in the goal's start-heading body frame.

        Returns ``(right, forward)`` travelled since the move began.
        """
        dxr = pose["rx"] - goal.rx0      # world right offset
        dxf = pose["rf"] - goal.rf0      # world forward offset
        c, s = math.cos(goal.yaw0), math.sin(goal.yaw0)
        right = dxr * c + dxf * s
        forward = -dxr * s + dxf * c
        return right, forward

    def _yaw_cmd(self, yaw_err, dt):
        """PI yaw rate to drive the heading error to zero (CCW+).

        The integral nulls a constant rotational bias (a base that creeps while
        strafing) so heading returns fully, not just to the edge of the P
        deadband. Clamped for anti-windup and reset at the start of each move.

        Inside the deadband the command is ZEROED OUTRIGHT, and the accumulator
        is bled away rather than merely frozen. Previously only the P term was
        dropped there while the integral kept being applied, so once the error
        fell inside the deadband the rover carried on rotating on stored
        integral alone with nothing to unwind it: it drove through centre, the
        error changed sign, the integral wound the other way, and the base sat
        oscillating left-right in place. A saturated integral commands ~20 deg/s,
        which crosses the whole deadband in well under a tenth of a second, so
        the hunting was fast and visible. Zeroing the command inside the band
        makes "close enough" actually mean stop.
        """
        if abs(yaw_err) >= self.cfg.YAW_DEADBAND:
            self._yaw_i += yaw_err * dt
            i_lim = self.cfg.YAW_I_MAX / max(self.cfg.YAW_IGAIN, 1e-6)
            self._yaw_i = _clamp(self._yaw_i, -i_lim, i_lim)
            cmd = self.cfg.YAW_GAIN * yaw_err + self.cfg.YAW_IGAIN * self._yaw_i
        else:
            # Settled: command nothing, and bleed the accumulator so a stale
            # integral cannot kick the base once the error next leaves the band.
            self._yaw_i -= self._yaw_i * min(1.0, self.cfg.YAW_I_BLEED * dt)
            cmd = 0.0
        return _clamp(cmd, -self.cfg.MAX_YAW, self.cfg.MAX_YAW)

    def _ramp(self, cur, target):
        f = _ramp_axis(cur[0], target[0], self.cfg.RAMP_STEP)
        r = _ramp_axis(cur[1], target[1], self.cfg.RAMP_STEP)
        w = _ramp_axis(cur[2], target[2], self.cfg.RAMP_STEP_YAW)
        return (f, r, w)

    def _finish(self, goal, ok, reason, right=None, forward=None):
        """Record a move's outcome and release any blocking caller. Lock held."""
        if goal.done.is_set():
            return
        pose = self._pose
        if right is None:
            right, forward = (self._displacement(pose, goal) if pose else (0.0, 0.0))
        err = math.hypot(goal.right - right, goal.forward - forward)
        yaw_err = _wrap(goal.yaw0 - pose["yaw"]) if pose else 0.0
        goal.result = MoveResult(
            ok=ok, reason=reason, axis=goal.axis,
            target=(goal.right, goal.forward), reached=(right, forward),
            error=err, yaw_error=yaw_err, elapsed=time.monotonic() - goal.t_start)
        goal.done.set()
        self._out = (0.0, 0.0, 0.0)
        self._goal = None

    def _publish_now(self, out):
        if not self._ros_ok or self._pub is None:
            return
        fwd, right, wz = out
        tw = Twist()
        tw.linear.x = self.cfg.X_SIGN * fwd
        tw.linear.y = -self.cfg.Y_SIGN * right     # ROS body +y is left; right = -y
        tw.linear.z = 0.0
        tw.angular.x = tw.angular.y = 0.0
        tw.angular.z = self.cfg.YAW_SIGN * wz
        try:
            self._pub.publish(tw)
        except Exception:
            self._ros_ok = False


# ------------------------------------------------------------------ helpers #
def _to_meters(value, units):
    units = units.lower()
    if units in ("m", "meter", "meters"):
        return float(value)
    if units in ("mm", "millimeter", "millimeters"):
        return float(value) / 1000.0
    if units in ("cm", "centimeter", "centimeters"):
        return float(value) / 100.0
    raise ValueError(f"unknown units {units!r} (use m/cm/mm)")


def _quat_to_yaw(x, y, z, w):
    """Yaw (rotation about the vertical axis) from a quaternion, in radians.

    The T265 reports pose in a right-handed system with -Z forward and +Y up, so
    heading is the rotation about the Y (up) axis. Matches the rover_ui derivation.
    """
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ramp_axis(current, target, step):
    if target > current:
        return min(target, current + step)
    if target < current:
        return max(target, current - step)
    return target


def _r2(pair):
    return (round(pair[0], 3), round(pair[1], 3))
