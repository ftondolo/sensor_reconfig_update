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
    rectilinear_mm.plan_rectilinear_path_ex — which maximises the route's
    MINIMUM distance to any obstacle rather than merely clearing them by the
    configured `clearance`, and reports the margin achieved (surfaced as
    state()["plan_clearance_mm"]) — and executes it one axis-aligned leg
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
from . import tag_fusion

# Repo root holds rectilinear_mm.py and the purely_control package.
if config.DEMO_ROOT not in sys.path:
    sys.path.insert(0, config.DEMO_ROOT)

from rectilinear_mm import (plan_rectilinear_path_ex,      # noqa: E402
                            obstacle_boxes, describe_obstacle_schema)
from purely_control import T265RoverService       # noqa: E402


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _wrap_deg(d):
    """Wrap a DEGREE difference to (-180, 180]. Used when accumulating how far
    the rover has turned, so a heading crossing +/-180 doesn't count as ~360."""
    return math.degrees(math.atan2(math.sin(math.radians(d)),
                                   math.cos(math.radians(d))))


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
        self._obstacles = []         # canonical collision boxes; set by _load_files
        self._bounds_slack = 0.0     # set properly by _load_files from rover_start;
                                     # defined here so the bounds e-stop is never
                                     # reading an undefined attribute if load fails.
        self._load_files(anchor_to_start=True)

        # Rover-motion service (sole T265 + cmd_vel owner in this process).
        self.rover = T265RoverService(
            allow_mock=config.ALLOW_MOCK,
            CMD_VEL_TOPIC=config.NAV_CMD_VEL_TOPIC,
            MAX_LINEAR=float(config.NAV_MAX_LINEAR),
            POS_TOL=float(config.NAV_POS_TOL),
            RAMP_STEP=float(config.NAV_RAMP_STEP),     # snappier start (clears base stiction)
            MIN_LINEAR=float(config.NAV_MIN_LINEAR),   # floor above the base's stiction speed
            STALL_TIMEOUT=float(config.NAV_STALL_TIMEOUT_S),  # fail fast on no-progress moves
            ARRIVE_SETTLE_TIMEOUT=float(config.NAV_ARRIVE_SETTLE_TIMEOUT_S),
            YAW_DEADBAND=float(config.NAV_YAW_DEADBAND),
            MOVE_TIMEOUT_BASE=float(config.NAV_MOVE_TIMEOUT_BASE_S),
            LIN_IGAIN=float(config.NAV_LIN_IGAIN),
            LIN_I_MAX=float(config.NAV_LIN_I_MAX),
            LIN_STICTION_EPS=float(config.NAV_LIN_STICTION_EPS),
            POSE_JUMP_ACCEPT_WHEN_STILL=bool(config.NAV_ACCEPT_RELOC_WHEN_STILL),
            # metres, rover body frame, from the TURN CENTRE
            SENSOR_OFFSET_FWD=float(config.NAV_T265_OFFSET_FWD_MM) / 1000.0,
            SENSOR_OFFSET_RIGHT=float(config.NAV_T265_OFFSET_RIGHT_MM) / 1000.0,
        )
        self._anchor_pose = None     # T265 pose snapshot mapped to _anchor_map

        # Navigation state (everything the UI shows lives here).
        self._status = "idle"        # idle|planning|moving|arrived|no_path|blocked|cancelled|error
        self._message = ""
        self._plan_clearance = None  # mm: min body-to-obstacle distance the CURRENT
                                     # route achieves (the planner maximises it).
                                     # None until the first successful plan.
        # ---- drift accounting (see config.NAV_DRIFT_*) ----
        self._drift_dist_mm = 0.0    # path length driven since the pose was last known-good
        self._drift_turn_deg = 0.0   # |yaw| turned since then
        self._drift_last_pose = None # pose sample the two above are integrated from
        self._zupt_mm_s = None       # measured drift rate while commanded stationary (mm/s)
        self._zupt_samples = 0
        self._leg_log = []           # recent [{leg, cmd_mm, got_mm, err_mm, ok, reason}]
        # The speed the OPERATOR asked for. cfg.MAX_LINEAR is also written by
        # the confidence gate (which throttles while tracking is poor), so the
        # live value cannot be used to remember the operator's intent — without
        # this, restoring after a throttle would silently reset the slider to
        # the startup default on every leg.
        self._speed_setpoint = float(config.NAV_MAX_LINEAR)
        # Operator-settable approach distance. compute_goal reads THIS, not the
        # config constant, so the UI field takes effect on the next goal
        # computation — immediately in AUTO/FOLLOW, which recompute every cycle.
        self._standoff_mm = float(config.NAV_STANDOFF_MM)
        # ---- ArUco tag fixes ----
        self._tag_last = None        # last APPLIED fix summary (for state()/UI)
        self._tag_run = []           # consecutive candidate fixes pending confirmation
        self._tag_applied = 0
        self._tag_rejected = 0
        self._tag_reject_reason = None   # why the last fix was refused (shown in the UI)
        self._tag_live = None            # most recent SOLVE, applied or not
        self._tag_idle = None            # why no solve is being produced at all
        self._tag_fix_t = 0.0            # last idle-loop fix attempt
        self._tag_yaw_hist = []          # recent heading residuals, for the UI
        # Rolling window of recent absolute position measurements. Trust moved
        # here from the individual frame: see _consume_fix and nav.tag_fusion.
        self._tag_win = tag_fusion.FixWindow(
            window_s=float(config.TAGS_WINDOW_S),
            max_n=int(config.TAGS_WINDOW_N),
            trim_mm=float(config.TAGS_TRIM_MM))
        self._tag_pending = None         # why a correction is being held back
        self._tag_reason = None          # localiser's machine-readable verdict
        self._degen_run = 0              # consecutive degenerate-geometry frames
        # ---- micro-parallax ----
        self._parallax_busy = False      # a manoeuvre owns the rover right now
        self._parallax_t = 0.0           # last attempt, for the cooldown
        self._tag_parallax = None        # last attempt's outcome, for the UI
        self._target = None          # {x, z, t, range_m, az_deg, source}
        self._target_win = []        # [(t, x, z)] raw projections for the median window
        self._track = None           # {x, z, t, snr} continuity-gated target (anti-teleport)
        self._track_reject_since = None  # when a far-jump detection started being held
        self._pending_run = []       # [{x,z,snr,t}] accumulating confirmations for a
                                      # fresh lock / relock (ghost-detection guard)
        self._last_tel_seq = None    # last detector telemetry seq consumed (de-dupes
                                      # repeated polls of the same frame in the run count)
        # UI-facing summary of the above (see project_detection): whether a
        # new location is still being accumulated/confirmed, or the track is
        # currently confirmed and trusted. "none" until the first detection.
        self._accum = {"phase": "none", "n": 0, "need": int(config.NAV_TARGET_CONFIRM_N)}
        self._goal = None            # {x, z, adjusted: bool}
        self._path = None            # [[x, z], ...] map waypoints of the active plan
        self._leg = 0                # index of the waypoint being driven to
        self._nav_thread = None
        self._cancel = threading.Event()
        # Operator override (default off): when True, the rover's OWN current
        # position being inside an obstacle/clearance zone never blocks
        # starting a plan or continuing a move. Obstacle avoidance everywhere
        # else (routing, the standoff-goal search) is unaffected.
        self._ignore_obstacles = False

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
        # Canonical COLLISION boxes: the panel wall plus its stabiliser-foot
        # pad, resolved from whichever schema map.json/config.json use (see
        # rectilinear_mm.obstacle_boxes). Everything below works on these, so
        # the schema is handled in exactly one place.
        obstacles = obstacle_boxes(m, pc, "collision")
        print("[nav] %s" % describe_obstacle_schema(m, pc))
        # Inflated obstacle boxes (x1, z1, x2, z2). A rover CENTER inside one of
        # these means the rover footprint is within `clearance` mm of the raw
        # obstacle -- i.e. it has entered the SAFETY BUFFER. It does NOT mean the
        # footprint overlaps the obstacle: that would need inflation by the car
        # half-extents alone, without the clearance term. Keeping the two ideas
        # distinct matters, because entering the buffer is a margin violation to
        # be corrected, while overlapping the raw obstacle is a real collision.
        mx = car_w / 2.0 + clearance
        mz = car_l / 2.0 + clearance
        inflated = [(o["x"] - mx, o["z"] - mz,
                     o["x"] + o["w"] + mx, o["z"] + o["h"] + mz)
                    for o in obstacles]
        # Detection no-target zones: the raw obstacle rectangle (map.json) grown
        # by the clear margin (config.json `clearance`). A detection that
        # projects into one of these is treated as invalid — a person can never
        # be standing inside an obstacle or its clear keep-out, so the rover
        # must not generate a target there. This is the same geometry the UI
        # already draws as the red obstacle halo.
        exclusion = [(o["x"] - clearance, o["z"] - clearance,
                      o["x"] + o["w"] + clearance, o["z"] + o["h"] + clearance)
                     for o in obstacles]
        start = m.get("rover_start", {})
        # Default start: bottom-edge middle, rear bumper flush with the map's
        # bottom edge (center is half a car length up from it).
        map_start = (float(start.get("x", map_w / 2)),
                     float(start.get("z", map_d - car_l / 2)))
        # How far the CONFIGURED start pose itself protrudes past the arena
        # edge. The planner grants a route the same allowance (see
        # rectilinear_mm._plan), so the bounds e-stop must too, or a rover
        # legitimately parked half-outside would stop the instant it moved.
        # A start placed properly inside gives 0 here, making both strict.
        bounds_slack = max(0.0,
                           car_w / 2.0 - map_start[0],
                           (map_start[0] + car_w / 2.0) - map_w,
                           car_l / 2.0 - map_start[1],
                           (map_start[1] + car_l / 2.0) - map_d)
        with self._lock:
            self.map, self.plan_cfg = m, pc
            self.map_w, self.map_d = map_w, map_d
            self.car_w, self.car_l, self.clearance = car_w, car_l, clearance
            self._inflated = inflated
            self._exclusion = exclusion
            self._obstacles = obstacles      # canonical collision boxes
            self._map_start = map_start
            self._bounds_slack = bounds_slack
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

    def set_start_cell(self, x, z):
        """Change the configured start cell (map.json `rover_start`) to map
        (x, z) and PERSIST it back to map.json. Refused mid-navigation.

        This only moves the home/start cell (the Return-to-start goal and the
        default re-anchor point); it does NOT move the rover or re-anchor the
        live pose — use the re-anchor button for that."""
        try:
            x, z = float(x), float(z)
        except (TypeError, ValueError):
            return False, "x and z must be numbers"
        with self._lock:
            if self._status in ("planning", "moving"):
                return False, "cannot change the start cell while navigating (cancel first)"
        # Center must be on the map and clear of any obstacle keep-out. (The
        # footprint may overhang the map edge — the default start sits with the
        # rear bumper flush against the bottom edge — so don't require the whole
        # footprint inside the map, only the center on-map.)
        if not (0.0 <= x <= self.map_w and 0.0 <= z <= self.map_d):
            return False, "start cell is outside the map"
        for (x1, z1, x2, z2) in self._inflated:
            if x1 < x < x2 and z1 < z < z2:
                return False, "start cell overlaps an obstacle keep-out"
        try:
            m = self._load_json(config.MAP_FILE)
            m["rover_start"] = {"x": round(x), "z": round(z)}
            with open(config.MAP_FILE, "w", encoding="utf-8") as fh:
                json.dump(m, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            return False, f"failed to write map.json: {exc}"
        with self._lock:
            self.map = m
            self._map_start = (float(m["rover_start"]["x"]), float(m["rover_start"]["z"]))
        return True, f"start cell set to ({round(x)}, {round(z)}) mm"

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
            # The pose is known-good again: everything the drift margin was
            # accumulating for has just been corrected, so start over.
            self._drift_dist_mm = 0.0
            self._drift_turn_deg = 0.0
            self._drift_last_pose = None
        return True

    @staticmethod
    def _camera_map_xz(p):
        """Map position of the COLOUR CAMERA, given a rover-centre pose.

        The rover pose is already lever-arm corrected for the T265 (see
        T265RoverService._make_pose); this walks out from the turn centre to
        where the D435 actually is, so a target's projected position no longer
        swings as the rover turns. Offsets are body-frame mm from the turn
        centre; both default to 0, which reproduces the old behaviour exactly
        until they are measured."""
        ofwd = float(config.NAV_CAM_OFFSET_FWD_MM)
        oright = float(config.NAV_CAM_OFFSET_RIGHT_MM)
        if not ofwd and not oright:
            return p["x"], p["z"]
        yaw = math.radians(p["yaw_deg"])
        c, sn = math.cos(yaw), math.sin(yaw)
        # +right is +x at yaw 0; +forward is -z at yaw 0 (map z grows backward)
        dx = oright * c + ofwd * sn
        dz = -(-oright * sn + ofwd * c)
        return p["x"] + dx, p["z"] + dz

    # ---- drift accounting ------------------------------------------------ #
    def _drift_margin_mm(self):
        """Extra planning margin (mm) earned by odometry error accumulated since
        the pose was last known-good. Grows with distance driven and with how
        much the rover has turned (heading error compounds hardest), capped by
        NAV_DRIFT_MARGIN_MAX_MM so a long session can't make the arena
        unreachable. Returns 0.0 when the mechanism is disabled."""
        cap = float(config.NAV_DRIFT_MARGIN_MAX_MM)
        if cap <= 0.0:
            return 0.0
        with self._lock:
            d, t = self._drift_dist_mm, self._drift_turn_deg
        m = (d / 1000.0) * float(config.NAV_DRIFT_PER_M_MM) \
            + t * float(config.NAV_DRIFT_PER_TURN_MM)
        return min(m, cap)

    def apply_tag_fix(self):
        """Consume an ArUco fix from the detector and correct the map anchor.

        This is the only mechanism here that BOUNDS drift: the pose is measured
        against surveyed markers rather than integrated from motion. Position
        only by default — a bad yaw fix rotates the entire map frame, which is
        far more damaging than a bad position fix — though the heading residual
        is always reported so it can be watched before being trusted.

        Applied ONLY while the rover is commanded stationary (motion blur ruins
        corner precision, and a correction mid-leg moves the goal under the
        planner), and eased in via the anchor so the control loop never sees a
        discontinuity. Deliberately NOT reset_pose(), which re-snapshots
        _anchor_pose and would zero the map yaw.

        Trust is no longer carried by the single frame. Frames are graded into
        two tiers — one that clears the original thresholds and may act alone,
        and a looser one that may only contribute to a CONSENSUS of several
        recent observations that agree with each other (see nav.tag_fusion).
        That is what lets the per-frame gates open up without the risk opening
        up with them: a bad frame slipping past a looser gate is diluted by its
        neighbours rather than applied outright.
        """
        if not config.TAGS_ENABLED or self._detector is None:
            return None
        if self._parallax_busy:
            # A parallax run owns the rover and the fix window right now; the
            # monitor loop must not clear the window out from under it.
            return None
        if not getattr(self.rover, "_cmd_still", True):
            # Moving: don't correct, and drop the window. An observation of
            # where the rover WAS is not an observation of where it is, and
            # keeping it would let a pre-move frame corroborate a post-move one.
            self._tag_win.clear("rover moving")
            return None
        tel, _seq = self._detector.telemetry.get()
        fix = (tel or {}).get("tag_fix")
        if not fix:
            with self._lock:
                self._tag_live = None
                self._tag_idle = (tel or {}).get("tag_idle") or "no tag data"
            self._note_tag_reason((tel or {}).get("tag_reason"))
            return None
        with self._lock:
            # Live solve, whether or not it ends up applied — this is what makes
            # a bad map.json entry visible instead of silently rejected.
            self._tag_idle = None
            self._tag_live = {"n_tags": fix.get("n_tags"), "ids": fix.get("ids"),
                              "rms_px": round(float(fix.get("rms_px", 0.0)), 2),
                              "spread_mm": fix.get("spread_mm"),
                              # Angular baseline of this observation: the number
                              # that actually decides whether the geometry is
                              # usable, now that the gate reasons in those terms.
                              "range_mm": fix.get("range_mm"),
                              "spread_ratio": fix.get("spread_ratio"),
                              "audit": fix.get("audit")}
        self._note_tag_reason(None)
        if time.time() - float(fix.get("t", 0.0)) > float(config.TAGS_FIX_MAX_AGE_S):
            return None
        return self._consume_fix(fix)

    # ------------------------------------------------------------ fix intake
    def _fix_tier(self, fix, yaw_err):
        """Grade a solve: "strict" (may act alone), "loose" (consensus only),
        or None with the reason it is refused outright.

        The strict thresholds are exactly the ones that were in force before;
        the loose ones are those scaled by TAGS_LOOSE_*_SCALE. Widening a gate
        is only safe because a loose frame cannot move the anchor by itself.
        """
        n_tags = int(fix.get("n_tags", 1) or 1)
        if n_tags < int(config.TAGS_MIN_TAGS_FOR_FIX):
            return None, ("only %d tag(s); need %d (a single tag cannot "
                          "self-check)" % (n_tags, config.TAGS_MIN_TAGS_FOR_FIX))
        # RMS allowance grows with tag count: more tags impose more mutual
        # constraints, so a legitimately-correct solve still shows a larger
        # residual than a single tag ever does.
        rms_lim = (float(config.TAGS_MAX_RMS_PX)
                   + float(config.TAGS_RMS_PER_TAG_PX) * max(0, n_tags - 1))
        rms = float(fix.get("rms_px", 1e9))
        yaw_lim = float(config.TAGS_MAX_YAW_ERR_DEG)
        loose = bool(config.TAGS_TEMPORAL_ENABLED)
        rms_loose = rms_lim * float(config.TAGS_LOOSE_RMS_SCALE)
        yaw_loose = yaw_lim * float(config.TAGS_LOOSE_YAW_SCALE)
        if rms > (rms_loose if loose else rms_lim):
            return None, ("rms %.1fpx > %.1f limit (%d tags) — the tag offsets "
                          "in map.json disagree with what the camera sees"
                          % (rms, (rms_loose if loose else rms_lim), n_tags))
        if yaw_lim > 0.0 and abs(yaw_err) > (yaw_loose if loose else yaw_lim):
            # The rover never turns deliberately, so a big tag-derived heading
            # error means the MAP is wrong, not the rover — and the position
            # half of the same solve is wrong by roughly 40 mm per degree.
            return None, ("yaw %.1f deg > %.1f limit — map offsets likely wrong"
                          % (yaw_err, (yaw_loose if loose else yaw_lim)))
        if rms <= rms_lim and (yaw_lim <= 0.0 or abs(yaw_err) <= yaw_lim):
            return "strict", None
        return "loose", ("rms %.1fpx / yaw %.1f deg clear only the loose tier — "
                         "consensus required" % (rms, yaw_err))

    def _consume_fix(self, fix, pose_ref=None, source="tag"):
        """Gate, corroborate and ease in one solve. Returns the applied summary
        or None.

        ``pose_ref`` is the pose the solve describes, for a fix that was taken
        somewhere other than where the rover is standing now (micro-parallax
        takes its first view, then moves). The anchor correction is a shift of
        the whole map frame, so a residual measured at an earlier pose stays
        valid however far the rover has driven since.
        """
        p = pose_ref or self.pose()
        if p is None:
            return None
        # The solve returns the CAMERA's map position; walk back along the
        # camera lever arm to the rover centre, which is what the anchor tracks.
        cx, cz = self._camera_map_xz(p)
        rx_fix = float(fix["x"]) - (cx - p["x"])
        rz_fix = float(fix["z"]) - (cz - p["z"])
        yaw_err = _wrap_deg(float(fix.get("yaw_deg", p["yaw_deg"])) - p["yaw_deg"])
        with self._lock:
            self._tag_yaw_hist.append(yaw_err)
            del self._tag_yaw_hist[:-40]
        tier, why = self._fix_tier(fix, yaw_err)
        if tier is None:
            self._tag_rejected += 1
            self._tag_reject_reason = why
            return None

        use_window = bool(config.TAGS_TEMPORAL_ENABLED)
        if use_window:
            # One frame must not satisfy an N-frame consensus by being polled N
            # times: the detector publishes at TAGS_DETECT_HZ while this is
            # called from two places at a higher rate, so the same solve comes
            # round again and again. add() de-duplicates on the fix timestamp
            # and returns False for a repeat, which is also the signal that
            # there is simply nothing new to act on.
            fresh = self._tag_win.add(rx_fix, rz_fix, float(fix.get("yaw_deg", 0.0)),
                                      float(fix.get("rms_px", 0.0)),
                                      int(fix.get("n_tags", 1) or 1),
                                      tier=tier, t=float(fix.get("t", time.time())),
                                      key=(source, fix.get("t")))
            if not fresh:
                return None

        mag0 = math.hypot(rx_fix - p["x"], rz_fix - p["z"])
        # How much corroboration this correction needs. A small correction from
        # a strict frame applies immediately, exactly as before. A large one
        # still needs TAGS_CONFIRM_N. A loose frame always needs a consensus,
        # whatever its size — that is the whole basis for having loosened the
        # per-frame thresholds in the first place.
        need = int(config.TAGS_SMALL_FIX_CONSENSUS_N)
        if mag0 > float(config.TAGS_BIG_FIX_MM):
            need = max(need, int(config.TAGS_CONFIRM_N))
        if tier != "strict":
            need = max(need, int(config.TAGS_LOOSE_CONSENSUS_N))

        cons = None
        if use_window:
            cons, cwhy = self._tag_win.consensus(need)
            if cons is None:
                with self._lock:
                    self._tag_pending = {"need": need, "tier": tier, "why": cwhy,
                                         "window": self._tag_win.summary()}
                return None
            fx, fz = cons["x"], cons["z"]
        else:
            if need > 1:
                # Temporal fusion off: fall back to the original behaviour of
                # requiring N successive candidates that all agree.
                self._tag_run.append((rx_fix, rz_fix))
                del self._tag_run[:-int(config.TAGS_CONFIRM_N)]
                if len(self._tag_run) < int(config.TAGS_CONFIRM_N):
                    return None
                ax = sum(v[0] for v in self._tag_run) / len(self._tag_run)
                az = sum(v[1] for v in self._tag_run) / len(self._tag_run)
                if any(math.hypot(v[0] - ax, v[1] - az) > float(config.TAGS_AGREE_MM)
                       for v in self._tag_run):
                    self._tag_rejected += 1
                    return None                       # candidates disagree
            else:
                self._tag_run = []
            fx, fz = rx_fix, rz_fix

        dx, dz = fx - p["x"], fz - p["z"]
        mag = math.hypot(dx, dz)
        a = float(config.TAGS_FIX_ALPHA)
        step = float(config.TAGS_FIX_MAX_STEP_MM)
        sx, sz = dx * a, dz * a
        smag = math.hypot(sx, sz)
        if smag > step and smag > 1e-9:
            sx, sz = sx * step / smag, sz * step / smag
        with self._lock:
            if self._anchor_map is not None:
                # Shifting the anchor moves the whole map frame under the rover;
                # yaw is untouched because it never references _anchor_map.
                self._anchor_map = (self._anchor_map[0] + sx, self._anchor_map[1] + sz)
            # A measured fix is a known-good pose: the drift margin starts over.
            self._drift_dist_mm = 0.0
            self._drift_turn_deg = 0.0
            self._drift_last_pose = None
            self._tag_applied += 1
            self._tag_pending = None
            self._tag_last = {"t": round(time.time(), 2),
                              "n_tags": fix.get("n_tags"), "ids": fix.get("ids"),
                              "rms_px": round(float(fix.get("rms_px", 0.0)), 2),
                              "resid_mm": round(mag), "applied_mm": round(math.hypot(sx, sz)),
                              "yaw_err_deg": round(yaw_err, 2),
                              "source": source, "tier": tier,
                              "spread_mm": fix.get("spread_mm"),
                              "range_mm": fix.get("range_mm"),
                              "spread_ratio": fix.get("spread_ratio"),
                              # What actually backed this correction: how many
                              # observations agreed, over what span, and how
                              # tightly. A wide scatter with a passing consensus
                              # is the early warning that the map is drifting
                              # out of agreement with the arena.
                              "consensus": ({"n": cons["n"], "need": cons["need"],
                                             "span_s": cons["span_s"],
                                             "scatter_mm": cons["scatter_mm"],
                                             "dropped": cons["dropped"],
                                             "strict": cons["strict"],
                                             "loose": cons["loose"]}
                                            if cons else {"n": 1, "need": need,
                                                          "span_s": 0.0,
                                                          "scatter_mm": 0.0,
                                                          "dropped": 0,
                                                          "strict": 1, "loose": 0})}
            out = dict(self._tag_last)
        return out

    def _note_tag_reason(self, code):
        """Track how long the localiser has been refusing on GEOMETRY.

        Only "degenerate" counts: it is the one failure that moving a little
        sideways can fix, and it must not be confused with "nothing in view"
        (where a jog would achieve nothing) or "unknown ids" (where the fix is
        in map.json). Counting it here is what lets the monitor loop decide the
        rover should manufacture its own baseline."""
        with self._lock:
            self._tag_reason = code
            if code == "degenerate":
                self._degen_run += 1
            else:
                self._degen_run = 0

    # ------------------------------------------------------- micro-parallax
    def parallax_fix(self, baseline_mm=None, direction=None):
        """Manufacture a stereo baseline by jogging sideways, and fuse the pair.

        When the rover can only see a narrow or single-column tag set there is
        nothing left to gate: the geometry itself carries no information about
        sideways position, and no threshold can recover what was never there.
        The angle has to come from somewhere else — so the rover makes it. It
        strafes a small, known distance while tracking the same tags, and the
        two views separated by a MEASURED baseline fuse into one well-conditioned
        solve, the way a stereo pair would (tag_localizer.solve_multiview).

        Three properties make this trustworthy where a single view is not:
          * the baseline is measured by the T265 over a sub-second move, which
            is the regime where VIO is reliable — it drifts over minutes, not
            over 200 mm;
          * the strafe holds yaw, so the rotation really is common to both views,
            which is what the fusion assumes (and it is re-checked afterwards);
          * the baseline runs across the line of sight, the optimal direction,
            rather than partly along it as panel-mounted tags usually do.

        Returns (ok, message, detail).
        """
        if not (config.TAGS_ENABLED and config.TAGS_PARALLAX_ENABLED):
            return False, "micro-parallax disabled in config", None
        det = self._detector
        if det is None or not hasattr(det, "tag_observation"):
            return False, "detector has no tag observations to pair", None
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return False, "navigation in progress", None
        with self._lock:
            if self._parallax_busy:
                return False, "a parallax run is already under way", None
            self._parallax_busy = True
        try:
            return self._parallax_run(det, baseline_mm, direction)
        except Exception as exc:
            # A manoeuvre that throws must not take the monitor thread or the
            # API request with it, and must not leave the busy flag stuck.
            return False, "micro-parallax failed: %s" % exc, None
        finally:
            with self._lock:
                self._parallax_busy = False
                self._parallax_t = time.time()
                self._degen_run = 0        # give the new geometry a fresh start

    def _parallax_run(self, det, baseline_mm, direction):
        t_out = float(config.TAGS_PARALLAX_OBS_TIMEOUT_S)
        if self.rover.is_busy():
            return False, "rover is busy", None
        p0 = self.pose()
        if p0 is None:
            return False, "no rover pose", None
        # The window describes where the rover is standing NOW; it is about to
        # stop being true.
        self._tag_win.clear("parallax manoeuvre")
        view_a = det.tag_observation(newer_than=time.time(), timeout=t_out)
        if view_a is None:
            return False, ("no tags in view to pair — micro-parallax adds a "
                           "viewpoint, it cannot conjure markers"), None
        # Size the jog from the range: what buys conditioning is the ANGLE the
        # baseline subtends, so a fixed number of millimetres is right at one
        # distance and useless at every other.
        rng = det.tag_range_mm()
        if baseline_mm is None:
            want = (float(config.TAGS_PARALLAX_RATIO_TARGET) * float(rng)
                    if rng else 200.0)
            baseline_mm = max(float(config.TAGS_PARALLAX_BASELINE_MIN_MM),
                              min(float(config.TAGS_PARALLAX_BASELINE_MAX_MM), want))
        baseline_mm = abs(float(baseline_mm))
        # Which way to step. Both are equally good geometrically, so the choice
        # is purely about what the rover can safely occupy.
        margin = float(config.TAGS_PARALLAX_CLEARANCE_MM)
        options = [1.0, -1.0]
        if direction in ("left", "-", -1):
            options = [-1.0, 1.0]
        sign = None
        for cand in options:
            nx, nz = tag_fusion.lateral_offset_xz(p0["x"], p0["z"], p0["yaw_deg"],
                                                  cand * (baseline_mm + margin))
            if self._ignore_obstacles or self._center_free(nx, nz):
                sign = cand
                break
        if sign is None:
            return False, ("no clear %.0f mm of lateral room either side — the "
                           "rover cannot make a baseline from here"
                           % (baseline_mm + margin)), None

        with self._lock:
            hold_yaw = self._anchor_pose["yaw"] if self._anchor_pose else None
        moved_back = False
        try:
            res = self.rover.move(right=sign * baseline_mm, forward=0.0,
                                  units="mm", hold_yaw=hold_yaw)
            if res is None or not res:
                return False, ("parallax jog failed (%s)"
                               % (res.reason if res is not None else "no result")), None
            # Motion blur destroys corner precision, which is the entire basis
            # of the fix. Let the base settle before looking.
            self._stop.wait(float(config.TAGS_PARALLAX_SETTLE_S))
            p1 = self.pose()
            if p1 is None:
                return False, "lost rover pose mid-manoeuvre", None
            # The fusion assumes one common rotation. Check that rather than
            # trusting it: a yaw excursion during the strafe would tilt the
            # second view and quietly bias the pooled solve.
            dyaw = _wrap_deg(p1["yaw_deg"] - p0["yaw_deg"])
            if abs(dyaw) > float(config.TAGS_PARALLAX_MAX_YAW_DRIFT_DEG):
                return False, ("heading moved %.1f deg during the jog (limit "
                               "%.1f) — the two views no longer share a rotation, "
                               "so they cannot be fused"
                               % (dyaw, config.TAGS_PARALLAX_MAX_YAW_DRIFT_DEG)), None
            dx, dz = p1["x"] - p0["x"], p1["z"] - p0["z"]
            achieved = math.hypot(dx, dz)
            if achieved < float(config.TAGS_PARALLAX_MIN_SPREAD_MM):
                return False, ("only %.0f mm of baseline achieved (wanted %.0f) "
                               "— the rover did not actually move"
                               % (achieved, baseline_mm)), None
            view_b = det.tag_observation(newer_than=time.time(), timeout=t_out)
            if view_b is None:
                return False, "lost sight of the tags after the jog", None
            # MEASURED displacement, not the commanded one: the point of using
            # the T265 here is that it reports what actually happened.
            delta = tag_fusion.map_delta_to_world_vec(dx, dz)
            fix, why = det.tag_solve_views(
                [{"obs": view_a["obs"], "delta": (0.0, 0.0, 0.0)},
                 {"obs": view_b["obs"], "delta": delta}],
                min_ratio=float(config.TAGS_PARALLAX_MIN_RATIO),
                min_spread_mm=float(config.TAGS_PARALLAX_MIN_SPREAD_MM))
        finally:
            # Put the rover back where it was found, whatever happened above —
            # including on a failure, so a refused fusion never leaves the demo
            # displaced. Skipped only if the operator cancelled.
            #
            # The return is driven by the MEASURED displacement, not by undoing
            # the commanded one. A jog that stalled against something, or that
            # timed out half way, would otherwise be "undone" by a full-length
            # move in the opposite direction and leave the rover further from
            # where it started than the manoeuvre ever took it.
            if config.TAGS_PARALLAX_RETURN and not self._cancel.is_set():
                try:
                    pnow = self.pose()
                    if pnow is not None:
                        mdx, mdz = p0["x"] - pnow["x"], p0["z"] - pnow["z"]
                        if math.hypot(mdx, mdz) > 5.0:
                            # Map delta -> this rover's own (right, forward),
                            # the same rotation each drive leg uses.
                            g = math.radians(pnow["yaw_deg"])
                            mr, mf = mdx, -mdz
                            cr = mr * math.cos(g) + mf * math.sin(g)
                            cf = -mr * math.sin(g) + mf * math.cos(g)
                            self.rover.move(right=cr, forward=cf, units="mm",
                                            hold_yaw=hold_yaw)
                            moved_back = True
                except Exception:
                    pass
        detail = {"t": round(time.time(), 2),
                  "baseline_mm": round(achieved),
                  "commanded_mm": round(sign * baseline_mm),
                  "range_mm": (round(rng) if rng else None),
                  "returned": moved_back,
                  "yaw_drift_deg": round(dyaw, 2),
                  "ids_a": view_a["obs"]["ids"], "ids_b": view_b["obs"]["ids"]}
        if fix is None:
            detail["ok"] = False
            detail["why"] = why
            with self._lock:
                self._tag_parallax = detail
            return False, "parallax pair not usable: %s" % why, detail
        detail.update({"ok": True, "rms_px": round(float(fix.get("rms_px", 0.0)), 2),
                       "spread_mm": fix.get("spread_mm"),
                       "spread_ratio": fix.get("spread_ratio"),
                       "n_tags": fix.get("n_tags")})
        # Consume against the pose at VIEW A, which is what the solve describes.
        applied = self._consume_fix(fix, pose_ref=p0, source="parallax")
        detail["applied"] = applied
        with self._lock:
            self._tag_parallax = detail
        if applied is None:
            return True, ("parallax fix solved (rms %.1f px, %.0f mm baseline) "
                          "but not yet applied — awaiting corroboration"
                          % (detail["rms_px"], detail["baseline_mm"])), detail
        return True, ("parallax fix applied: %d mm residual, %d mm eased in, "
                      "%.0f mm baseline"
                      % (applied["resid_mm"], applied["applied_mm"],
                         detail["baseline_mm"])), detail

    def _parallax_worker(self):
        ok, msg, _d = self.parallax_fix()
        with self._lock:
            if not ok:
                self._message = "micro-parallax: %s" % msg

    def _zupt_sample(self):
        """Zero-velocity update. Called while the rover is stopped between legs:
        true velocity is zero, so any pose movement the T265 reports over the
        window is pure drift. Records it as a rate (mm/s) — the coefficient the
        drift margin and any 'stop and re-fix' policy would key off."""
        win = float(config.NAV_ZUPT_WINDOW_S)
        if win <= 0.0:
            return
        a = self.pose()
        if a is None:
            return
        self._stop.wait(win)
        b = self.pose()
        if b is None:
            return
        rate = math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / win
        with self._lock:
            n = self._zupt_samples
            prev = self._zupt_mm_s
            # running mean, so one noisy sample doesn't dominate the readout
            self._zupt_mm_s = rate if prev is None else (prev * n + rate) / (n + 1)
            self._zupt_samples = n + 1

    def _await_confidence(self):
        """Hold before starting a leg until T265 tracking confidence recovers.

        confidence is 0=Failed, 1=Low, 2=Medium, 3=High; only 0 previously
        stopped anything, yet Low is exactly when drift accrues fastest. Waits
        up to NAV_CONF_WAIT_S for NAV_MIN_START_CONF. If it never arrives the
        leg still runs — stranding a live demo is worse — but the speed cap is
        scaled down while confidence is poor, and restored afterwards."""
        need = int(config.NAV_MIN_START_CONF)
        wait_s = float(config.NAV_CONF_WAIT_S)
        deadline = time.time() + wait_s
        conf = None
        while time.time() < deadline and not self._cancel.is_set():
            pr = self.rover.get_pose()
            conf = pr.get("confidence") if pr else None
            if conf is None or conf >= need:
                break
            self._stop.wait(0.1)
        with self._lock:
            want = float(self._speed_setpoint)
        if conf is not None and conf < need:
            scale = float(config.NAV_LOW_CONF_SPEED_SCALE)
            self.rover.cfg.MAX_LINEAR = max(float(config.NAV_SPEED_MIN), want * scale)
            with self._lock:
                self._message = ("tracking confidence %d (<%d): driving at %.2f m/s "
                                 "(%.0f%% of the %.2f m/s requested)"
                                 % (conf, need, self.rover.cfg.MAX_LINEAR, scale * 100, want))
        else:
            # restore the OPERATOR's setting, not the startup default
            self.rover.cfg.MAX_LINEAR = want

    def _log_leg(self, leg, cmd_mm, wx, wz, res):
        """Record commanded vs achieved displacement for this leg."""
        p = self.pose()
        got = None
        if p is not None:
            got = cmd_mm - math.hypot(wx - p["x"], wz - p["z"])
        entry = {"leg": leg,
                 "cmd_mm": round(cmd_mm, 1),
                 "got_mm": (round(got, 1) if got is not None else None),
                 "err_mm": (round(cmd_mm - got, 1) if got is not None else None),
                 "ok": bool(res) if res is not None else False,
                 "reason": (res.reason if res is not None else "no result")}
        with self._lock:
            self._leg_log.append(entry)
            del self._leg_log[:-20]      # keep the last 20

    def _drift_accumulate(self, p):
        """Integrate path length and |turn| from successive pose samples."""
        if p is None:
            return
        with self._lock:
            prev = self._drift_last_pose
            self._drift_last_pose = (p["x"], p["z"], p["yaw_deg"])
            if prev is None:
                return
            self._drift_dist_mm += math.hypot(p["x"] - prev[0], p["z"] - prev[1])
            self._drift_turn_deg += abs(_wrap_deg(p["yaw_deg"] - prev[2]))

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

    def _in_exclusion(self, x, z):
        """True when map point (x, z) falls inside a detection no-target zone
        (an obstacle rectangle grown by the clear margin). Used to invalidate
        detections that project onto a spot where a target can never be."""
        for (x1, z1, x2, z2) in self._exclusion:
            if x1 <= x <= x2 and z1 <= z <= z2:
                return True
        return False

    def _bounds_excess(self, x, z):
        """How far a rover footprint centred at (x, z) protrudes past the arena
        edge, in mm (0 = fully inside)."""
        hx = self.car_w / 2.0
        hz = self.car_l / 2.0
        return max(0.0,
                   hx - x, (x + hx) - self.map_w,
                   hz - z, (z + hz) - self.map_d)

    def _out_of_bounds(self, x, z):
        """Emergency-stop test: True when the rover footprint has left the arena
        by more than it was ever entitled to.

        The planner keeps the footprint inside the map, so on a valid path this
        never fires. It is the runtime backstop for the case the planner cannot
        cover — the rover DRIFTING out, or a leg overshooting — which previously
        had no check at all: _footprint_blocked only ever consulted obstacles,
        so nothing stopped the rover leaving the arena entirely.

        `_bounds_slack` mirrors the planner's allowance for a start pose that
        already protrudes (e.g. parked in a corner with its centre on the corner
        point); without it the rover would e-stop the instant it was placed."""
        return (self._bounds_excess(x, z)
                > self._bounds_slack + float(config.NAV_BOUNDS_STOP_MARGIN_MM))

    def _footprint_blocked(self, x, z):
        """Emergency-stop test: True when a rover centered at (x, z) has its
        footprint (car half-extent + NAV_OBSTACLE_STOP_MARGIN_MM) overlapping a
        RAW obstacle.

        NOTE: this only fires below the planner's own margin when
        NAV_OBSTACLE_STOP_MARGIN_MM < the clearance the route achieved (reported
        as state()["plan_clearance_mm"]). If the stop margin is the larger of
        the two, a perfectly valid path trips it the moment it is driven."""
        m = float(config.NAV_OBSTACLE_STOP_MARGIN_MM)
        hx = self.car_w / 2.0 + m
        hz = self.car_l / 2.0 + m
        for o in self._obstacles:
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
        with self._lock:
            standoff = float(self._standoff_mm)
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
        tel, seq = self._detector.telemetry.get()
        if not tel:
            return None
        tgt = tel.get("target")
        if not tgt or tgt.get("range") is None or tgt.get("az") is None:
            return None
        # Radar is only rejected here while moving if the detector has no
        # lock to maintain (it clears the target in that case, so tgt would
        # already be None/non-radar). An ALREADY-acquired lock is kept live
        # by the detector's tracker through the rover's own motion (purely
        # spatial maintenance, no Doppler involved -- see radar_tracker.py),
        # so it's trusted here the same as any other live source. Acquiring
        # a brand-new radar lock still requires the rover to be stationary;
        # that rule lives in RadarTracker/detector.py, not here.
        p = self.pose()
        if p is None:
            return None
        rng_mm = float(tgt["range"]) * 1000.0
        # Bearing and range are measured from the COLOUR CAMERA, which sits at
        # its own lever arm from the rover's turn centre. Projecting from the
        # rover centre makes a stationary target appear to swing every time the
        # rover rotates, by the camera's arc. Walk out to the camera first.
        cx, cz = self._camera_map_xz(p)
        bearing = math.radians(p["yaw_deg"] + float(tgt["az"]))
        tx = cx + rng_mm * math.sin(bearing)
        tz = cz - rng_mm * math.cos(bearing)
        # Clamp into the map so a noisy range can't paint the marker off-canvas.
        tx = max(0.0, min(self.map_w, tx))
        tz = max(0.0, min(self.map_d, tz))
        # Invalidate detections that land in a no-target zone (obstacle + clear
        # margin): a real target can't be there, so do NOT update the target /
        # track — the rover never generates a goal from such a detection. The
        # previous target simply holds and ages out as usual.
        if self._in_exclusion(tx, tz):
            return None
        # Reflected intensity (mmWave only; None for a vision-sourced target).
        # Used below purely as a CONSISTENCY check against a reference value —
        # never as an absolute quality gate here (the radar tracker already
        # applies its own SNR floor before it ever offers a target).
        is_radar = tgt.get("source") == "radar"
        snr = float(tgt["snr_peak"]) if (is_radar and tgt.get("snr_peak") is not None) else None
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
            is_new_frame = seq != self._last_tel_seq
            self._last_tel_seq = seq

            jump = float(config.NAV_TARGET_JUMP_MM)
            relock = float(config.NAV_TARGET_RELOCK_S)
            forget = float(config.NAV_TARGET_FORGET_S)
            confirm_n = max(1, int(config.NAV_TARGET_CONFIRM_N))
            snr_tol = float(config.NAV_TARGET_SNR_TOL)

            def _snr_close(a, b):
                """True when two radar reflection intensities are within
                snr_tol of each other (relative tolerance) -- a stand-in for
                "this still looks like the same reflector, not interference".
                Either side missing (non-radar, or no baseline yet) -> no
                opinion, so it never blocks a vision-side comparison."""
                if a is None or b is None:
                    return True
                return abs(a - b) <= snr_tol * max(a, b, 1e-6)

            tr = self._track
            stale = tr is None or (now - tr["t"]) > forget
            near = (not stale
                    and math.hypot(fx - tr["x"], fz - tr["z"]) <= jump
                    and _snr_close(snr, tr.get("snr")))

            if near:
                # Trusted continuity: light EMA update straight onto the track,
                # same as before -- an already-locked target doesn't need to
                # re-earn trust every frame, only a NEW or relocated one does.
                a = 0.5
                self._track = {"x": a * fx + (1 - a) * tr["x"],
                               "z": a * fz + (1 - a) * tr["z"], "t": now,
                               "snr": snr if snr is not None else tr.get("snr")}
                self._track_reject_since = None
                self._pending_run = []
                self._accum = {"phase": "confirmed", "n": confirm_n, "need": confirm_n}
            else:
                # Disagrees with the current track (continuity gate), or there's
                # no usable track yet (first-ever lock / long loss). GHOST GUARD:
                # this is only committed once NAV_TARGET_CONFIRM_N mutually-
                # consistent, DISTINCT detector frames have accumulated — a
                # single-frame ghost never earns enough corroboration to
                # overwrite the track (or seed a new one) and just ages out of
                # the accumulation window. `is_new_frame` de-dupes repeated
                # polls of the same detector output (this method is called
                # faster than the detector produces frames) so the count
                # reflects genuinely separate observations.
                if not stale and self._track_reject_since is None:
                    self._track_reject_since = now
                if is_new_frame:
                    run = self._pending_run
                    ref = run[-1] if run else None
                    consistent = (ref is None
                                  or (math.hypot(fx - ref["x"], fz - ref["z"]) <= jump
                                      and _snr_close(snr, ref.get("snr"))))
                    if consistent:
                        run.append({"x": fx, "z": fz, "snr": snr, "t": now})
                    else:
                        run = [{"x": fx, "z": fz, "snr": snr, "t": now}]  # restart the run here
                    # Bound the run to the relock window so it can't be stitched
                    # together from detections spread too far apart in time.
                    self._pending_run = [q for q in run if now - q["t"] <= relock]

                ready = len(self._pending_run) >= confirm_n
                sustained = stale or (now - self._track_reject_since >= relock)
                if ready and sustained:
                    gx = _median([q["x"] for q in self._pending_run])
                    gz = _median([q["z"] for q in self._pending_run])
                    snrs = [q["snr"] for q in self._pending_run if q["snr"] is not None]
                    gsnr = _median(snrs) if snrs else None
                    self._track = {"x": gx, "z": gz, "t": now, "snr": gsnr}
                    self._track_reject_since = None
                    self._pending_run = []
                    self._accum = {"phase": "confirmed", "n": confirm_n, "need": confirm_n}
                else:
                    if tr is not None:
                        tr["t"] = now   # hold the last verified position while we wait
                    self._accum = {"phase": "accumulating",
                                   "n": len(self._pending_run), "need": confirm_n}

            if self._track is None:
                return None   # still accumulating confirmations; nothing verified yet
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
        # The planner maximises the route's MINIMUM distance to any obstacle
        # (not just the configured minimum) and reports what it achieved, so the
        # UI/operator can see how much margin this particular route really has.
        # Odometry error since the last known-good pose earns extra planning
        # margin, so routes are held further off obstacles the longer it has
        # been since a fix. Applied as a raised clearance FLOOR; the planner's
        # own max-min search still pushes above it wherever geometry allows.
        drift_mm = self._drift_margin_mm()
        plan_cfg = self.plan_cfg
        if drift_mm > 0.0:
            plan_cfg = dict(plan_cfg)
            plan_cfg["clearance"] = float(plan_cfg.get("clearance", 0)) + drift_mm
        segs, achieved = plan_rectilinear_path_ex(
            self.map, plan_cfg, start, end,
            ignore_start_obstacle=self._ignore_obstacles)
        if segs is None and drift_mm > 0.0:
            # The drift margin alone made this unreachable. Fall back to the
            # configured clearance rather than refusing to move: a conservative
            # margin must never be the reason the rover strands itself.
            segs, achieved = plan_rectilinear_path_ex(
                self.map, self.plan_cfg, start, end,
                ignore_start_obstacle=self._ignore_obstacles)
        if segs is None:
            with self._lock:
                self._status = "no_path"
                self._message = "planner found no obstacle-free rectilinear path"
                self._plan_clearance = None
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
            self._plan_clearance = achieved
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
                self._drift_accumulate(p)
                dx = wx - p["x"]
                dz = wz - p["z"]
                # Residual already inside the arrival tolerance: driving it would
                # satisfy POS_TOL immediately and still burn SETTLE_TIME for
                # nothing. Skipping is safe because the NEXT leg is re-derived
                # from the live pose, and for the final leg the residual is
                # within tolerance by definition.
                leg_mm = math.hypot(dx, dz)
                if leg_mm < float(config.NAV_POS_TOL) * 1000.0:
                    continue
                if self._cancel.is_set():
                    return
                # Zero-velocity drift sample: the rover is stopped between legs,
                # so its TRUE velocity is zero and any pose change the T265
                # reports here is drift, measured directly.
                self._zupt_sample()
                self.apply_tag_fix()        # absolute correction while stopped
                # Tracking-confidence gate: Low confidence is exactly when VIO
                # drift accrues fastest. Wait briefly for it to recover; if it
                # doesn't, still go (never strand a demo) but at reduced speed.
                self._await_confidence()
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
                    if lp is not None and self._out_of_bounds(lp["x"], lp["z"]):
                        # Bounds violations are NOT suppressed by
                        # ignore_obstacles: that override exists to let the rover
                        # drive out of an obstacle keep-out it is already inside,
                        # which is a deliberate recovery. Leaving the arena is
                        # never a recovery, so this stop is unconditional.
                        self.rover.stop()
                        self._cancel.set()
                        with self._lock:
                            self._status = "blocked"
                            self._message = ("E-STOP: rover footprint left the arena "
                                             "at (%.0f, %.0f) mm — %.0f mm past the edge"
                                             % (lp["x"], lp["z"],
                                                self._bounds_excess(lp["x"], lp["z"])))
                        return
                    if (not self._ignore_obstacles and lp is not None
                            and self._footprint_blocked(lp["x"], lp["z"])):
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
                # Per-leg fidelity log: commanded vs achieved displacement. A
                # consistent ratio across many legs indicates a fixed VIO SCALE
                # error (correctable); random scatter indicates slip or noise.
                self._log_leg(i + 1, leg_mm, wx, wz, res)
                if res is None or not res:
                    reason = res.reason if res is not None else "no result"
                    with self._lock:
                        self._status = "error"
                        self._message = (f"leg {i + 1} ({dx:+.0f}, {dz:+.0f})mm "
                                         f"failed ({reason})")
                        # Clear slate: drop the dead plan so this failed/incomplete
                        # move leaves no residual goal/path behind it — the next
                        # navigate call (manual, AUTO, or FOLLOW) starts fresh
                        # instead of being queued behind stale state.
                        self._goal, self._path, self._leg = None, None, 0
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

    # --------------------------------------------------------- standoff
    def set_standoff(self, mm):
        """Set how far short of the target the rover stops (mm), clamped to
        [NAV_STANDOFF_MIN_MM, NAV_STANDOFF_MAX_MM].

        This is the SOFT preference in compute_goal: obstacle avoidance still
        wins, so the achieved distance may differ when the preferred spot is
        blocked. Returns the clamped value actually applied."""
        lo = float(config.NAV_STANDOFF_MIN_MM)
        hi = float(config.NAV_STANDOFF_MAX_MM)
        v = max(lo, min(hi, float(mm)))
        with self._lock:
            self._standoff_mm = v
        return v
    # ------------------------------------------------------------ speed
    def set_speed(self, mps):
        """Set the navigation translation-speed cap (MAX_LINEAR), clamped to
        [NAV_SPEED_MIN, NAV_SPEED_MAX]. The control loop reads cfg.MAX_LINEAR
        live, so this takes effect on the next tick — even mid-move. Returns
        the clamped value actually applied."""
        lo, hi = float(config.NAV_SPEED_MIN), float(config.NAV_SPEED_MAX)
        v = max(lo, min(hi, float(mps)))
        with self._lock:
            self._speed_setpoint = v      # remembered across confidence throttling
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

    # ------------------------------------------------------------ obstacle override
    def set_ignore_obstacles(self, enabled):
        """Operator override: when enabled, the rover's OWN current position
        being inside an obstacle/clearance zone never blocks starting a plan
        (rectilinear_mm's start-blocked check) or continuing a move (the
        mid-move footprint e-stop in _drive). Obstacle avoidance elsewhere
        (routing around every obstacle, the standoff-goal search near the
        target) is unaffected. Takes effect immediately, including mid-move."""
        with self._lock:
            self._ignore_obstacles = bool(enabled)
        return self._ignore_obstacles

    # ------------------------------------------------------------ ghost-guard confirm count
    def set_confirm_n(self, n):
        """Operator-settable ghost-detection guard threshold: how many
        mutually-consistent detector frames must accumulate before a new/
        relocated target is trusted (see project_detection). Clamped to
        [1, 10]. project_detection() re-reads config.NAV_TARGET_CONFIRM_N
        live every call, so this takes effect on the very next detection --
        no restart needed. Returns the clamped value actually applied."""
        applied = max(1, min(10, int(n)))
        with self._lock:
            config.NAV_TARGET_CONFIRM_N = applied
        return applied

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
                # Still has to pass the confirm-N ghost guard before it commits.
                self._track = None
                self._track_reject_since = None
                self._pending_run = []

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
          2. COMMIT + MOVE: plan + drive to the standoff goal. Either source may
             re-plan early if the live target moves > NAV_FOLLOW_REPLAN_MM — an
             already-ACQUIRED radar lock is maintained by the tracker purely
             spatially through the rover's own motion (no Doppler involved, so
             ego-motion doesn't corrupt it; see radar_tracker.py), the same as
             a vision target. NAV_FOLLOW_CYCLE_S is a stuck-backstop.
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
                # Mid-move re-plan when the live target has moved. Vision is
                # always live; radar now stays live mid-move too (an already-
                # acquired lock is maintained purely spatially, unaffected by
                # the rover's own ego-motion), so either source can trigger
                # this re-plan.
                nt = self.project_detection()
                if (nt is not None
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

    def _maybe_parallax(self, busy):
        """Start a parallax run if the geometry has been unusable long enough.

        Deliberately conservative about WHEN: never during a navigation, never
        while a run is already going, and never more often than the cooldown —
        the manoeuvre moves the rover, and a rover that jogs sideways on its own
        during a demo had better have a good reason each time."""
        if not (config.TAGS_ENABLED and config.TAGS_PARALLAX_ENABLED
                and config.TAGS_PARALLAX_AUTO):
            return
        if busy or self._parallax_busy or self.rover.is_busy():
            return
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return
        with self._lock:
            run = self._degen_run
        if run < int(config.TAGS_PARALLAX_TRIGGER_N):
            return
        if time.time() - self._parallax_t < float(config.TAGS_PARALLAX_COOLDOWN_S):
            return
        self._parallax_t = time.time()      # claim the slot before the thread
        threading.Thread(target=self._parallax_worker, daemon=True).start()

    def _monitor_loop(self):
        """Continuously project the detection onto the map (for the UI), and in
        AUTO mode dispatch a navigation once the target holds still. (FOLLOW is
        driven by its own _follow_loop.)"""
        while not self._stop.is_set():
            t = self.project_detection()
            with self._lock:
                auto = self._auto
                busy = self._status in ("planning", "moving")
            # Absolute tag correction whenever the rover is STOPPED — not just
            # between the legs of an active move, which was the only place this
            # ran. Parked, arrived, or waiting between navigations, the rover
            # was accumulating drift with no correction at all, while sitting
            # still and looking straight at the tags: the one condition under
            # which a fix is both possible and most reliable. apply_tag_fix
            # already refuses while the rover is commanded to move, so this is
            # safe to call unconditionally; the interval just avoids spinning.
            if time.time() - self._tag_fix_t >= float(config.TAGS_IDLE_FIX_INTERVAL_S):
                self._tag_fix_t = time.time()
                self.apply_tag_fix()
            # Micro-parallax rescue. When the localiser has been refusing on
            # GEOMETRY for several frames running, no amount of waiting will
            # help: a single tag column carries no sideways information at all,
            # so the rover has to go and make some. Only the "degenerate" verdict
            # triggers this — jogging would achieve nothing if the real problem
            # were an empty view or an unmapped marker id.
            self._maybe_parallax(busy)
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
            # Canonical min-corner COLLISION boxes, so the browser draws the
            # keep-out the rover actually respects and needs no schema logic.
            "obstacles": [dict(o) for o in self._obstacles],
            "car": {"width": self.car_w, "length": self.car_l},
            "clearance": self.clearance,
            # The CONFIGURED start cell (map.json rover_start) — i.e. where the
            # operator should physically place the rover, which is what the
            # "Re-anchor to start cell" button anchors to. Not the live anchor,
            # which only matches after a re-anchor.
            "rover_start": {"x": self._map_start[0], "z": self._map_start[1]},
            "standoff_mm": round(float(self._standoff_mm)),
            "standoff_min_mm": float(config.NAV_STANDOFF_MIN_MM),
            "standoff_max_mm": float(config.NAV_STANDOFF_MAX_MM),
            "speed": round(float(self.rover.cfg.MAX_LINEAR), 3),
            "speed_min": float(config.NAV_SPEED_MIN),
            "speed_max": float(config.NAV_SPEED_MAX),
            "jog_speed": round(float(self._jog_speed), 3),
            "jog_speed_min": float(config.NAV_JOG_SPEED_MIN),
            "jog_speed_max": float(config.NAV_JOG_SPEED_MAX),
            "confirm_n": int(config.NAV_TARGET_CONFIRM_N),
        }

    def state(self):
        p = self.pose()
        with self._lock:
            target = dict(self._target) if self._target else None
            goal = dict(self._goal) if self._goal else None
            path = [list(w) for w in self._path] if self._path else None
            status, message, leg = self._status, self._message, self._leg
            auto, follow = self._auto, self._follow
            ignore_obstacles = self._ignore_obstacles
            accum = dict(self._accum)
            plan_clearance = self._plan_clearance
            zupt = self._zupt_mm_s
            drift_d, drift_t = self._drift_dist_mm, self._drift_turn_deg
            leg_log = list(self._leg_log[-5:])
            tag_last, tag_ap, tag_rj = self._tag_last, self._tag_applied, self._tag_rejected
            tag_why = self._tag_reject_reason
            tag_live = self._tag_live
            tag_idle = self._tag_idle
            tag_pending = dict(self._tag_pending) if self._tag_pending else None
            tag_parallax = dict(self._tag_parallax) if self._tag_parallax else None
            parallax_busy = self._parallax_busy
            degen_run = self._degen_run
            yh = list(self._tag_yaw_hist)
        tag_window = self._tag_win.summary()
        if tag_last is not None:
            # Age is computed HERE, against the same clock that stamped it. The
            # browser previously did (Date.now()/1000 - t), so any skew between
            # the Jetson and the operator laptop showed up directly as a bogus
            # age (a Jetson with no RTC battery and no NTP reads hours out).
            tag_last = dict(tag_last)
            tag_last["age_s"] = round(max(0.0, time.time() - float(tag_last.get("t", 0.0))), 1)
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
            # mm: the minimum body-to-obstacle distance this route achieves.
            # The planner maximises it, so it is normally well above the
            # configured `clearance` floor. Compare against
            # NAV_OBSTACLE_STOP_MARGIN_MM: a value at or below that margin means
            # the route runs close enough to trip the mid-move e-stop.
            "plan_clearance_mm": (round(plan_clearance)
                                  if plan_clearance is not None else None),
            # Drift diagnostics. pose_jumps counts T265 steps REJECTED as
            # glitches; reloc_accepted counts those ACCEPTED while stationary as
            # genuine relocalisations (a healthy non-zero value means the device
            # is correcting its own drift and the fix is being kept).
            # zupt_drift_mm_s is measured drift with the rover provably still.
            "reloc_accepted": getattr(self.rover, "reloc_accepted", 0),
            "zupt_drift_mm_s": (round(zupt, 1) if zupt is not None else None),
            "drift_since_fix": {"dist_mm": round(drift_d),
                                "turn_deg": round(drift_t, 1),
                                "margin_mm": round(self._drift_margin_mm())},
            "leg_log": leg_log,
            # Last APPLIED ArUco fix, plus applied/rejected counts. yaw_err_deg
            # is reported even though yaw is not corrected by default — watch it
            # settle before enabling TAGS_CORRECT_YAW.
            "tag_fix": tag_last,
            "tag_counts": {"applied": tag_ap, "rejected": tag_rj, "why": tag_why},
            "tag_live": tag_live,
            "tag_idle": tag_idle,
            # What is currently in the agreement window, and why a correction is
            # being held back if one is. "n agreeing of m" is the honest answer
            # to "why hasn't it corrected yet" — far more useful than silence.
            "tag_window": tag_window,
            "tag_pending": tag_pending,
            # Last micro-parallax attempt: the baseline it managed, whether the
            # pair fused, and what it applied.
            "tag_parallax": tag_parallax,
            "parallax_busy": parallax_busy,
            "degenerate_run": degen_run,
            # Spread of the tag-vs-T265 heading residual. A tight spread near
            # zero means the map agrees with reality; a wide one is the map
            # error showing, and is what forces the yaw gate open.
            "tag_yaw": ({"n": len(yh), "med": round(_median(yh), 2),
                         "min": round(min(yh), 2), "max": round(max(yh), 2)}
                        if yh else None),
            "accum": accum,
            "status": status,
            "message": message,
            "auto": auto,
            "follow": follow,
            "ignore_obstacles": ignore_obstacles,
            "confirm_n": int(config.NAV_TARGET_CONFIRM_N),
            "speed": round(float(self.rover.cfg.MAX_LINEAR), 3),
            # Operator-settable approach distance, so the UI field stays in
            # sync if another client changes it mid-session.
            "standoff_mm": round(float(self._standoff_mm)),
            "jog_speed": round(float(self._jog_speed), 3),
            "moving": self.rover.is_busy(),
            "mock": self.rover.using_mock,
            "ros": self.rover.ros_connected,
            "pose_jumps": getattr(self.rover, "pose_jumps", 0),
        }
