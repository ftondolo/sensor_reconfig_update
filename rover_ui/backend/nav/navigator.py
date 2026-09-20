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
    (x, z + 2000). If the rover footprint there is closer than `clearance` to
    an obstacle, or closer than `wall_clearance` to the arena edge, the goal
    is nudged to the nearest acceptable point;
  * plans an obstacle-avoiding rectilinear path with
    rectilinear_mm.plan_rectilinear_path_ex — which maximises the route's
    MINIMUM distance to any obstacle rather than merely clearing them by the
    configured `clearance`, and reports the margin achieved (surfaced as
    state()["plan_clearance_mm"]) — and executes it one axis-aligned leg
    at a time through T265RoverService.move_axis(axis, mm) — re-deriving each
    leg from the LIVE pose so per-move arrival error (~POS_TOL) does not
    accumulate along the path;
  * watches the rover FOOTPRINT (the car rectangle centred on the pose) while
    driving: within NAV_OBSTACLE_STOP_MARGIN_MM of an obstacle -> halt; inside
    an obstacle's clearance zone, or partly outside the arena -> shortest move
    back to a safe pose, re-plan to the same goal, continue (see _classify,
    _recover, _drive).

The T265 has exactly one owner in this process: the T265RoverService here.
(Do not start a second T265 pose sensor thread alongside this.)
"""
import collections
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

from rectilinear_mm import _seg_hits, footprint_gap  # noqa: E402  (footprint geometry)
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


def _robust_average(samples, inlier_mm, jump_mm):
    """Robust weighted mean of map samples [{x, z, snr, t}, ...] (oldest first).

    Centre = medoid: the SAMPLE with the smallest summed distance to all the
    others (always a real observation, never a midpoint between two clusters);
    ties go to the newest sample. Each sample is then weighted by its distance
    d from that centre: 1 within inlier_mm, tapering smoothly to 0 at jump_mm
    (a Huber-style taper), so a one-frame ghost contributes nothing while the
    genuine cluster is averaged. Radar samples whose reflection strength is far
    from the window's typical value are down-weighted (soft, never to zero).
    Returns {x, z, spread (weighted RMS distance to the mean, mm), inliers}."""
    n = len(samples)
    if n == 1:
        s0 = samples[0]
        return {"x": s0["x"], "z": s0["z"], "spread": 0.0, "inliers": 1}
    best_i, best_cost = n - 1, None
    for i in range(n - 1, -1, -1):          # newest first -> newest wins ties
        si = samples[i]
        cost = sum(math.hypot(si["x"] - q["x"], si["z"] - q["z"]) for q in samples)
        if best_cost is None or cost < best_cost - 1e-6:
            best_i, best_cost = i, cost
    cx, cz = samples[best_i]["x"], samples[best_i]["z"]
    snrs = [q["snr"] for q in samples if q.get("snr") is not None]
    sref = _median(snrs) if snrs else None
    inlier_mm = max(1.0, inlier_mm)
    jump_mm = max(inlier_mm + 1.0, jump_mm)
    ws, xs, zs = [], [], []
    for q in samples:
        d = math.hypot(q["x"] - cx, q["z"] - cz)
        if d <= inlier_mm:
            w = 1.0
        elif d >= jump_mm:
            w = 0.0
        else:
            w = (inlier_mm / d) * (jump_mm - d) / (jump_mm - inlier_mm)
        if w > 0.0 and sref is not None and q.get("snr") is not None:
            rel = abs(q["snr"] - sref) / max(q["snr"], sref, 1e-6)
            w *= max(0.25, 1.0 - rel)
        ws.append(w); xs.append(q["x"]); zs.append(q["z"])
    wsum = sum(ws)
    if wsum <= 1e-9:                         # cannot happen (centre has w=1), but be safe
        return {"x": cx, "z": cz, "spread": 0.0, "inliers": 1}
    mx = sum(w * x for w, x in zip(ws, xs)) / wsum
    mz = sum(w * z for w, z in zip(ws, zs)) / wsum
    spread = math.sqrt(sum(w * ((x - mx) ** 2 + (z - mz) ** 2)
                           for w, x, z in zip(ws, xs, zs)) / wsum)
    return {"x": mx, "z": mz, "spread": spread,
            "inliers": sum(1 for w in ws if w >= 0.5)}


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
        self._obstacle_rects = []    # (x1, z1, x2, z2) of _obstacles; set by _load_files
        self.wall_margin = 0.0       # config.json wall_clearance; set by _load_files
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
        self._zupt_last_t = 0.0      # clock time of the last ZUPT sample actually taken
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
        # Running-average window: the last NAV_TARGET_AVG_N_MAX DETECTION FRAMES
        # (fresh detections only), newest last, in map mm. The estimate uses
        # the newest NAV_TARGET_AVG_N of them. See _update_target_tracking.
        self._det_buf = collections.deque(maxlen=max(1, int(config.NAV_TARGET_AVG_N_MAX)))
        self._last_tel_seq = None    # last detector telemetry seq consumed (each
                                      # detector frame enters the window at most once)
        # UI-facing summary of the window (see _update_target_tracking):
        # phase "none" until the first detection frame, then "averaging" with
        # n = frames used, need = N, spread/inliers of the estimate.
        self._accum = {"phase": "none", "n": 0, "need": int(config.NAV_TARGET_AVG_N)}
        # ---- FOLLOW blind-time / corner-look state (see _corner_check) ----
        self._last_fresh_t = None    # when the last detection frame was ACCEPTED
                                      # into the window (any sensor) -> blind time
        self._prior = None           # {x, z, t}: last known target position, kept
                                      # when the window is emptied (look / long loss)
        self._prior_pending = []     # detections outside the prior's gate, waiting
                                      # for NAV_TARGET_PRIOR_CONFIRM_N to agree
        self._follow_ctx = None      # {tx, tz}: target the current FOLLOW plan aims at
        self._corner_replan_req = False  # re-plan at the next corner (deferred)
        self._drive_outcome = None   # "replan" when _drive stopped at a corner
        self._last_replan_t = 0.0    # hysteresis for corner re-plans
        self._look = {"state": "idle"}   # UI: last corner-look status
        self._plan_info = None       # UI: turn penalty / legs of the active plan
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
        self._track_thread = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self.rover.start(wait_for_pose=True, timeout=8.0)
        self.reset_pose()            # current physical spot == map rover_start
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()
        self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._follow_thread.start()
        # Single dedicated home for the stateful target tracker (see
        # _track_loop / _update_target_tracking): _monitor_loop, _follow_loop
        # and the UI all read its cached output through project_detection()
        # instead of each re-running the tracker at their own cadence.
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._track_thread.start()
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
        # Detection no-target zones: the raw obstacle rectangle (map.json) grown
        # by the clear margin (config.json `clearance`). A detection that
        # projects into one of these is treated as invalid — a person can never
        # be standing inside an obstacle or its clear keep-out, so the rover
        # must not generate a target there. This is the same geometry the UI
        # already draws as the red obstacle halo.
        ex = float(config.NAV_TARGET_EXCLUSION_MM)
        exclusion = [(o["x"] - ex, o["z"] - ex, o["x"] + o["w"] + ex, o["z"] + o["h"] + ex)
                     for o in obstacles]
        start = m.get("rover_start", {})
        # Default start: bottom-edge middle, rear bumper flush with the map's
        # bottom edge (center is half a car length up from it).
        map_start = (float(start.get("x", map_w / 2)),
                     float(start.get("z", map_d - car_l / 2)))
        wall_margin = float(pc.get("wall_clearance", 0))
        # Safety bands (footprint distance to an obstacle's collision box):
        #   hard halt < recover trigger < plan floor < escape target.
        hard = float(config.NAV_OBSTACLE_STOP_MARGIN_MM)
        tol = float(config.NAV_RECOVER_TOL_MM)
        if not (0.0 <= hard < clearance - tol < clearance):
            print("[nav] WARNING: safety bands out of order: hard halt %.0f mm must be "
                  "< recover trigger %.0f mm (clearance %.0f - NAV_RECOVER_TOL_MM %.0f)"
                  % (hard, clearance - tol, clearance, tol))
        with self._lock:
            self.map, self.plan_cfg = m, pc
            self.map_w, self.map_d = map_w, map_d
            self.car_w, self.car_l, self.clearance = car_w, car_l, clearance
            self._exclusion = exclusion
            self._obstacles = obstacles      # canonical collision boxes
            self._obstacle_rects = [(o["x"], o["z"], o["x"] + o["w"], o["z"] + o["h"])
                                    for o in obstacles]
            self.wall_margin = wall_margin
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
        if self._fp_obstacle_gap(x, z) < self.clearance:
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
            # A re-anchor moves the whole map frame by an unknown amount:
            # detections projected before it no longer line up with new ones.
            self._det_buf.clear()
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
                # Samples already in the target window were projected from the
                # pre-correction pose; move them with the map frame so the
                # running average stays consistent with new projections.
                for q in self._det_buf:
                    q["x"] += sx
                    q["z"] += sz
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
        drift margin and any 'stop and re-fix' policy would key off.

        This blocks for NAV_ZUPT_WINDOW_S, so sampling before every single leg
        put that pause on every waypoint of a multi-leg path. The drift rate
        does not need a fresh reading that often: skip the sample (and its
        wait) when the last one completed less than NAV_ZUPT_MIN_INTERVAL_S
        ago, UNLESS T265 confidence is currently below NAV_MIN_START_CONF --
        that is exactly when a fresh reading is worth pausing for."""
        win = float(config.NAV_ZUPT_WINDOW_S)
        if win <= 0.0:
            return
        min_interval = float(config.NAV_ZUPT_MIN_INTERVAL_S)
        now = time.time()
        if min_interval > 0.0:
            with self._lock:
                due = (now - self._zupt_last_t) >= min_interval
            if not due:
                pr = self.rover.get_pose()
                conf = pr.get("confidence") if pr else None
                if conf is None or conf >= int(config.NAV_MIN_START_CONF):
                    return    # sampled recently enough and tracking is healthy
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
            self._zupt_last_t = now

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
    # Every rover-vs-world test below is on the rover's FOOTPRINT: the
    # axis-aligned car rectangle (car_w along x, car_l along z) centred on the
    # pose, never the centre point alone. The pose is the chassis turn centre
    # (T265 lever arm already compensated), which is the footprint centre.
    def _fp_obstacle_gap(self, x, z):
        """Distance from the footprint at (x, z) to the nearest obstacle
        collision box, mm (per axis, the same measure the planner inflates by;
        negative = overlapping). +inf with no obstacles."""
        return footprint_gap(x, z, x, z, self._obstacle_rects, self.car_w, self.car_l)

    def _fp_wall_inset(self, x, z):
        """How far the footprint at (x, z) is inside the arena, mm (distance of
        its nearest side to the nearest edge; negative = protruding)."""
        hx, hz = self.car_w / 2.0, self.car_l / 2.0
        return min(x - hx, self.map_w - (x + hx), z - hz, self.map_d - (z + hz))

    def _pose_ok(self, x, z, clear, wall):
        return self._fp_obstacle_gap(x, z) >= clear and self._fp_wall_inset(x, z) >= wall

    def _center_free(self, x, z):
        """True when a rover centred at (x, z) keeps its footprint >= clearance
        from every obstacle and >= wall_margin inside the arena."""
        return self._pose_ok(x, z, self.clearance, self.wall_margin)

    def _classify(self, x, z):
        """Safety state of the footprint at (x, z):
          "hard"    within NAV_OBSTACLE_STOP_MARGIN_MM of an obstacle -> halt;
          "recover" inside the clearance zone (gap < clearance - NAV_RECOVER_TOL_MM)
                    or any part outside the arena -> leave by the shortest move;
          "ok"      otherwise.
        Obstacle tests are skipped while the operator's ignore_obstacles
        override is on; the arena test never is. Walls never cause a halt."""
        if not self._ignore_obstacles:
            g = self._fp_obstacle_gap(x, z)
            if g < float(config.NAV_OBSTACLE_STOP_MARGIN_MM):
                return "hard"
            if g < self.clearance - float(config.NAV_RECOVER_TOL_MM):
                return "recover"
        return "recover" if self._fp_wall_inset(x, z) < 0.0 else "ok"

    def _in_exclusion(self, x, z):
        """True when map point (x, z) falls inside a detection no-target zone
        (an obstacle rectangle grown by the clear margin). Used to invalidate
        detections that project onto a spot where a target can never be."""
        for (x1, z1, x2, z2) in self._exclusion:
            if x1 <= x <= x2 and z1 <= z <= z2:
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
        n = int(max_d / step)
        # Same candidate set and the same cost function as before (changing
        # standoff is cheap, moving sideways is expensive -- see the formula
        # below), but evaluated in ASCENDING cost order and returned on the
        # first one that turns out to be free. _center_free() is the
        # expensive part (it walks every obstacle) and used to be called for
        # all ~(2n+1)^2 candidates whenever the ideal spot was blocked; now
        # it is only called until a usable candidate is found, which for a
        # lightly-obstructed arena is typically the first few checked.
        # Ranking is computed for every candidate up front and _center_free()
        # is the only thing skipped early, so the candidate this returns is
        # IDENTICAL to the old exhaustive arg-min search -- just reached
        # without necessarily visiting every candidate.
        candidates = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                dx, dz = i * step, j * step
                if dx == 0 and dz == 0:
                    continue
                if gz + dz < tz + smin:     # would crowd the person
                    continue
                cost = 3.0 * abs(dx) + (1.5 * -dz if dz < 0 else 1.0 * dz)
                candidates.append((cost, dx, dz))
        candidates.sort(key=lambda c: c[0])   # stable: ties keep this scan's order
        for _cost, dx, dz in candidates:
            cx, cz = gx + dx, gz + dz
            if self._center_free(cx, cz):
                return {"x": cx, "z": cz, "adjusted": True}
        return None

    # ------------------------------------------------------------ target projection
    def _track_loop(self):
        """Background home of the stateful target tracker (see
        _update_target_tracking): runs it on ONE dedicated cadence
        (NAV_TRACK_UPDATE_S) so the running-average window is advanced
        exactly once per detector frame,
        however many places want to read the result. _monitor_loop,
        _follow_loop and the UI (via state()) all just read the cached
        self._target through project_detection() below instead of each
        re-running the tracker themselves at their own cadence."""
        interval = float(config.NAV_TRACK_UPDATE_S)
        if interval <= 0.0:
            interval = 0.1
        while not self._stop.is_set():
            try:
                self._update_target_tracking()
            except Exception:
                pass
            self._stop.wait(interval)

    def project_detection(self):
        """Current tracked target (see _track_loop / _update_target_tracking):
        a cheap read of the shared cache, NOT a re-run of the tracker. This is
        what navigation, FOLLOW and the UI all call. Returns the target dict
        or None."""
        with self._lock:
            return dict(self._target) if self._target else None

    def _update_target_tracking(self):
        """Advance the running-average target by at most ONE detection frame.

        Runs on its own cadence via _track_loop -- see project_detection()
        for the cheap accessor everything else should call instead.

        Only a NEW detector frame that carries a FRESH detection (detector.py
        target["fresh"]: a real model box / a real radar cluster this frame,
        not a coasted box or a held radar lock) and projects to a valid map
        position is added to the window. Frames without a detection do NOT
        advance it, so the window is always "the last N frames that contained
        a detection", ending at the current one. The estimate uses whatever is
        in the window (1..N): it never waits for the window to fill."""
        if self._detector is None:
            return None
        now = time.time()
        forget = float(config.NAV_TARGET_FORGET_S)
        with self._lock:
            # Long loss: the window holds a position the person has likely
            # left. Clear it, but keep that position as a PRIOR (last known
            # position): the next detection near it is accepted at once, one
            # far from it has to be corroborated (see _accept_sample).
            if self._det_buf and now - self._det_buf[-1]["t"] > forget:
                self._set_prior_locked(self._det_buf[-1]["t"])
                self._det_buf.clear()
                self._accum = {"phase": "none", "n": 0,
                               "need": int(config.NAV_TARGET_AVG_N)}
        tel, seq = self._detector.telemetry.get()
        if not tel:
            return None
        with self._lock:
            if seq == self._last_tel_seq:
                return None                 # same detector frame as last time
            self._last_tel_seq = seq
        tgt = tel.get("target")
        if not tgt:
            return None
        # Detectors that predate the freshness flag: treat every frame with a
        # target as a detection frame (the old behaviour).
        if not tgt.get("fresh", True):
            return None
        az = tgt.get("raw_az") if tgt.get("raw_az") is not None else tgt.get("az")
        rng = tgt.get("raw_range") if tgt.get("raw_range") is not None else tgt.get("range")
        if rng is None or az is None:
            return None
        # Radar is only rejected here while moving if the detector has no
        # lock to maintain (it clears the target in that case). An ALREADY-
        # acquired lock is kept live by the detector's tracker through the
        # rover's own motion (purely spatial, no Doppler) -- see radar_tracker.py.
        p = self.pose()
        if p is None:
            return None
        rng_mm = float(rng) * 1000.0
        # Bearing and range are measured from the COLOUR CAMERA, which sits at
        # its own lever arm from the rover's turn centre. Projecting from the
        # rover centre makes a stationary target appear to swing every time the
        # rover rotates, by the camera's arc. Walk out to the camera first.
        cx, cz = self._camera_map_xz(p)
        bearing = math.radians(p["yaw_deg"] + float(az))
        tx = cx + rng_mm * math.sin(bearing)
        tz = cz - rng_mm * math.cos(bearing)
        # Clamp into the map so a noisy range can't paint the marker off-canvas.
        tx = max(0.0, min(self.map_w, tx))
        tz = max(0.0, min(self.map_d, tz))
        # A detection inside a no-target zone (obstacle + clear margin) cannot
        # be a real target: it is not a valid detection frame and does not
        # enter the window. The previous target simply holds.
        if self._in_exclusion(tx, tz):
            return None
        is_radar = tgt.get("source") == "radar"
        snr = (float(tgt["snr_peak"]) if (is_radar and tgt.get("snr_peak") is not None)
               else None)
        with self._lock:
            if not self._accept_sample_locked({"t": now, "x": tx, "z": tz, "snr": snr}):
                return None             # outside the prior's gate, still confirming
            self._last_fresh_t = now
            n_want = max(1, int(config.NAV_TARGET_AVG_N))
            window = list(self._det_buf)[-n_want:]
            est = _robust_average(window,
                                  float(config.NAV_TARGET_INLIER_MM),
                                  float(config.NAV_TARGET_JUMP_MM))
            self._accum = {"phase": "averaging", "n": len(window), "need": n_want,
                           "spread_mm": round(est["spread"]),
                           "inliers": est["inliers"]}
            t = {"x": est["x"], "z": est["z"], "t": now,
                 "range_m": round(float(rng), 2),
                 "az_deg": float(az), "source": tgt.get("source"),
                 "spread_mm": round(est["spread"]), "n": len(window)}
            self._target = t
        return t

    # ------------------------------------------------------------ last known position (prior)
    def _set_prior_locked(self, t_seen):
        """Remember the current target estimate as the last known position.
        Lock held."""
        if self._target is not None and self._target.get("x") is not None:
            self._prior = {"x": float(self._target["x"]), "z": float(self._target["z"]),
                           "t": float(t_seen)}
            self._prior_pending = []

    def _prior_gate_mm(self, now):
        """Radius around the last known position that a person could have
        reached by now: R0 + walking speed x time since last seen, capped."""
        age = max(0.0, now - self._prior["t"])
        return min(float(config.NAV_TARGET_PRIOR_R0_MM)
                   + float(config.NAV_TARGET_PRIOR_SPEED_MM_S) * age,
                   float(config.NAV_TARGET_PRIOR_MAX_MM))

    def _accept_sample_locked(self, q):
        """Add a projected detection to the averaging window, applying the
        last-known-position gate when the window is empty. Lock held.
        Returns True when the sample entered the window.

          * no prior, or the window already has samples -> accept (the robust
            average itself handles outliers);
          * window empty and the sample lies INSIDE the prior's gate -> accept
            from this single frame, prior consumed;
          * window empty and OUTSIDE the gate (a far jump: could be a ghost) ->
            held as pending until NAV_TARGET_PRIOR_CONFIRM_N mutually
            consistent frames agree, then all of them enter the window."""
        prior = self._prior
        if prior is None or self._det_buf:
            self._det_buf.append(q)
            return True
        now = q["t"]
        if math.hypot(q["x"] - prior["x"], q["z"] - prior["z"]) <= self._prior_gate_mm(now):
            self._prior, self._prior_pending = None, []
            self._det_buf.append(q)
            return True
        jump = float(config.NAV_TARGET_JUMP_MM)
        max_age = max(2.0, 2.0 * float(config.NAV_FOLLOW_LOOK_MAX_S))
        pend = [p for p in self._prior_pending if now - p["t"] <= max_age]
        if pend and math.hypot(q["x"] - pend[-1]["x"], q["z"] - pend[-1]["z"]) > jump:
            pend = []                       # inconsistent with the run: restart it
        pend.append(q)
        need = max(1, int(config.NAV_TARGET_PRIOR_CONFIRM_N))
        if len(pend) < need:
            self._prior_pending = pend
            self._accum = {"phase": "confirming", "n": len(pend), "need": need}
            return False
        self._prior, self._prior_pending = None, []
        for p in pend:
            self._det_buf.append(p)
        return True

    # ------------------------------------------------------------ FOLLOW corner look
    def _corner_check(self, i, waypoints):
        """Called by _drive at a corner (rover stopped, before leg i). FOLLOW
        only. Returns "replan" to end this drive here so the follow loop
        re-plans from the corner, else "continue".

        1. Deferred re-plan requested (target moved mid-leg, or cycle cap) ->
           replan now, without stopping mid-straight.
        2. Blind-time limit: look when the time since the last accepted
           detection (ANY sensor) is >= NAV_FOLLOW_BLIND_S, or would exceed it
           by the end of the next leg.
        3. Look: wait up to NAV_FOLLOW_LOOK_MAX_S for a fresh detection, exiting
           as soon as one is accepted.
        4. Re-plan only if needed: standoff goal shifted > NAV_GOAL_TOLERANCE_MM,
           or target moved >= NAV_FOLLOW_REPLAN_MM, or it switched sides —
           and not within NAV_FOLLOW_REPLAN_MIN_INTERVAL_S of the last one."""
        with self._lock:
            follow = self._follow
            ctx = dict(self._follow_ctx) if self._follow_ctx else None
            req = self._corner_replan_req
            last_fresh = self._last_fresh_t
        if not follow or ctx is None:
            return "continue"
        now = time.time()
        if req:
            with self._lock:
                self._corner_replan_req = False
                self._last_replan_t = now
            return "replan"
        p = self.pose()
        if p is None:
            return "continue"
        # Remaining path length from the live pose through the rest of the plan.
        remain, px, pz = 0.0, p["x"], p["z"]
        for (wx, wz) in waypoints[i:]:
            remain += math.hypot(wx - px, wz - pz)
            px, pz = wx, wz
        if remain < float(config.NAV_FOLLOW_LOOK_MIN_REMAIN_MM):
            return "continue"             # the arrival look covers it
        blind = (now - last_fresh) if last_fresh is not None else float("inf")
        wx, wz = waypoints[i]
        speed = max(float(self.rover.cfg.MAX_LINEAR), 0.05)          # m/s
        next_leg_s = math.hypot(wx - p["x"], wz - p["z"]) / 1000.0 / speed * 1.2 + 1.0
        limit = float(config.NAV_FOLLOW_BLIND_S)
        if blind < limit and blind + next_leg_s <= limit:
            return "continue"
        # ---- look ----
        look_t0 = time.time()
        with self._lock:
            # Samples from before the stop are stale: move the estimate into
            # the prior so a detection now is judged against the last known
            # position rather than out-voted by old frames.
            if self._det_buf:
                self._set_prior_locked(self._det_buf[-1]["t"])
                self._det_buf.clear()
            elif self._prior is None:
                self._set_prior_locked(self._last_fresh_t or look_t0)
            self._look = {"state": "looking", "since": round(look_t0, 2),
                          "blind_s": round(blind, 1) if blind != float("inf") else None}
            self._message = "follow: looking at corner (blind %.1f s)…" % min(blind, 999.0)
        cap = float(config.NAV_FOLLOW_LOOK_MAX_S)
        got = False
        while time.time() - look_t0 < cap:
            if self._cancel.is_set() or self._stop.is_set():
                return "continue"
            with self._lock:
                got = self._last_fresh_t is not None and self._last_fresh_t >= look_t0
            if got:
                break
            self._stop.wait(0.05)
        look_s = round(time.time() - look_t0, 2)
        if not got:
            with self._lock:
                self._look = {"state": "timeout", "look_s": look_s}
                self._message = "follow: nothing seen at corner, continuing to last known position"
            return "continue"
        t = self.project_detection()
        if t is None:
            return "continue"
        # ---- re-plan only when needed ----
        moved = math.hypot(t["x"] - ctx["tx"], t["z"] - ctx["tz"])
        side_old, side_new = ctx["tx"] - p["x"], t["x"] - p["x"]
        switched = (side_old * side_new < 0
                    and abs(side_new - side_old) >= float(config.NAV_FOLLOW_SIDE_SWITCH_MM))
        new_goal = self.compute_goal(t["x"], t["z"])
        with self._lock:
            cur_goal = dict(self._goal) if self._goal else None
        goal_shift = (new_goal is not None and cur_goal is not None
                      and math.hypot(new_goal["x"] - cur_goal["x"], new_goal["z"] - cur_goal["z"])
                      > float(config.NAV_GOAL_TOLERANCE_MM))
        need = goal_shift or moved >= float(config.NAV_FOLLOW_REPLAN_MM) or switched
        recent = time.time() - self._last_replan_t < float(config.NAV_FOLLOW_REPLAN_MIN_INTERVAL_S)
        with self._lock:
            if need and not recent:
                self._look = {"state": "replan", "look_s": look_s, "moved_mm": round(moved)}
                self._last_replan_t = time.time()
                return "replan"
            self._look = {"state": "unchanged", "look_s": look_s, "moved_mm": round(moved)}
            self._message = "follow: target unchanged, continuing"
        return "continue"

    # ------------------------------------------------------------ navigation
    def navigate_to_detection(self):
        """Plan + drive to the standoff goal of the CURRENT detection target."""
        t = self.project_detection()
        if t is None:
            return False, "no detection target with range available"
        return self.navigate_to_target(t["x"], t["z"])

    def navigate_to_target(self, tx, tz, follow=False):
        """Plan + drive to the standoff goal of a target at map (tx, tz).
        follow=True (FOLLOW loop only) enables the FOLLOW path preference and
        the corner checks in _drive."""
        goal = self.compute_goal(float(tx), float(tz))
        with self._lock:
            self._follow_ctx = {"tx": float(tx), "tz": float(tz)} if follow else None
            self._corner_replan_req = False
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
        return self._dispatch(goal, follow=follow)

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
        # The start cell may be flush with the arena edge (e.g. parked in a
        # corner); as a GOAL it is moved in far enough to keep wall_margin.
        m = self.car_w / 2.0 + self.wall_margin, self.car_l / 2.0 + self.wall_margin
        gx = min(max(float(self._map_start[0]), m[0]), self.map_w - m[0])
        gz = min(max(float(self._map_start[1]), m[1]), self.map_d - m[1])
        with self._lock:
            self._target = {"x": gx, "z": gz, "t": time.time(),
                            "range_m": None, "az_deg": None, "source": "home"}
        return self._dispatch({"x": gx, "z": gz, "adjusted": False})

    def _plan_segments(self, plan_cfg, start, end, follow):
        """Plan a path; for FOLLOW, add corners to look from (see
        _staircase). Returns (segs, achieved_clearance, info)."""
        # The rover may start (or stop short) a little inside the clearance
        # zone without being in "recover" territory: allow endpoints down to
        # the recover trigger, so such a pose is plannable instead of refused.
        plan_cfg = dict(plan_cfg, endpoint_clearance=(
            self.clearance - float(config.NAV_RECOVER_TOL_MM)))
        segs, achieved = plan_rectilinear_path_ex(
            self.map, plan_cfg, start, end, ignore_start_obstacle=self._ignore_obstacles)
        info = {"staircase": False}
        if not follow or segs is None or self._ignore_obstacles:
            return segs, achieved, info
        segs2 = self._staircase(start, segs)
        if len(segs2) > len(segs):
            info["staircase"] = True
        return segs2, achieved, info

    def _staircase(self, start, segs):
        """More corners without extra distance. The planner always prefers the
        fewest turns (on a grid, a path with more turns is never shorter, so
        any positive turn_penalty picks the L-shape). For FOLLOW, each L-shaped
        pair of legs (one along x, the next along z, or vice versa) where a leg
        is longer than NAV_FOLLOW_MAX_LEG_MM is re-cut into a staircase of k
        alternating steps: same start, same end, same total length, but a
        corner every ~MAX_LEG where the rover can stop and look.

        A staircase is used only when its steps keep the rover footprint at
        least as far from every obstacle as the original pair of legs did
        (min footprint_gap); otherwise fewer steps are tried, and failing that
        the original pair is kept. Steps stay inside the pair's bounding box,
        so the arena margin is kept automatically. Steps shorter than NAV_FOLLOW_MIN_STEP_MM are
        never produced. A single long straight leg with no perpendicular
        neighbour is left as is (it cannot gain a corner without a detour)."""
        max_leg = float(config.NAV_FOLLOW_MAX_LEG_MM)
        min_step = float(config.NAV_FOLLOW_MIN_STEP_MM)
        if max_leg <= 0:
            return segs

        def advance(pt, steps):
            x, z = pt
            for ax, d in steps:
                if ax == "x":
                    x += d
                else:
                    z += d
            return (x, z)

        def min_gap(pt, steps):
            x, z, g = pt[0], pt[1], float("inf")
            for ax, d in steps:
                nx, nz = (x + d, z) if ax == "x" else (x, z + d)
                g = min(g, footprint_gap(x, z, nx, nz, self._obstacle_rects,
                                         self.car_w, self.car_l))
                x, z = nx, nz
            return g

        out, i, cur = [], 0, (float(start[0]), float(start[1]))
        while i < len(segs):
            ax, d = segs[i]
            if i + 1 < len(segs) and segs[i + 1][0] != ax:
                bx_, e = segs[i + 1]
                big, small = max(abs(d), abs(e)), min(abs(d), abs(e))
                k_want = int(math.ceil(big / max_leg)) if big > max_leg else 1
                k_max = int(small // min_step) if min_step > 0 else k_want
                chosen = None
                floor = min_gap(cur, [segs[i], segs[i + 1]])
                for k in range(min(k_want, k_max), 1, -1):
                    steps = [(ax, d / k), (bx_, e / k)] * k
                    if min_gap(cur, steps) >= floor - 1e-6:
                        chosen = steps
                        break
                steps = chosen if chosen else [segs[i], segs[i + 1]]
                out.extend(steps)
                cur = advance(cur, steps)
                i += 2
            else:
                out.append(segs[i])
                cur = advance(cur, [segs[i]])
                i += 1
        return out

    def _dispatch(self, goal, follow=False):
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return False, "a navigation is already running (cancel it first)"
        p = self.pose()
        if p is None:
            return False, "no rover pose yet"
        if math.hypot(goal["x"] - p["x"], goal["z"] - p["z"]) <= float(config.NAV_GOAL_TOLERANCE_MM):
            with self._lock:
                self._goal, self._status, self._message = goal, "arrived", "already at the goal"
            return True, "already at the goal"
        with self._lock:
            self._goal = goal
            self._status, self._message = "planning", ""
            self._path, self._leg = None, 0
            self._drive_outcome = None
        state = self._classify(p["x"], p["z"])
        if state == "hard":
            with self._lock:
                self._status = "blocked"
                self._message = ("rover footprint is within %.0f mm of an obstacle; use "
                                 "Ignore obstacles to drive out" % float(config.NAV_OBSTACLE_STOP_MARGIN_MM))
            return False, self._message
        wps = None
        if state == "ok":
            wps = self._plan_waypoints(goal, follow, p)
            if wps is None:
                return False, self._message
        else:
            # Inside a clearance zone or partly outside the arena: the drive
            # thread first returns to a safe pose, then plans (see _drive).
            with self._lock:
                self._status, self._message = "moving", "returning to a safe position before planning"
        self._cancel.clear()
        self._nav_thread = threading.Thread(target=self._drive, args=(goal, follow, wps),
                                            daemon=True)
        self._nav_thread.start()
        return True, (f"navigating: {len(wps)} leg(s)" if wps else "navigating (recovering first)")

    def _plan_waypoints(self, goal, follow, p):
        """Plan from pose p to goal. Publishes the path for the UI and returns
        absolute map waypoints, or None (status/message set) when no path.

        The planner maximises the route's clearance (see rectilinear_mm).
        Odometry error since the last known-good pose earns extra planning
        margin, applied as a raised clearance FLOOR; if that margin alone makes
        the goal unreachable, the configured clearance is used instead."""
        start = (round(p["x"]), round(p["z"]))
        end = (round(goal["x"]), round(goal["z"]))
        drift_mm = self._drift_margin_mm()
        plan_cfg = self.plan_cfg
        if drift_mm > 0.0:
            plan_cfg = dict(plan_cfg)
            plan_cfg["clearance"] = float(plan_cfg.get("clearance", 0)) + drift_mm
        segs, achieved, pinfo = self._plan_segments(plan_cfg, start, end, follow)
        if segs is None and drift_mm > 0.0:
            segs, achieved, pinfo = self._plan_segments(self.plan_cfg, start, end, follow)
        if segs is None:
            with self._lock:
                self._status = "no_path"
                self._message = "planner found no obstacle-free rectilinear path"
                self._plan_clearance = None
            return None
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
            self._plan_info = dict(pinfo, legs=len(wps),
                                   longest_mm=round(max(abs(float(d)) for _, d in segs)))
        return wps

    # ------------------------------------------------------------ driving + safety
    def _to_body(self, dx, dz, yaw_deg):
        """Map-frame delta (mm) -> the move's own start-heading frame
        (right, forward), so a move lands on its map target even when the
        rover's heading has drifted."""
        g = math.radians(yaw_deg)
        mr, mf = dx, -dz                  # map delta as (right, forward)
        return (mr * math.cos(g) + mf * math.sin(g),
                -mr * math.sin(g) + mf * math.cos(g))

    def _halt(self, lp):
        """Hard stop: the footprint came within NAV_OBSTACLE_STOP_MARGIN_MM of
        an obstacle. The only case that still halts."""
        self.rover.stop()
        self._cancel.set()
        with self._lock:
            self._status = "blocked"
            self._message = ("E-STOP: rover footprint %.0f mm from an obstacle at (%.0f, %.0f) mm"
                             % (max(0.0, self._fp_obstacle_gap(lp["x"], lp["z"])),
                                lp["x"], lp["z"]))

    def _run_move(self, right_mm, forward_mm, recover_ok):
        """Drive one straight move while polling the footprint at ~20 Hz.
        Returns (outcome, result): "done" (result = MoveResult), "cancel",
        "hard" (halted, status set), or "recover" (stopped because the
        footprint entered a clearance zone / left the arena; only when
        recover_ok -- a recovery move itself is exempt, as it starts there)."""
        # Hold true map-forward (the T265 yaw that == anchor heading) so the
        # rover corrects accumulated yaw drift WHILE driving.
        with self._lock:
            hold_yaw = self._anchor_pose["yaw"] if self._anchor_pose else None
        mv = self.rover.move(right=right_mm, forward=forward_mm, units="mm",
                             hold_yaw=hold_yaw, blocking=False)
        while self.rover.is_busy():
            if self._cancel.is_set() or self._stop.is_set():
                self.rover.stop()
                return "cancel", None
            lp = self.pose()
            if lp is not None:
                st = self._classify(lp["x"], lp["z"])
                if st == "hard":
                    self._halt(lp)
                    return "hard", None
                if st == "recover" and recover_ok:
                    self.rover.stop()
                    return "recover", None
            self._stop.wait(0.05)   # ~20 Hz safety poll
        return "done", self.rover.wait(mv)

    def _recovery_target(self, x, z):
        """Shortest straight move out of a clearance zone and/or back inside
        the arena: a target whose footprint is >= clearance + NAV_RECOVER_TOL_MM
        from every obstacle and >= wall_margin inside the arena, reached by a
        straight move whose sweeping footprint never comes within
        NAV_OBSTACLE_STOP_MARGIN_MM of an obstacle (exact: the centre segment
        vs obstacles grown by the car half-extents + that margin). Searches 16
        directions in 10 mm steps up to NAV_RECOVER_MAX_MM, axis-aligned first
        on ties. Returns (tx, tz) or None."""
        clear_t = self.clearance + float(config.NAV_RECOVER_TOL_MM)
        hard = float(config.NAV_OBSTACLE_STOP_MARGIN_MM)
        hx, hz = self.car_w / 2.0 + hard, self.car_l / 2.0 + hard
        grown = [] if self._ignore_obstacles else [
            (x1 - hx, z1 - hz, x2 + hx, z2 + hz) for (x1, z1, x2, z2) in self._obstacle_rects]
        dirs = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                for a in (0, 90, 180, 270, 45, 135, 225, 315,
                          22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5)]
        step = 10.0
        for k in range(1, int(float(config.NAV_RECOVER_MAX_MM) / step) + 1):
            d = k * step
            for ux, uz in dirs:
                tx, tz = x + ux * d, z + uz * d
                if self._fp_wall_inset(tx, tz) < self.wall_margin:
                    continue
                if not self._ignore_obstacles and self._fp_obstacle_gap(tx, tz) < clear_t:
                    continue
                if any(_seg_hits(x, z, tx, tz, b) for b in grown):
                    continue
                return tx, tz
        return None

    def _recover(self, p):
        """Leave a clearance zone / return inside the arena by the shortest
        straight move, then report whether the rover is back in a safe pose.
        On failure the status says why and the navigation ends."""
        x, z = p["x"], p["z"]
        inset = self._fp_wall_inset(x, z)
        why = ("returning inside the arena (%.0f mm outside)" % -inset if inset < 0.0 else
               "leaving obstacle clearance zone (%.0f mm from obstacle)"
               % self._fp_obstacle_gap(x, z))
        target = self._recovery_target(x, z)
        if target is None:
            self.rover.stop()
            self._cancel.set()
            with self._lock:
                self._status = "blocked"
                self._message = "%s: no clear move within %.0f mm" % (
                    why, float(config.NAV_RECOVER_MAX_MM))
            return False
        with self._lock:
            self._message = why
        cr, cf = self._to_body(target[0] - x, target[1] - z, p["yaw_deg"])
        out, res = self._run_move(cr, cf, recover_ok=False)
        if out != "done":
            return False
        if res is None or not res:
            with self._lock:
                self._status = "error"
                self._message = "%s: move failed (%s)" % (
                    why, res.reason if res is not None else "no result")
            return False
        return True

    def _drive(self, goal, follow, waypoints):
        """Navigation thread: drive the plan; whenever the footprint enters a
        clearance zone or leaves the arena, stop, return to a safe pose by the
        shortest move, re-plan from there to the SAME goal, and continue.
        At most NAV_RECOVER_MAX_TRIES recoveries per navigation."""
        try:
            recovers = 0
            while True:
                if waypoints is None:
                    p = self.pose()
                    if p is None:
                        raise RuntimeError("lost rover pose")
                    state = self._classify(p["x"], p["z"])
                    if state == "hard":
                        self._halt(p)
                        return
                    if state == "recover":
                        if recovers >= int(config.NAV_RECOVER_MAX_TRIES):
                            self.rover.stop()
                            with self._lock:
                                self._status = "blocked"
                                self._message = ("gave up after %d recoveries; still outside "
                                                 "a safe position" % recovers)
                            return
                        recovers += 1
                        if not self._recover(p):
                            return
                        continue
                    waypoints = self._plan_waypoints(goal, follow, p)
                    if waypoints is None:
                        return
                if self._drive_legs(waypoints) != "recover":
                    return
                waypoints = None
        except Exception as exc:
            with self._lock:
                self._status, self._message = "error", str(exc)

    def _drive_legs(self, waypoints):
        """Drive the planned legs. Returns "arrived", "replan" (FOLLOW corner),
        "recover" (safety zone entered: _drive recovers and re-plans),
        "cancel", "hard", or "error"."""
        driven = 0                      # legs actually driven so far
        for i, (wx, wz) in enumerate(waypoints):
            with self._lock:
                self._leg = i + 1
            # Re-derive this leg from the LIVE pose so per-move arrival
            # error does not accumulate across legs.
            p = self.pose()
            if p is None:
                raise RuntimeError("lost rover pose")
            state = self._classify(p["x"], p["z"])
            if state == "hard":
                self._halt(p)
                return "hard"
            if state == "recover":
                return "recover"
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
                return "cancel"
            # FOLLOW corner check: the rover is stopped at a corner of the
            # plan, the only place it may pause. Look again if it has been
            # blind too long, and end this drive here if the target has
            # moved enough to need a new plan (see _corner_check).
            if driven > 0 and self._corner_check(i, waypoints) == "replan":
                with self._lock:
                    self._drive_outcome = "replan"
                    self._status = "arrived"
                    self._message = "follow: re-planning at corner"
                return "replan"
            # Zero-velocity drift sample: the rover is stopped between legs,
            # so its TRUE velocity is zero and any pose change the T265
            # reports here is drift, measured directly.
            self._zupt_sample()
            # Absolute (tag) correction is no longer forced HERE. It runs
            # as its own continuous background service (_monitor_loop),
            # applying whenever the rover happens to be stationary rather
            # than being woven into this leg's critical path. Navigation
            # just reads self.pose() -- above and on the next leg -- which
            # already carries whatever the localisation service has
            # applied so far. Same gates, same consensus/RMS/spread
            # thresholds, same eased-in correction: only WHEN it runs has
            # changed, never what it does or how cautious it is.
            # Tracking-confidence gate: Low confidence is exactly when VIO
            # drift accrues fastest. Wait briefly for it to recover; if it
            # doesn't, still go (never strand a demo) but at reduced speed.
            self._await_confidence()
            # One mecanum move per waypoint, in the move's start-heading frame
            # and holding true map-forward (see _to_body / _run_move). The
            # footprint is checked at ~20 Hz while it drives.
            cr, cf = self._to_body(dx, dz, p["yaw_deg"])
            out, res = self._run_move(cr, cf, recover_ok=True)
            if out != "done":
                return out
            if self._cancel.is_set():
                return "cancel"
            # Per-leg fidelity log: commanded vs achieved displacement. A
            # consistent ratio across many legs indicates a fixed VIO SCALE
            # error (correctable); random scatter indicates slip or noise.
            self._log_leg(i + 1, leg_mm, wx, wz, res)
            driven += 1
            if res is None or not res:
                reason = res.reason if res is not None else "no result"
                with self._lock:
                    self._status = "error"
                    self._message = (f"leg {i + 1} ({dx:+.0f}, {dz:+.0f})mm "
                                     f"failed ({reason})")
                    # Clear slate: the next navigate call starts fresh.
                    self._goal, self._path, self._leg = None, None, 0
                return "error"
        with self._lock:
            self._status, self._message = "arrived", ""
        return "arrived"

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
        (rectilinear_mm's start-blocked check), and the mid-move obstacle
        checks (hard halt and clearance-zone recovery, see _classify) are off.
        Keeping the footprint inside the arena is NOT affected. Obstacle
        avoidance elsewhere (routing around every obstacle, the standoff-goal
        search) is unaffected. Takes effect immediately, including mid-move."""
        with self._lock:
            self._ignore_obstacles = bool(enabled)
        return self._ignore_obstacles

    # ------------------------------------------------------------ running-average window
    def set_avg_n(self, n):
        """Operator-settable running-average window: how many of the most
        recent DETECTION FRAMES the target estimate averages over (see
        _update_target_tracking). Clamped to [NAV_TARGET_AVG_N_MIN,
        NAV_TARGET_AVG_N_MAX]. Takes effect on the next detection frame; the
        buffer already holds up to the max, so a larger N uses existing history
        immediately rather than waiting. Returns the value actually applied."""
        lo = int(config.NAV_TARGET_AVG_N_MIN)
        hi = int(config.NAV_TARGET_AVG_N_MAX)
        applied = max(lo, min(hi, int(n)))
        with self._lock:
            config.NAV_TARGET_AVG_N = applied
            if self._accum.get("phase") != "none":
                self._accum["need"] = applied
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
                # Fresh lock: empty the running-average window so follow locks
                # onto wherever the person is right now (no carry-over from
                # detections made before follow was switched on).
                self._det_buf.clear()
                self._prior, self._prior_pending = None, []
                self._corner_replan_req = False
                self._accum = {"phase": "none", "n": 0,
                               "need": int(config.NAV_TARGET_AVG_N)}

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
        side_switch_mm = float(config.NAV_FOLLOW_SIDE_SWITCH_MM)
        still_since = time.time()
        was_following = False
        # Once a radar (moving-point) target has cleared the stationary-dwell
        # check below, its lock is maintained by the detector's own tracker
        # purely spatially through the rover's own motion (see
        # project_detection / radar_tracker.py) -- it does not need to
        # re-earn stillness after every stop. Reset only when the track is
        # actually lost (a fresh acquisition, which DOES need to prove itself
        # stationary again).
        radar_confirmed = False
        while not self._stop.is_set():
            with self._lock:
                follow = self._follow
            if not follow:
                was_following = False
                radar_confirmed = False
                self._stop.wait(0.1)
                continue
            if not was_following:                  # just enabled -> fresh dwell
                was_following = True
                still_since = time.time()
                radar_confirmed = False
            # ---------------- SENSE PHASE (must be stationary) ----------------
            if self.rover.is_busy():
                still_since = time.time()          # still moving -> reset the dwell
                self._stop.wait(0.1)
                continue
            t = self.project_detection()
            if t is None:
                radar_confirmed = False   # track lost -> next lock must re-earn stillness
                self._stop.wait(idle)
                continue
            # HARD INVARIANT: a BRAND-NEW radar/moving-point acquisition is accepted
            # ONLY once the rover has been confirmed stationary long enough to have
            # re-acquired a moving cluster while still. Vision (RGB/thermal) is
            # unconstrained. An already-confirmed radar lock is exempt: it is kept
            # live by the detector's tracker purely spatially through the rover's
            # own motion (no Doppler involved), the same as any other live source,
            # so it does not need to wait out this dwell again after every stop.
            if (t.get("source") == "radar" and not radar_confirmed
                    and (time.time() - still_since) < still_confirm):
                with self._lock:
                    if self._status not in ("moving", "planning"):
                        self._status = "idle"
                        self._message = "follow: confirming stationary mmWave lock…"
                self._stop.wait(0.1)
                continue
            if t.get("source") == "radar":
                radar_confirmed = True
            # ---------------- COMMIT + MOVE PHASE ----------------
            ok, _msg = self.navigate_to_target(t["x"], t["z"], follow=True)
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
                if nt is not None:
                    moved_mm = math.hypot(nt["x"] - gx, nt["z"] - gz)
                    # Receding-horizon rule: a leg already under way stays
                    # alive through small target motion -- only cut it short
                    # when the move is BIG (moved_mm >= replan_mm) or the
                    # target crossed to the rover's other side (walked from
                    # the left half-plane to the right, or vice versa), which
                    # means the leg in progress now heads toward the wrong
                    # side however small the raw distance is. The minimum-
                    # swing guard keeps noise near dead-centre from flapping
                    # this.
                    p_now = self.pose()
                    switched_side = False
                    if p_now is not None:
                        side_old = gx - p_now["x"]
                        side_new = nt["x"] - p_now["x"]
                        switched_side = (side_old * side_new < 0
                                        and abs(side_new - side_old) >= side_switch_mm)
                    # Only a side switch or a VERY large move stops the rover
                    # mid-leg. An ordinary move (>= replan_mm) is deferred to
                    # the next corner, where _drive ends the plan cleanly.
                    if switched_side or moved_mm >= float(config.NAV_FOLLOW_MIDLEG_REPLAN_MM):
                        self.cancel()
                        with self._lock:
                            self._status = "moving"
                            self._message = ("follow: target crossed sides, re-targeting"
                                             if switched_side else
                                             "follow: target moved, re-targeting")
                        break
                    if moved_mm >= replan_mm:
                        with self._lock:
                            if not self._corner_replan_req:
                                self._corner_replan_req = True
                                self._message = "follow: target moved, re-planning at next corner"
                if time.time() >= deadline:
                    # Cycle cap: re-plan at the next corner rather than stopping
                    # mid-straight. A hard backstop (twice the cap) still
                    # cancels outright if no corner is ever reached.
                    with self._lock:
                        if not self._corner_replan_req:
                            self._corner_replan_req = True
                            self._message = "follow: cycle time cap reached, re-planning at next corner"
                    if time.time() >= deadline + cap:
                        self.cancel()
                        with self._lock:
                            self._status = "arrived"
                            self._message = "follow: cycle time cap reached, re-targeting"
                        break
                self._stop.wait(0.1)
            # Move ended -> rover is stopping; open a fresh stationary window so the
            # next radar lock must be re-confirmed while still.
            still_since = time.time()
            with self._lock:
                corner_replan = self._drive_outcome == "replan"
                self._drive_outcome = None
            if corner_replan:
                # The corner look just acquired the target while the rover was
                # stationary, so a radar lock does not need to re-earn stillness.
                radar_confirmed = True

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
            "wall_margin": float(self.wall_margin),
            "avg_n": int(config.NAV_TARGET_AVG_N),
            "avg_n_min": int(config.NAV_TARGET_AVG_N_MIN),
            "avg_n_max": int(config.NAV_TARGET_AVG_N_MAX),
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
            follow_info = {
                "blind_s": (round(time.time() - self._last_fresh_t, 1)
                            if self._last_fresh_t is not None else None),
                "look": dict(self._look),
                "corner_replan_pending": bool(self._corner_replan_req),
                "plan": dict(self._plan_info) if self._plan_info else None,
                "prior": ({"x": round(self._prior["x"]), "z": round(self._prior["z"]),
                           "gate_mm": round(self._prior_gate_mm(time.time()))}
                          if self._prior else None),
            }
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
            # mm: the smallest footprint-to-obstacle distance along this route.
            # The planner maximises it and never plans below config.json's
            # `clearance` (except right at a start/goal pose, down to the
            # recover trigger).
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
            "follow_info": follow_info,
            "status": status,
            "message": message,
            "auto": auto,
            "follow": follow,
            "ignore_obstacles": ignore_obstacles,
            "avg_n": int(config.NAV_TARGET_AVG_N),
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
