"""Central configuration for the rover UI backend.

All tunables live here so the demo can be re-targeted without code edits.
Override any value with an environment variable of the same name.

Motion is map-based: a pre-stored BEV map (map.json) with marked obstacle
regions, the detection target projected onto that map, a standoff goal 2 m on
the rover side of the target, a rectilinear obstacle-avoiding plan
(rectilinear_mm.py), and execution through the
purely_control.T265RoverService move_axis() interface.
"""
import os

# Repo root (holds map.json, config.json, rectilinear_mm.py, purely_control/)
DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name: str, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


# ---------------------------------------------------------------- server
HOST = _env("ROVER_UI_HOST", "0.0.0.0")
PORT = _env("ROVER_UI_PORT", 8000)

# ---------------------------------------------------------------- FLIR thermal (FLIR One Pro)
FLIR_DEVICE = _env("FLIR_DEVICE", "/dev/video1")
FLIR_WIDTH = _env("FLIR_WIDTH", 640)
FLIR_HEIGHT = _env("FLIR_HEIGHT", 480)
FLIR_FPS = _env("FLIR_FPS", 15)
FLIR_COLORMAP = _env("FLIR_COLORMAP", "INFERNO")

# ---------------------------------------------------------------- Intel RealSense D435i (RGB)
D435_SERIAL = _env("D435_SERIAL", "")
D435_WIDTH = _env("D435_WIDTH", 640)
D435_HEIGHT = _env("D435_HEIGHT", 480)
D435_FPS = _env("D435_FPS", 30)
D435_ENABLE_DEPTH = _env("D435_ENABLE_DEPTH", True)
D435_DEPTH_COLORMAP = _env("D435_DEPTH_COLORMAP", "JET")
D435_DEPTH_MAX_M = _env("D435_DEPTH_MAX_M", 6.0)

# ---------------------------------------------------------------- Intel RealSense T265 (pose)
# NOTE: the T265 is owned by purely_control.T265RoverService (the navigation
# executor), NOT by a rover_ui sensor thread — only ONE process-wide owner may
# open the device. These are kept for the (unused) T265Pose class.
T265_SERIAL = _env("T265_SERIAL", "")
T265_POSE_HZ = _env("T265_POSE_HZ", 30)

# ---------------------------------------------------------------- IWR6843ISK mmWave radar
MMWAVE_CLI_PORT = _env("MMWAVE_CLI_PORT", "/dev/ttyUSB0")
MMWAVE_DATA_PORT = _env("MMWAVE_DATA_PORT", "/dev/ttyUSB1")
MMWAVE_CLI_BAUD = _env("MMWAVE_CLI_BAUD", 115200)
MMWAVE_DATA_BAUD = _env("MMWAVE_DATA_BAUD", 921600)
# Radar cfg: clutterRemoval ON -> the radar reports MOVING returns only
# (static clutter is removed at the source). See iwr6843_moving_only.cfg.
MMWAVE_CONFIG_FILE = _env(
    "MMWAVE_CONFIG_FILE",
    os.path.join(DEMO_ROOT, "iwr6843_moving_only.cfg"),
)
MMWAVE_STALL_TIMEOUT = _env("MMWAVE_STALL_TIMEOUT", 3.0)

# ---------------------------------------------------------------- map navigation
# The pre-stored BEV map + planner config (units: mm).
MAP_FILE = _env("MAP_FILE", os.path.join(DEMO_ROOT, "map.json"))
PLAN_CONFIG_FILE = _env("PLAN_CONFIG_FILE", os.path.join(DEMO_ROOT, "config.json"))
# Standoff: the rover's goal is the target position shifted this many mm toward
# the bottom of the map (the rover side): goal = (target_x, target_z + STANDOFF).
# This is a SOFT preference — when it conflicts with an obstacle keep-out the
# navigator varies the distance (never below NAV_STANDOFF_MIN_MM from the
# person) and only then shifts sideways. Obstacle avoidance always wins.
NAV_STANDOFF_MM = _env("NAV_STANDOFF_MM", 2000)
NAV_STANDOFF_MIN_MM = _env("NAV_STANDOFF_MIN_MM", 1000)
# Upper bound for the OPERATOR-settable standoff (the UI field). The floor is
# NAV_STANDOFF_MIN_MM above, which is also the hard "never crowd the person"
# limit used when the goal has to be nudged around an obstacle.
NAV_STANDOFF_MAX_MM = _env("NAV_STANDOFF_MAX_MM", 4000)
# When the standoff goal's rover footprint overlaps an obstacle, search nearby
# free positions on a grid of this step, out to this max displacement.
NAV_ADJUST_STEP_MM = _env("NAV_ADJUST_STEP_MM", 100)
NAV_ADJUST_MAX_MM = _env("NAV_ADJUST_MAX_MM", 2500)
# Skip a re-navigation if the new goal is within this distance of where the
# rover already is / is already heading (don't twitch on detection jitter).
NAV_GOAL_TOLERANCE_MM = _env("NAV_GOAL_TOLERANCE_MM", 250)
# AUTO mode: re-navigate whenever a (stable) detection target appears. The
# target's projected map position must hold within NAV_AUTO_STABLE_MM for
# NAV_AUTO_STABLE_S seconds before a move is dispatched.
NAV_AUTO = _env("NAV_AUTO", False)
NAV_AUTO_STABLE_S = _env("NAV_AUTO_STABLE_S", 1.0)
NAV_AUTO_STABLE_MM = _env("NAV_AUTO_STABLE_MM", 400)
NAV_AUTO_COOLDOWN_S = _env("NAV_AUTO_COOLDOWN_S", 2.0)
# FOLLOW mode: a continuous "navigate to detection" loop. Each cycle re-projects
# the LIVE detection and drives to its standoff goal. The move is allowed to RUN
# TO COMPLETION (no fixed-timer interruption) so the rover doesn't lurch to a
# halt mid-path; it is re-planned early ONLY when the target has actually moved
# more than NAV_FOLLOW_REPLAN_MM (the person walked), and NAV_FOLLOW_CYCLE_S is a
# generous stuck-backstop (per-leg timeouts already break genuine hangs).
# Mutually exclusive with AUTO. When there's nothing to drive (already at the
# goal / no target), it idles NAV_FOLLOW_MIN_INTERVAL_S before re-checking.
NAV_FOLLOW = _env("NAV_FOLLOW", False)
NAV_FOLLOW_CYCLE_S = _env("NAV_FOLLOW_CYCLE_S", 20.0)      # stuck-backstop only
NAV_FOLLOW_MIN_INTERVAL_S = _env("NAV_FOLLOW_MIN_INTERVAL_S", 1.0)
NAV_FOLLOW_REPLAN_MM = _env("NAV_FOLLOW_REPLAN_MM", 300)   # re-plan when target moves this far (mm)
# Strict move-stop-sense gate: a radar (moving-point) target is accepted only
# after the rover has been confirmed STATIONARY this long — enough for the
# detector to re-acquire a moving cluster while still (>= ~RADAR_ACCUM_SEC plus
# settle). Raise if the rover commits to a radar target before it has truly
# stopped; lower for snappier hops. (Vision targets are not gated by this.)
NAV_FOLLOW_STILL_CONFIRM_S = _env("NAV_FOLLOW_STILL_CONFIRM_S", 1.0)
# How long a projected detection target stays shown on the map after the
# detection drops (purely cosmetic; navigation uses the live value).
NAV_TARGET_HOLD_S = _env("NAV_TARGET_HOLD_S", 1.5)
# Anti-jump window: the projected map target is the MEDIAN of all raw
# projections within this window, so a brief wrong detection (an outlier that
# corrects itself) can't jump the target — it is out-voted by the window. Larger
# = steadier but laggier; smaller = more responsive but jumpier.
NAV_TARGET_WINDOW_S = _env("NAV_TARGET_WINDOW_S", 0.7)
# Anti-teleport continuity gate (on top of the median window). Once a target is
# locked, a projected detection that lands more than NAV_TARGET_JUMP_MM from the
# current track is DROPPED (the track holds its place), so a spurious detection
# or a target lost-and-refound elsewhere cannot teleport the marker. A far
# detection only re-locks after it persists NAV_TARGET_RELOCK_S (the person
# really walked there), or after the track goes stale with no accepted update for
# NAV_TARGET_FORGET_S (a long loss -> allow a fresh lock anywhere).
NAV_TARGET_JUMP_MM = _env("NAV_TARGET_JUMP_MM", 700)
NAV_TARGET_RELOCK_S = _env("NAV_TARGET_RELOCK_S", 2.5)
NAV_TARGET_FORGET_S = _env("NAV_TARGET_FORGET_S", 5.0)
# Ghost guard (navigator.project_detection): how many DISTINCT, mutually
# consistent detector frames must accumulate before a new or relocated target
# is committed. 1 restores the old single-frame behaviour. Operator-settable
# at runtime via Navigator.set_confirm_n(), which clamps to [1, 10].
NAV_TARGET_CONFIRM_N = _env("NAV_TARGET_CONFIRM_N", 3)
# Companion to the above, for mmWave targets only: two reflected-intensity
# (SNR) readings count as "the same reflector" when they differ by no more than
# this FRACTION of the larger one. Loose on purpose — real returns fluctuate
# frame to frame, and a vision target has no SNR at all (comparison is skipped).
NAV_TARGET_SNR_TOL = _env("NAV_TARGET_SNR_TOL", 0.5)
# Emergency stop: while moving, if the rover footprint (car half-extent + this
# margin) overlaps a RAW obstacle, cancel the move and stop immediately.
#
# IMPORTANT — this margin and config.json's `clearance` must be ordered:
# a planned path holds the car body `clearance` mm off every obstacle, and this
# e-stop fires when the body comes within NAV_OBSTACLE_STOP_MARGIN_MM of one.
# So a valid path only avoids tripping the e-stop when
#     clearance  >  NAV_OBSTACLE_STOP_MARGIN_MM.
# With clearance below this value, a path planned at the base clearance trips
# the stop the moment it is driven. The clearance-maximising search below
# mitigates that (it raises the achieved clearance wherever the geometry
# allows, and reports what it achieved), but the ordering is still worth
# fixing in config.json.
NAV_OBSTACLE_STOP_MARGIN_MM = _env("NAV_OBSTACLE_STOP_MARGIN_MM", 150)
# Clearance-maximising planner (rectilinear_mm): rather than planning once at
# config.json's `clearance`, binary-search the LARGEST clearance that still
# admits a path, so the route keeps as far from obstacles as the arena allows.
# Feasibility is monotonic in clearance (inflating obstacles can only shrink
# free space), so the largest feasible value IS the max-min clearance.
# NAV_PLAN_MAX_CLEARANCE_MM caps how much detour is worth buying; the search
# resolution is NAV_PLAN_CLEARANCE_TOL_MM. Set MAX <= the base clearance to
# disable the search and plan exactly as before.
NAV_PLAN_MAX_CLEARANCE_MM = _env("NAV_PLAN_MAX_CLEARANCE_MM", 400)
NAV_PLAN_CLEARANCE_TOL_MM = _env("NAV_PLAN_CLEARANCE_TOL_MM", 10)
# Arena-bounds emergency stop. The planner keeps the rover FOOTPRINT inside the
# map, so on a valid path this never fires; it is the runtime backstop for the
# rover drifting out or a leg overshooting. Measured as protrusion past the
# arena edge, beyond whatever the configured rover_start already protrudes
# (a rover parked in a corner with its centre on the corner point protrudes by
# half its size, and must not e-stop just for sitting there).
# Unlike the obstacle e-stop this is NOT suppressed by ignore_obstacles:
# driving out of a keep-out is a legitimate recovery, leaving the arena is not.
NAV_BOUNDS_STOP_MARGIN_MM = _env("NAV_BOUNDS_STOP_MARGIN_MM", 100)
# purely_control wiring (passed through to T265RoverService).
NAV_CMD_VEL_TOPIC = _env("NAV_CMD_VEL_TOPIC", "/cmd_vel")
NAV_MAX_LINEAR = _env("NAV_MAX_LINEAR", 0.25)   # m/s translation cap during moves (initial)
# Start/stop behaviour (passed to T265RoverService).
# Slew-rate limit on the COMMANDED body velocity, per control tick. At
# CONTROL_HZ=20 this is the acceleration cap: 0.05 -> 1.0 m/s^2 (was 0.08 ->
# 1.6). It is symmetric, so it bounds braking as well as acceleration.
# Commanding a ramp steeper than the loaded base can physically follow shows up
# as WHEEL SLIP at both ends of every leg — which on a mecanum base is
# asymmetric across the four wheels and injects yaw, and feeds vibration into
# the T265's IMU. Set this at or below the base's real achievable acceleration.
NAV_RAMP_STEP = _env("NAV_RAMP_STEP", 0.05)
# Floor on the commanded translation speed while outside POS_TOL. Below
# MIN_LINEAR/LINEAR_GAIN of remaining distance the P term is floored, so the
# rover crosses the last stretch at a CONSTANT speed and arrives at the
# tolerance ball still doing exactly this. It therefore sets the approach speed,
# and the stopping distance from it must stay well inside POS_TOL or the rover
# sails through the ball, the settle timer resets, and it hunts.
# Lowered 0.12 -> 0.06 to suit the tighter POS_TOL; the stiction integral below
# (LIN_IGAIN/LIN_I_MAX) restores breakaway authority on demand, which is what
# the old high floor was crudely providing all the time.
NAV_MIN_LINEAR = _env("NAV_MIN_LINEAR", 0.06)
# Operator-settable speed range for the UI slider (clamps POST /api/nav/speed).
NAV_SPEED_MIN = _env("NAV_SPEED_MIN", 0.05)     # m/s slowest selectable
NAV_SPEED_MAX = _env("NAV_SPEED_MAX", 0.80)     # m/s fastest selectable
# Arrival tolerance per leg (m). This is the width of the envelope the rover
# wanders within around its planned path: at 0.05 a leg could finish 50 mm off,
# eating over half of the ~95 mm worst-case obstacle clearance the planner works
# to achieve. Tightened to 0.025. Legs are re-derived from the live pose
# (see Navigator._drive) so this does NOT accumulate across legs — it bounds
# path-tracking fidelity, not drift.
NAV_POS_TOL = _env("NAV_POS_TOL", 0.025)
# Seconds of no meaningful progress before a leg is failed. MUST stay clear of
# NAV_ARRIVE_SETTLE_TIMEOUT_S: while the rover settles heading it rotates in
# place and makes no distance progress by design, so that time would otherwise
# count against this budget. The watchdog now skips ticks where position is
# already inside tolerance (see T265RoverService._tick), and this margin is the
# second line of defence.
NAV_STALL_TIMEOUT_S = _env("NAV_STALL_TIMEOUT_S", 8.0)
# Heading deadband (rad). MUST be >= the arrival tolerance YAW_TOL (0.020 rad):
# a controller that keeps correcting inside the band it is judged "arrived" in
# will hunt. Inside the deadband the yaw command is zeroed and the integral is
# bled off, so "close enough" means stop rather than coast on stored integral.
NAV_YAW_DEADBAND = _env("NAV_YAW_DEADBAND", 0.020)
# Once POSITION is inside POS_TOL, the rover would rotate in place to bring
# heading inside YAW_TOL. This caps that.
# 0 = never rotate in place after arriving: accept the leg on POSITION alone and
# let the next leg's absolute heading-hold pull the residual out while the rover
# is actually making progress. The rover never turns deliberately (omni base,
# fixed sensor heading), so all yaw is drift — and correcting it standing still
# is where the left-right oscillation lived. Raise this (e.g. 2.0) only if final
# heading accuracy turns out to matter more than not spinning between legs.
NAV_ARRIVE_SETTLE_TIMEOUT_S = _env("NAV_ARRIVE_SETTLE_TIMEOUT_S", 0.0)
# Slack added to every move's deadline, on top of distance/MOVE_MIN_SPEED.
# MUST exceed NAV_ARRIVE_SETTLE_TIMEOUT_S: the deadline is checked BEFORE the
# settle branch in _tick, so a short leg whose deadline expires mid-settle
# finishes as "timeout" — a FAILURE that aborts the whole path — instead of
# arriving. 4.0 was smaller than the new settle window; raised to 9.0.
NAV_MOVE_TIMEOUT_BASE_S = _env("NAV_MOVE_TIMEOUT_BASE_S", 9.0)

# ---- stiction integral (translation) --------------------------------------
# The translation controller is pure P, so it has no way to push through static
# friction: if the floor speed does not break the base away, it simply sits
# there until the stall watchdog fires. This adds targeted integral authority.
# It does NOT integrate distance error (that winds up during normal cruise and
# overshoots); it integrates TIME SPENT STUCK — accruing only while the rover is
# commanded to move but is closing on the goal slower than LIN_STICTION_EPS, and
# bleeding off again as soon as it moves properly. Mirrors YAW_IGAIN/YAW_I_MAX.
NAV_LIN_IGAIN = _env("NAV_LIN_IGAIN", 0.6)          # m/s of boost per second stuck
NAV_LIN_I_MAX = _env("NAV_LIN_I_MAX", 0.10)         # m/s cap on that boost (anti-windup)
NAV_LIN_STICTION_EPS = _env("NAV_LIN_STICTION_EPS", 0.02)   # m/s: "not really moving"

# ---- T265 relocalisation ---------------------------------------------------
# The T265 maps its surroundings continuously and relocalises against that map
# on its own — no pre-built map or survey needed. Those corrections arrive as a
# single large pose STEP, which _filter_pose treats as a glitch and discards
# permanently, re-applying the drift the device just removed. When True, a step
# is accepted while the rover is commanded stationary: a real teleport is
# physically impossible then, so the step is either a relocalisation (wanted) or
# a tracking fault (caught by the confidence gate). Steps during motion are
# still rejected, where a jump genuinely would be a glitch.
NAV_ACCEPT_RELOC_WHEN_STILL = _env("NAV_ACCEPT_RELOC_WHEN_STILL", True)

# ---- tracking-confidence gate ---------------------------------------------
# T265 tracker_confidence is 0=Failed, 1=Low, 2=Medium, 3=High. Only 0 stopped
# the rover; 1 and 2 were treated exactly like 3, yet Low is precisely when VIO
# drift accrues fastest. Before starting a leg, wait up to NAV_CONF_WAIT_S for
# confidence to reach NAV_MIN_START_CONF; if it never does, proceed anyway (so a
# demo is never simply stuck) but scale the speed cap by NAV_LOW_CONF_SPEED_SCALE.
NAV_MIN_START_CONF = _env("NAV_MIN_START_CONF", 2)
NAV_CONF_WAIT_S = _env("NAV_CONF_WAIT_S", 2.0)
NAV_LOW_CONF_SPEED_SCALE = _env("NAV_LOW_CONF_SPEED_SCALE", 0.5)

# ---- drift accounting ------------------------------------------------------
# Odometry error grows with how far the rover has driven and how much it has
# turned since the pose was last known-good (a reset_pose / re-anchor). These
# convert that into an extra planning margin, so the planner routes further from
# obstacles the longer it has been since a fix. NAV_DRIFT_MARGIN_MAX_MM caps it
# so a long session cannot make the whole arena unreachable; set the max to 0 to
# disable the mechanism entirely.
NAV_DRIFT_PER_M_MM = _env("NAV_DRIFT_PER_M_MM", 8.0)        # mm of margin per metre driven
NAV_DRIFT_PER_TURN_MM = _env("NAV_DRIFT_PER_TURN_MM", 1.5)  # mm per degree of |yaw| turned
NAV_DRIFT_MARGIN_MAX_MM = _env("NAV_DRIFT_MARGIN_MAX_MM", 120.0)
# Zero-velocity drift sampling: while the rover is commanded stationary its true
# velocity is zero, so ANY pose change the T265 reports is drift, measured
# directly and for free. Samples are taken over this window between legs.
NAV_ZUPT_WINDOW_S = _env("NAV_ZUPT_WINDOW_S", 0.5)

# ---- sensor lever arms -----------------------------------------------------
# A sensor mounted away from the rover's TURN CENTRE swings through an arc when
# the rover rotates in place, and reports that arc as genuine translation. With
# no compensation the rover's believed position (and any target projected from
# it) slides sideways every time it corrects heading — the swing is
# 2*d*sin(dtheta/2), so a 10 deg correction with the sensor 150 mm off centre
# moves the estimate 26 mm, more than the whole arrival tolerance.
#
# Offsets are in the ROVER BODY frame, millimetres, measured from the turn
# centre (the centre of the chassis footprint): +forward is toward the front,
# +right is toward the rover's right. For a 600 mm-long chassis the turn centre
# sits 300 mm behind the front bumper, so a sensor 240 mm back from the front is
# 300 - 240 = +60 mm FORWARD of centre.
NAV_T265_OFFSET_FWD_MM = _env("NAV_T265_OFFSET_FWD_MM", 60.0)
NAV_T265_OFFSET_RIGHT_MM = _env("NAV_T265_OFFSET_RIGHT_MM", 0.0)
# Same for the D435: bearing and range are measured from the COLOUR CAMERA, so
# a target must be projected from the camera's map position, not the rover
# centre's, or it swings every time the rover corrects heading.
# D435 at 240 mm behind the front of a 600 mm chassis -> 300 - 240 = +60 mm
# forward of the turn centre, laterally centred.
NAV_CAM_OFFSET_FWD_MM = _env("NAV_CAM_OFFSET_FWD_MM", 60.0)
NAV_CAM_OFFSET_RIGHT_MM = _env("NAV_CAM_OFFSET_RIGHT_MM", 0.0)

# ---- ArUco tag localisation ------------------------------------------------
# Absolute position fixes from markers at surveyed positions on the panels.
# This is the only mechanism here that BOUNDS drift rather than slowing it:
# the pose is measured against the arena instead of integrated from motion.
# Tags are authored per-obstacle in map.json (see tag_localizer.build_tag_table).
TAGS_ENABLED = _env("TAGS_ENABLED", True)
TAGS_DICT = _env("TAGS_DICT", "DICT_4X4_50")
TAGS_SIZE_MM = _env("TAGS_SIZE_MM", 190.0)   # side of the BLACK square, excl. quiet zone
TAGS_DETECT_HZ = _env("TAGS_DETECT_HZ", 4.0) # detection rate; cheap, but no need to run flat out
# A fix is only APPLIED while the rover is commanded stationary: motion blur
# wrecks corner precision, and a correction mid-leg moves the goal underneath
# the planner. Fixes older than TAGS_FIX_MAX_AGE_S are ignored.
TAGS_FIX_MAX_AGE_S = _env("TAGS_FIX_MAX_AGE_S", 1.5)
# How often the monitor loop attempts a fix while the rover is stopped. Fixes
# used to happen ONLY between the legs of an active move, so a parked or idle
# rover drifted uncorrected however long it sat there — even though standing
# still in front of a panel is the best possible moment to take a fix.
TAGS_IDLE_FIX_INTERVAL_S = _env("TAGS_IDLE_FIX_INTERVAL_S", 1.0)
# Reprojection RMS gate. This is NOT fussiness: RMS is a direct proxy for how
# wrong the resulting fix would be. Measured on this arena with one tag of a
# panel mis-recorded in map.json:
#     rms  4 px -> the accepted fix would be  ~86 mm out
#     rms  7 px ->                            ~230 mm out
#     rms 10 px ->                            ~395 mm out  (and ~10 deg of yaw)
# Raising this does not improve a fix; it lets a WRONG one through. Persistent
# high RMS means the tag offsets in map.json disagree with the physical layout.
# NOTE a single tag (4 corners, 6 pose DOF) can always be fitted exactly, so it
# CANNOT fail this gate -- see TAGS_MIN_TAGS_FOR_FIX below.
TAGS_MAX_RMS_PX = _env("TAGS_MAX_RMS_PX", 6.0)
# Scale the RMS allowance with the number of tags: more tags means more mutual
# constraints and a legitimately higher residual even when everything is right.
# Effective limit = TAGS_MAX_RMS_PX + TAGS_RMS_PER_TAG_PX * (n_tags - 1).
TAGS_RMS_PER_TAG_PX = _env("TAGS_RMS_PER_TAG_PX", 1.0)
# Require at least this many tags before a fix is APPLIED. 1 allows single-tag
# fixes. Raised from 1 to 2. A single tag gives 4 corners against 6 pose DOF, so it
# fits EXACTLY however wrong the map is — its reprojection residual is always
# ~0.2 px and it cannot fail the RMS gate. In practice that inverted the safety
# logic: the checkable multi-tag frames were rejected while the uncheckable
# single-tag ones were applied, including a 1278 mm correction carrying 21 deg
# of heading error on a rover that never turns. Two tags can at least disagree.
TAGS_MIN_TAGS_FOR_FIX = _env("TAGS_MIN_TAGS_FOR_FIX", 2)
# Reject a fix whose solved heading disagrees with the T265 by more than this.
# 0 disables the check. Enabled at 5 deg. The rover has an omni base with a fixed sensor heading and
# never turns deliberately, so map yaw should sit near zero: a tag solve
# claiming 9 or 21 deg is evidence the MAP is wrong, not the rover. Because the
# position half of the same solve is wrong by roughly 40 mm per degree,
# rejecting on yaw also throws out the position error that comes with it.
# Measured residuals on this arena scatter +/-8 to 21 deg with a median of only
# -0.84 deg: it is NOISE from the map/tag placement errors, not a fixed mounting
# bias, so a tight absolute gate rejects most frames rather than a few. At 5.0
# it rejected 132 of 132 fixes, leaving drift completely uncorrected — far worse
# than applying an imperfect position fix. 12 deg still catches the genuinely
# broken solves (the 21 deg and 13.8 deg outliers) while letting the usable
# majority through. Yaw itself is NOT applied (TAGS_CORRECT_YAW is False); the
# position half is protected instead by TAGS_BIG_FIX_MM confirmation, the
# TAGS_FIX_ALPHA easing and TAGS_FIX_MAX_STEP_MM cap.
# Tighten this back toward 5 once the map offsets are corrected and the reported
# yaw spread (state()["tag_yaw"]) drops.
TAGS_MAX_YAW_ERR_DEG = _env("TAGS_MAX_YAW_ERR_DEG", 12.0)
# Above this reprojection RMS the detector runs a diagnostic pass (see
# TagLocalizer.audit) that separates a wrong tag SIZE — a uniform scale error a
# single ratio absorbs — from a misplaced or mis-identified tag, which is local
# and does not rescale away. Only runs on bad frames, so it is effectively free.
TAGS_AUDIT_RMS_PX = _env("TAGS_AUDIT_RMS_PX", 3.0)
# ...but no more often than this. The audit is a DIAGNOSTIC whose answer does
# not change frame to frame, and every real frame in this arena sits above the
# trigger above, so it was running constantly and costing more CPU than the
# detection it was diagnosing. The last verdict is kept and re-displayed in
# between, so the UI still shows it continuously.
TAGS_AUDIT_MIN_INTERVAL_S = _env("TAGS_AUDIT_MIN_INTERVAL_S", 5.0)
# Minimum horizontal separation (mm) between the visible tags. Tags stacked in
# one vertical column leave sideways position and heading under-determined, and
# the solver answers with a confident mirrored pose that reprojects at ~0.3 px —
# invisible to the RMS gate and repeatable enough to survive a confirmation run.
# Two tags at different HEIGHTS do not substitute for two at different x.
TAGS_MIN_SPREAD_MM = _env("TAGS_MIN_SPREAD_MM", 400.0)
# ArUco corner refinement: NONE | SUBPIX | CONTOUR | APRILTAG. OpenCV's own
# default is NONE, which finds a corner only to the quad detector's contour
# vertex. SUBPIX roughly halves the corner error (0.72-0.98 px -> 0.34-0.47 px
# measured on rendered tags) and cut pose error from 2.7 mm to 0.4 mm on a
# 4-tag panel at 2.5 m, for a few ms per frame. Set to NONE to A/B it.
TAGS_CORNER_REFINE = _env("TAGS_CORNER_REFINE", "SUBPIX")
# ArUco binarises with a LOCAL adaptive threshold, sweeping window sizes from
# MIN to MAX in STEP. OpenCV's default (3/23/10 -> tries 3, 13, 23) starts at a
# 3 px window, which is almost exactly the perforation pitch of the pegboard
# panels: the threshold locks onto hole texture instead of the marker. Measured
# on a real captured frame with 4 tags physically present:
#     min=3 (default)  1 tag  -- and 1-3 depending on STEP, i.e. luck
#     min=5            1-3 tags, still step-sensitive
#     min=7            3 tags at EVERY step value tried  <- robust
#     min=9            2 tags
#     min=11           1 tag
# 7 is chosen because it is stable under STEP, not merely best in one cell.
TAGS_THRESH_WIN_MIN = _env("TAGS_THRESH_WIN_MIN", 7)
TAGS_THRESH_WIN_MAX = _env("TAGS_THRESH_WIN_MAX", 25)
TAGS_THRESH_WIN_STEP = _env("TAGS_THRESH_WIN_STEP", 8)
# Unsharp mask applied to the grayscale before detection. With the window above
# it recovered the 4th tag in 12 of 12 amount/sigma combinations (0.4-1.0 x
# 1.5-3.0) on the same real frame — insensitive to tuning, which is what makes
# it trustworthy. It is the COMBINATION that works: unsharp with the DEFAULT
# threshold window still found only one tag.
# It costs ~6 ms/frame and does NOT degrade corner precision (measured against
# known ground truth: 0.708 -> 0.700 px at 140 px span, 0.740 -> 0.741 at 60 px,
# 1.002 -> 0.987 under glare) because SUBPIX re-finds the true edge afterwards.
# Set TAGS_UNSHARP_AMOUNT to 0 to disable.
TAGS_UNSHARP_AMOUNT = _env("TAGS_UNSHARP_AMOUNT", 0.6)
TAGS_UNSHARP_SIGMA = _env("TAGS_UNSHARP_SIGMA", 2.0)
# A correction larger than this is only trusted after TAGS_CONFIRM_N successive
# fixes agree within TAGS_AGREE_MM - the same challenge-counter pattern the
# radar ghost guard uses. Small corrections apply immediately.
TAGS_BIG_FIX_MM = _env("TAGS_BIG_FIX_MM", 300.0)
TAGS_CONFIRM_N = _env("TAGS_CONFIRM_N", 3)
TAGS_AGREE_MM = _env("TAGS_AGREE_MM", 150.0)
# Corrections are eased in rather than snapped, so the control loop never sees
# a discontinuity: anchor += TAGS_FIX_ALPHA * residual, capped per application.
TAGS_FIX_ALPHA = _env("TAGS_FIX_ALPHA", 0.35)
TAGS_FIX_MAX_STEP_MM = _env("TAGS_FIX_MAX_STEP_MM", 120.0)
# Position-only to start: a bad yaw fix rotates the whole map frame, which is
# far more damaging than a bad position fix. The solved heading is still
# reported (state()["tag_fix"]["yaw_err_deg"]) so it can be watched before
# being trusted.
TAGS_CORRECT_YAW = _env("TAGS_CORRECT_YAW", False)
# Operator hold-to-move (jog): speed, and the dead-man window — the UI refreshes
# the jog every ~200 ms while the button is held; if refreshes stop (release,
# tab close, network drop) the rover stops within this many seconds.
NAV_JOG_SPEED = _env("NAV_JOG_SPEED", 0.12)     # m/s (initial hold-to-move speed)
# Operator-settable jog speed range for the Drive-card slider. Note the rover
# service also hard-caps jog at MAX_LINEAR (the nav speed slider), so the
# effective jog speed is min(jog_speed, current speed cap).
NAV_JOG_SPEED_MIN = _env("NAV_JOG_SPEED_MIN", 0.05)
NAV_JOG_SPEED_MAX = _env("NAV_JOG_SPEED_MAX", 0.40)
NAV_JOG_DEADMAN_S = _env("NAV_JOG_DEADMAN_S", 0.5)

# ---------------------------------------------------------------- detection (audience view)
DETECT_ENABLE = _env("DETECT_ENABLE", True)
# The detector model code is vendored under <repo>/model_stack (see that dir's
# README). The trained weights (*.pt) are NOT in the repo — drop them into
# model_stack/weights/, or point JETSON_DEPLOY_DIR at a tree that has both the
# model code and a weights/ subdir.
JETSON_DEPLOY_DIR = _env(
    "JETSON_DEPLOY_DIR", os.path.join(DEMO_ROOT, "model_stack"))
DETECT_CONF = _env("DETECT_CONF", 0.05)
DETECT_CONF_NORADAR = _env("DETECT_CONF_NORADAR", 0.30)
DETECT_TRACK_MIN_HITS = _env("DETECT_TRACK_MIN_HITS", 1)
TARGET_SMOOTH = _env("TARGET_SMOOTH", 0.4)
TARGET_MAX_STEP = _env("TARGET_MAX_STEP", 0.08)
TARGET_COAST = _env("TARGET_COAST", 0.4)
# RGB-detection range source: D435 depth is the baseline. If depth drops out and
# the radar (mmWave) range that would replace it differs from the last depth by
# more than DETECT_DEPTH_HOLD_JUMP_M (a spurious jump on the depth->mmwave
# handoff), keep using the last depth value for up to DETECT_DEPTH_HOLD_S before
# accepting the radar range.
DETECT_DEPTH_HOLD_S = _env("DETECT_DEPTH_HOLD_S", 1.0)
DETECT_DEPTH_HOLD_JUMP_M = _env("DETECT_DEPTH_HOLD_JUMP_M", 1.0)
DETECT_IMG_SIZE = _env("DETECT_IMG_SIZE", 416)
DETECT_FPS = _env("DETECT_FPS", 10)
DETECT_ADAPTIVE = _env("DETECT_ADAPTIVE", True)
DETECT_USE_TRT = _env("DETECT_USE_TRT", True)
DETECT_GATE_EVERY = _env("DETECT_GATE_EVERY", 2)
DETECT_RENDER_FPS = _env("DETECT_RENDER_FPS", 30)
DETECT_SHOW_RES = _env("DETECT_SHOW_RES", True)
RESOLUTION_DEBOUNCE = _env("RESOLUTION_DEBOUNCE", 2.0)
# DQN tier hysteresis: the policy must choose the SAME new tier this many
# detection frames IN A ROW before it commits (40 @ ~10 det fps ≈ 4 s). Any
# flip-flopping in between resets the counter, so the displayed resolution
# (tier boxes + degraded streams) changes at most every few seconds.
DETECT_TIER_HOLD_FRAMES = _env("DETECT_TIER_HOLD_FRAMES", 40)
DETECT_RGB_HFOV = _env("DETECT_RGB_HFOV", 69.0)
DETECT_RADAR_FLIP = _env("DETECT_RADAR_FLIP", False)
# Vision <-> radar bearing agreement (deg). VISION IS THE AUTHORITY: radar range
# is fused into a vision detection, and a radar-only det may sustain a recently
# seen target through a dropout, ONLY when the radar bearing agrees with the
# vision bearing within this tolerance. A disagreeing radar lock (e.g. a
# transient multipath ghost) is ignored rather than allowed to corrupt an
# accurate track (see _assemble_dets in the detector).
FOLLOW_FUSE_TOL = _env("FOLLOW_FUSE_TOL", 14.0)
# Radar may REDEFINE the target (radar-only detection at a NEW bearing) only
# after vision has produced no accepted detection for this long. Inside the
# window a radar-only det is accepted solely at a FOLLOW_FUSE_TOL-consistent
# bearing — sustaining the SAME object through a brief vision dropout — so a
# momentary radar ghost can never relocate an accurately tracked target.
DETECT_RADAR_TAKEOVER_S = _env("DETECT_RADAR_TAKEOVER_S", 1.5)
# mmWave presence / tracker tuning.
RADAR_EPS = _env("RADAR_EPS", 0.7)
RADAR_MIN_PTS = _env("RADAR_MIN_PTS", 2)
RADAR_ENTER_NEED = _env("RADAR_ENTER_NEED", 2)
RADAR_RELEASE_WIN = _env("RADAR_RELEASE_WIN", 18)
RADAR_AZ_SMOOTH = _env("RADAR_AZ_SMOOTH", 0.35)
RADAR_USE_TRACKER = _env("RADAR_USE_TRACKER", True)
# Occlusion robustness: with intermittent returns (the person is blocked by a
# material for a moment), accumulate over a LONGER window so the cloud stays
# dense enough to cluster across sparse frames, HOLD the lock through longer
# dropouts, and SMOOTH the position harder so a partial return doesn't yank the
# centroid. Raise ACCUM/HOLD further if blockages last longer than ~3 s.
RADAR_ACCUM_SEC = _env("RADAR_ACCUM_SEC", 0.5)     # was 0.35 — denser cloud, rides sparse frames
RADAR_V_MOVE = _env("RADAR_V_MOVE", 0.12)
RADAR_GATE_RADIUS = _env("RADAR_GATE_RADIUS", 1.0)
# Ghost rejection at ACQUIRE time (the "last X positions" check): a fresh lock
# requires the winning moving cluster to persist within
# RADAR_ACQUIRE_CONFIRM_RADIUS (m) of itself for RADAR_ACQUIRE_CONFIRM_S (s).
# A transient multipath ghost only remains clusterable for roughly its own
# duration + RADAR_ACCUM_SEC (points age out of the accumulation cloud), so a
# few-frame ghost never confirms; a real person's continuous returns confirm
# ~1 s after they start moving. Keep CONFIRM_S comfortably above
# RADAR_ACCUM_SEC; 0 restores the old instant single-window acquisition.
RADAR_ACQUIRE_CONFIRM_S = _env("RADAR_ACQUIRE_CONFIRM_S", 1.0)
RADAR_ACQUIRE_CONFIRM_RADIUS = _env("RADAR_ACQUIRE_CONFIRM_RADIUS", 0.6)
RADAR_HOLD_TIMEOUT = _env("RADAR_HOLD_TIMEOUT", 3.0)   # was 1.5 — coast through occlusion gaps
RADAR_POS_SMOOTH = _env("RADAR_POS_SMOOTH", 0.2)       # was 0.3 — heavier smoothing on partial returns
RADAR_CONFIRM_HOLD = _env("RADAR_CONFIRM_HOLD", 15.0)
RADAR_MIN_SNR_PEAK = _env("RADAR_MIN_SNR_PEAK", 0)
RADAR_MIN_SNR_SUM = _env("RADAR_MIN_SNR_SUM", 0)
# Software-side static filter on top of the hardware clutterRemoval: drop
# points with |radial v| below this (m/s) before they reach the tracker.
RADAR_MIN_POINT_V = _env("RADAR_MIN_POINT_V", 0.05)
# Displayed radar point cloud: drop points slower than this (m/s) as micro-
# jitter, and blank the whole cloud while the rover is moving (ego-motion fakes
# Doppler on static clutter, so the cloud is only meaningful when stationary).
MMWAVE_DISPLAY_MIN_V = _env("MMWAVE_DISPLAY_MIN_V", 0.08)
DETECT_THERMAL_ENHANCE = _env("DETECT_THERMAL_ENHANCE", True)
DETECT_THERMAL_INVERT = _env("DETECT_THERMAL_INVERT", False)
DETECT_THERMAL_CONF = _env("DETECT_THERMAL_CONF", 0.15)
DETECT_MIN_BOX_FRAC = _env("DETECT_MIN_BOX_FRAC", 0.02)
DETECT_ALIVE_STD = _env("DETECT_ALIVE_STD", 5.0)
DETECT_ALIVE_MEAN = _env("DETECT_ALIVE_MEAN", 3.0)

# ---------------------------------------------------------------- misc
ALLOW_MOCK = _env("ALLOW_MOCK", True)
JPEG_QUALITY = _env("JPEG_QUALITY", 70)
