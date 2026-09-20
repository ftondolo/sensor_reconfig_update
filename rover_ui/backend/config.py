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
# Raised 250 -> 350 (2026-09-18): re-dispatching for a goal only ~250 mm away
# re-planned on target jitter and made the rover shuttle back and forth.
NAV_GOAL_TOLERANCE_MM = _env("NAV_GOAL_TOLERANCE_MM", 350)
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
NAV_FOLLOW_CYCLE_S = _env("NAV_FOLLOW_CYCLE_S", 20.0)      # re-plan at next corner after this; hard cancel at 2x
NAV_FOLLOW_MIN_INTERVAL_S = _env("NAV_FOLLOW_MIN_INTERVAL_S", 1.0)
NAV_FOLLOW_REPLAN_MM = _env("NAV_FOLLOW_REPLAN_MM", 500)   # re-plan when target moves this far (mm)
# Receding-horizon refinement of the above: a target that crosses from one side
# of the rover to the other (its map-x goes from left of the rover's current
# position to right, or vice versa) means the leg in progress is walking
# toward the wrong half-plane, however small the raw displacement -- so this
# forces an early re-plan even when NAV_FOLLOW_REPLAN_MM has not been reached.
# Guarded by a minimum lateral swing so noise near dead-centre can't flap it.
NAV_FOLLOW_SIDE_SWITCH_MM = _env("NAV_FOLLOW_SIDE_SWITCH_MM", 300)
# Strict move-stop-sense gate: a radar (moving-point) target is accepted only
# after the rover has been confirmed STATIONARY this long — enough for the
# detector to re-acquire a moving cluster while still (>= ~RADAR_ACCUM_SEC plus
# settle). Raise if the rover commits to a radar target before it has truly
# stopped; lower for snappier hops. (Vision targets are not gated by this.)
NAV_FOLLOW_STILL_CONFIRM_S = _env("NAV_FOLLOW_STILL_CONFIRM_S", 0.5)
# ---- FOLLOW: blind-time limit + corner look (added 2026-09-19) -------------
# While the rover drives, radar is switched off by the detector (ego-motion), so
# on a long path a radar-only target goes unseen until arrival. FOLLOW now uses
# the CORNERS of the plan (where the rover stops anyway) to look again:
#   * blind time = seconds since the last accepted detection, from ANY sensor;
#   * at a corner, look when blind >= NAV_FOLLOW_BLIND_S, or when it would pass
#     that limit by the end of the next leg (bounds the worst case to ~1 leg);
#   * skip the look when less than NAV_FOLLOW_LOOK_MIN_REMAIN_MM of path is left
#     (the normal look on arrival covers it);
#   * a look waits for a fresh detection and ends as soon as one arrives (vision:
#     almost at once; radar: ~1.5 s to re-lock), capped at NAV_FOLLOW_LOOK_MAX_S.
NAV_FOLLOW_BLIND_S = _env("NAV_FOLLOW_BLIND_S", 4.0)
NAV_FOLLOW_LOOK_MAX_S = _env("NAV_FOLLOW_LOOK_MAX_S", 2.0)
NAV_FOLLOW_LOOK_MIN_REMAIN_MM = _env("NAV_FOLLOW_LOOK_MIN_REMAIN_MM", 800)
# After a look, re-plan ONLY when needed: the new standoff goal is more than
# NAV_GOAL_TOLERANCE_MM from the current one, the target moved at least
# NAV_FOLLOW_REPLAN_MM, or it crossed sides. Never twice within this interval.
NAV_FOLLOW_REPLAN_MIN_INTERVAL_S = _env("NAV_FOLLOW_REPLAN_MIN_INTERVAL_S", 2.0)
# Mid-leg re-planning. A target move >= NAV_FOLLOW_REPLAN_MM seen WHILE driving
# is now deferred to the next corner (no stop mid-straight). Only a side switch
# or a move this large still stops the rover mid-leg. Set equal to
# NAV_FOLLOW_REPLAN_MM to restore the old always-immediate behaviour.
NAV_FOLLOW_MIDLEG_REPLAN_MM = _env("NAV_FOLLOW_MIDLEG_REPLAN_MM", 1500)
# More corners in FOLLOW (without extra distance). The planner always picks
# the fewest-turn path, so lowering turn_penalty doesn't add corners on a grid.
# Instead, each L-shaped pair of legs with a leg longer than
# NAV_FOLLOW_MAX_LEG_MM is re-cut into an equal-length staircase with a corner
# about every MAX_LEG, used only if it keeps the clearance the original route
# achieved. Steps are never shorter than NAV_FOLLOW_MIN_STEP_MM. A single long
# straight leg is left alone. Set MAX_LEG to 0 to disable.
NAV_FOLLOW_MAX_LEG_MM = _env("NAV_FOLLOW_MAX_LEG_MM", 1500)
NAV_FOLLOW_MIN_STEP_MM = _env("NAV_FOLLOW_MIN_STEP_MM", 250)
# Last known position (prior). When the averaging window is emptied (a corner
# look, or NAV_TARGET_FORGET_S without detections) the last estimate is kept.
# A new detection within R0 + SPEED x (time since last seen), capped at MAX, is
# accepted from one frame; one further away could be a ghost and needs
# NAV_TARGET_PRIOR_CONFIRM_N consistent frames first.
NAV_TARGET_PRIOR_R0_MM = _env("NAV_TARGET_PRIOR_R0_MM", 300.0)
NAV_TARGET_PRIOR_SPEED_MM_S = _env("NAV_TARGET_PRIOR_SPEED_MM_S", 1000.0)
NAV_TARGET_PRIOR_MAX_MM = _env("NAV_TARGET_PRIOR_MAX_MM", 4000.0)
NAV_TARGET_PRIOR_CONFIRM_N = _env("NAV_TARGET_PRIOR_CONFIRM_N", 2)
# How long a projected detection target stays shown on the map after the
# detection drops (purely cosmetic; navigation uses the live value).
NAV_TARGET_HOLD_S = _env("NAV_TARGET_HOLD_S", 1.5)
# ---- target running average (replaces the Confirm-N ghost guard) ----------
# The projected map target is a ROBUST RUNNING AVERAGE over the last
# NAV_TARGET_AVG_N DETECTION FRAMES -- only detector frames that carry a FRESH
# detection count (see detector.py target["fresh"]); frames with no detection,
# or where the detector is merely coasting/holding an old box or radar lock,
# do not advance the window. The window always ends at the current detection
# frame and uses however many frames exist (1..N), so it never waits to fill.
#
# Ghost handling is by WEIGHTING, not by a hard block: the centre is the medoid
# of the window (the sample closest to all others; ties -> newest), samples
# within NAV_TARGET_INLIER_MM of it count fully, the weight tapers to zero at
# NAV_TARGET_JUMP_MM. A one-frame ghost therefore has no effect; a real move is
# followed once it holds the majority of the window (~N/2+1 detection frames).
# Operator-settable at runtime via the UI (Navigator.set_avg_n), clamped to
# [NAV_TARGET_AVG_N_MIN, NAV_TARGET_AVG_N_MAX]. Below 3 there is little ghost
# rejection left.
NAV_TARGET_AVG_N = _env("NAV_TARGET_AVG_N", 5)
NAV_TARGET_AVG_N_MIN = _env("NAV_TARGET_AVG_N_MIN", 1)
NAV_TARGET_AVG_N_MAX = _env("NAV_TARGET_AVG_N_MAX", 20)
NAV_TARGET_INLIER_MM = _env("NAV_TARGET_INLIER_MM", 250)
# Beyond this distance from the window centre a sample gets zero weight.
NAV_TARGET_JUMP_MM = _env("NAV_TARGET_JUMP_MM", 700)
# Detection frames only advance the window when they arrive, so after a long
# loss the window could hold a position the person has long left. If the newest
# sample is older than this, the window is cleared and the next detection
# starts a fresh lock.
NAV_TARGET_FORGET_S = _env("NAV_TARGET_FORGET_S", 5.0)
# How often the tracker polls the detector for a new frame (Navigator._track_loop).
# Faster than the detector frame rate so no detection frame is missed; repeated
# polls of the same frame are ignored.
NAV_TRACK_UPDATE_S = _env("NAV_TRACK_UPDATE_S", 0.05)
# Removed 2026-09-18 with the running average: NAV_TARGET_WINDOW_S (median
# window), NAV_TARGET_RELOCK_S, NAV_TARGET_CONFIRM_N, NAV_TARGET_SNR_TOL.
# ---- rover footprint safety (all tests use the car RECTANGLE, not its centre) --
# The rover is the axis-aligned car rectangle from config.json (car.width along
# x, car.length along z) centred on its pose. Distances below are from that
# rectangle to an obstacle's collision box (panel + stabiliser feet), measured
# per axis like the red halo on the map. Bands, which must stay ordered:
#   hard halt  <  recover trigger  <  plan floor  <  recovery target
#   NAV_OBSTACLE_STOP_MARGIN_MM  <  clearance - NAV_RECOVER_TOL_MM  <  clearance
#                                <  clearance + NAV_RECOVER_TOL_MM
# (clearance = config.json `clearance`, 150 mm). Walls: any part of the
# footprint outside the arena -> recover, back to >= config.json
# `wall_clearance` (50 mm) inside. Walls never cause a halt.
#
# HARD HALT: while moving, the footprint within this distance of an obstacle ->
# stop and stay stopped (the only e-stop left). A fixed buffer: the footprint is
# checked at ~20 Hz on the CURRENT pose, so 30 mm suits speeds up to ~0.3 m/s;
# at higher speed-slider settings the rover can cover more than this between a
# check and standing still.
NAV_OBSTACLE_STOP_MARGIN_MM = _env("NAV_OBSTACLE_STOP_MARGIN_MM", 30)
# RECOVER: footprint closer than (clearance - this) to an obstacle, or partly
# outside the arena -> stop, take the shortest straight move back to a safe pose
# (>= clearance + this from obstacles, >= wall_clearance inside the arena,
# never sweeping through the hard-halt buffer), re-plan to the same goal and
# continue. The tolerance absorbs normal path-tracking error.
NAV_RECOVER_TOL_MM = _env("NAV_RECOVER_TOL_MM", 25)
NAV_RECOVER_MAX_MM = _env("NAV_RECOVER_MAX_MM", 400)    # longest recovery move searched
NAV_RECOVER_MAX_TRIES = _env("NAV_RECOVER_MAX_TRIES", 3)  # per navigation, then halt
# Detection no-target zone around each obstacle (a projected person inside it is
# ignored). Kept separate from the rover clearance so raising the clearance does
# not widen it (it used to follow config.json `clearance`, then 75 mm).
NAV_TARGET_EXCLUSION_MM = _env("NAV_TARGET_EXCLUSION_MM", 75)
# Planner clearance settings live in config.json: clearance (floor),
# max_clearance, preferred_clearance, clearance_weight, wall_clearance,
# turn_penalty (see rectilinear_mm.plan_rectilinear_path_ex).
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
# Lowered 0.06 -> 0.045 (2026-09-18): a slower final approach keeps the
# stopping distance well inside POS_TOL. Must stay ABOVE the speed at which the
# base actually breaks away from rest -- if legs now stall short of the goal,
# raise this back toward 0.06.
NAV_MIN_LINEAR = _env("NAV_MIN_LINEAR", 0.045)
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
# Lowered 8.0 -> 4.0 (2026-09-18): end-of-leg hunting never improves the best
# distance, so this is how long any residual hunting can run before the leg is
# failed as "stalled" and FOLLOW/AUTO re-plans from the live pose.
NAV_STALL_TIMEOUT_S = _env("NAV_STALL_TIMEOUT_S", 4.0)
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
#
# Softened 2026-09-18 (was IGAIN 0.6 / I_MAX 0.10 / EPS 0.02) -- this was the main
# driver of the end-of-move hunting: an OVERSHOOT also reads as "not closing",
# so the boost wound up on every overshoot and sent the rover back through the
# tolerance ball at ~0.15 m/s, overshooting further each time. (It also jumps
# straight to I_MAX on leaving the ball, because prev_dist/prev_t are not
# updated while inside it.) Smaller gain + cap keep breakaway help but bound
# the return speed to ~MIN_LINEAR + 0.04. A/B with IGAIN = 0 to disable it.
NAV_LIN_IGAIN = _env("NAV_LIN_IGAIN", 0.2)          # m/s of boost per second stuck
NAV_LIN_I_MAX = _env("NAV_LIN_I_MAX", 0.04)         # m/s cap on that boost (anti-windup)
NAV_LIN_STICTION_EPS = _env("NAV_LIN_STICTION_EPS", 0.01)   # m/s: "not really moving"

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
NAV_CONF_WAIT_S = _env("NAV_CONF_WAIT_S", 0.5)
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
NAV_ZUPT_WINDOW_S = _env("NAV_ZUPT_WINDOW_S", 0.1)
# A full ZUPT sample blocks for NAV_ZUPT_WINDOW_S before the leg it precedes can
# start, so taking one before EVERY leg adds that pause to every waypoint on a
# multi-leg path for little extra information -- the drift rate barely moves
# leg to leg. Skip a sample (and its wait) when the last one completed less
# than this long ago, UNLESS T265 confidence is currently below
# NAV_MIN_START_CONF (see _await_confidence) -- that is exactly when a fresh
# reading is worth pausing for. 0 restores the old every-leg behaviour.
NAV_ZUPT_MIN_INTERVAL_S = _env("NAV_ZUPT_MIN_INTERVAL_S", 2.5)

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
# Horizontal separation (mm) between the visible tags that is ALWAYS accepted,
# at any range. Tags stacked in one vertical column leave sideways position and
# heading under-determined, and the solver answers with a confident mirrored
# pose that reprojects at ~0.3 px — invisible to the RMS gate and repeatable
# enough to survive a confirmation run. Two tags at different HEIGHTS do not
# substitute for two at different x.
# This is no longer the whole gate: it is the unconditional PASS threshold, and
# TAGS_MIN_SPREAD_RATIO below adds a second way in for close-range observations
# that are well conditioned despite a narrow spread. Nothing that passed before
# fails now.
TAGS_MIN_SPREAD_MM = _env("TAGS_MIN_SPREAD_MM", 400.0)
# --- angular baseline: spread / range ---------------------------------------
# The absolute floor above rejects a genuinely good close-range observation in
# exactly the same way as a genuinely bad far-range one, because it never asks
# how far away the tags are. What actually governs the error is the ANGULAR
# baseline — spread divided by range — since that is what decides how much
# perspective difference is available to tell yaw apart from lateral
# translation. At ratio -> 0 the view tends to orthographic, where strafing
# sideways and rotating in yaw produce nearly the same image motion.
# Measured (two tags, one height, 0.4 px corner noise, 200 trials, median
# lateral pose error) — read down the RATIO column, the error tracks it and
# barely tracks spread or range on their own:
#     spread  range  ratio   error
#      150 mm  0.8 m  0.188    4.5 mm
#      250 mm  1.0 m  0.250    5.6 mm   <- rejected today for no reason
#      250 mm  2.0 m  0.125   48.5 mm
#      250 mm  3.5 m  0.071  287.4 mm   <- correctly bad
#      400 mm  2.0 m  0.200   28.1 mm
#      400 mm  3.5 m  0.114  208.0 mm   <- ACCEPTED today
#      650 mm  2.0 m  0.325   20.3 mm   <- accepted today
#      650 mm  3.5 m  0.186  105.8 mm   <- accepted today
# 0.18 is calibrated against what the CURRENT gate already tolerates, not to
# taste: a 650 mm pair at 3.5 m is ratio 0.186 with ~106 mm of error and passes
# today, so anything at 0.18 or better is no worse conditioned than something
# the system already trusts. Set to 0 to disable and restore the old
# floor-only behaviour exactly.
TAGS_MIN_SPREAD_RATIO = _env("TAGS_MIN_SPREAD_RATIO", 0.18)
# Hard floor the ratio can never argue past: below this the two tags are
# physically almost one tag whatever the range, and it also stops a range
# UNDER-estimate (a partly-occluded tag measures small, so it reads as far
# away) from talking a near-zero spread through the gate.
TAGS_MIN_SPREAD_FLOOR_MM = _env("TAGS_MIN_SPREAD_FLOOR_MM", 150.0)
# Make the ratio a REQUIREMENT rather than an alternative route, so a
# wide-but-distant pair is rejected too. Note from the table above that the
# absolute floor is the MORE permissive rule at long range — it passes a 400 mm
# pair at 3.5 m carrying ~208 mm of error, worse than anything the ratio lets
# through. Default False so this change only ever ADDS acceptances; turn it on
# once you have watched the ratio reported in each fix for a session.
TAGS_SPREAD_RATIO_STRICT = _env("TAGS_SPREAD_RATIO_STRICT", False)
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

# ---- temporal robustness: consensus instead of per-frame strictness --------
# Every fix used to have to pass its own gates alone, so those gates had to be
# strict enough that ONE bad frame could not do damage — which meant discarding
# a great many good frames to catch a few bad ones. This shifts part of that
# burden onto agreement ACROSS TIME: the per-frame thresholds get a looser
# second tier, and a frame that only clears the loose tier can never move the
# anchor by itself — it may only contribute to a consensus of several recent
# observations that agree with each other. A single bad frame that slips past a
# looser gate is then diluted by its neighbours instead of being applied, so the
# accept rate goes up without the risk going up with it.
# It is the same challenge-counter idea the radar ghost guard and the existing
# TAGS_BIG_FIX_MM confirmation already use, extended to the rest of the pipeline
# and made robust: a median with outlier trimming rather than an all-must-agree
# veto, so one outlier among four good observations is discarded instead of
# voiding the whole run.
TAGS_TEMPORAL_ENABLED = _env("TAGS_TEMPORAL_ENABLED", True)
# How long an observation stays eligible, and how many are kept. The window is
# cleared whenever the rover moves — a measurement of where it WAS is not a
# measurement of where it is.
TAGS_WINDOW_S = _env("TAGS_WINDOW_S", 6.0)
TAGS_WINDOW_N = _env("TAGS_WINDOW_N", 12)
# Distance from the window median beyond which an observation is trimmed as an
# outlier. Same scale as TAGS_AGREE_MM, which it generalises.
TAGS_TRIM_MM = _env("TAGS_TRIM_MM", 150.0)
# The loose tier, as multiples of the strict thresholds. A frame inside these
# but outside the strict ones is consensus-only. 1.6 x 6.0 px = 9.6 px of RMS
# and 1.5 x 12 deg = 18 deg of yaw residual: wide enough to recover the frames
# that were being thrown away for ordinary map/detector noise, narrow enough
# that a genuinely broken solve (the 21 deg outliers) is still refused outright.
TAGS_LOOSE_RMS_SCALE = _env("TAGS_LOOSE_RMS_SCALE", 1.6)
TAGS_LOOSE_YAW_SCALE = _env("TAGS_LOOSE_YAW_SCALE", 1.5)
# How many agreeing observations a LOOSE-tier frame needs before the anchor
# moves. Strict-tier frames keep today's behaviour: a small correction applies
# immediately (TAGS_SMALL_FIX_CONSENSUS_N = 1) and a large one still needs
# TAGS_CONFIRM_N, so nothing about the current accept path gets slower.
TAGS_LOOSE_CONSENSUS_N = _env("TAGS_LOOSE_CONSENSUS_N", 3)
TAGS_SMALL_FIX_CONSENSUS_N = _env("TAGS_SMALL_FIX_CONSENSUS_N", 1)

# ---- micro-parallax: manufacture a baseline instead of waiting for one -----
# When the rover can only see a narrow or single-column tag set, no amount of
# gating helps: the geometry itself is unrecoverable, and the fix has to come
# from somewhere else. It can jog a small, known lateral distance while tracking
# the same tags and use the T265's SHORT-timescale displacement — reliable over
# a sub-second move even though it drifts over minutes — as a synthetic second
# viewpoint. Two views of the same tags, separated by a measured baseline, fuse
# into one well-conditioned solve exactly the way stereo triangulation would
# (see tag_localizer.solve_multiview for why this needs no new solver: the
# second view's object points are simply translated back along the baseline).
# Measured, single vertical column of two tags, 0.4 px corner noise, median
# lateral error, one view vs the fused pair:
#     range   baseline  pooled ratio   1 view    2 views
#     0.8 m     200 mm     0.250        3.1 mm    1.4 mm
#     1.2 m     200 mm     0.167       10.4 mm    4.6 mm
#     1.6 m     300 mm     0.188       24.3 mm    8.0 mm
#     2.0 m     300 mm     0.150       47.5 mm   15.6 mm
#     2.5 m     300 mm     0.120      100.9 mm   30.0 mm
# The pair beats the single view at every baseline tried, including 80 mm.
TAGS_PARALLAX_ENABLED = _env("TAGS_PARALLAX_ENABLED", True)
# Run it automatically when the localiser keeps reporting degenerate geometry
# while the rover is stopped. Set False to leave it operator-triggered only
# (POST /api/nav/parallax).
TAGS_PARALLAX_AUTO = _env("TAGS_PARALLAX_AUTO", True)
# How many consecutive degenerate-geometry reports trigger the auto attempt,
# and how long to wait before trying again either way.
TAGS_PARALLAX_TRIGGER_N = _env("TAGS_PARALLAX_TRIGGER_N", 3)
TAGS_PARALLAX_COOLDOWN_S = _env("TAGS_PARALLAX_COOLDOWN_S", 20.0)
# The jog is sized from the estimated range so it buys a useful angle rather
# than a fixed number of millimetres: baseline = TARGET_RATIO * range, clamped.
# 0.20 at 1 m is a 200 mm jog; at 3 m it saturates at the max below.
TAGS_PARALLAX_RATIO_TARGET = _env("TAGS_PARALLAX_RATIO_TARGET", 0.20)
TAGS_PARALLAX_BASELINE_MIN_MM = _env("TAGS_PARALLAX_BASELINE_MIN_MM", 120.0)
TAGS_PARALLAX_BASELINE_MAX_MM = _env("TAGS_PARALLAX_BASELINE_MAX_MM", 350.0)
# Accept gate on the FUSED pair. Lower than TAGS_MIN_SPREAD_RATIO on purpose and
# with a reason: the same tag is seen in both views, so the pooled cloud carries
# twice the corners (~sqrt(2) less corner noise), the baseline runs exactly
# across the line of sight rather than partly along it as panel-mounted tags
# usually do, and it is METRICALLY MEASURED rather than read from a map that may
# itself be wrong. At 0.12 the measured fused error is 15-30 mm, better than the
# ~106 mm a single view at ratio 0.186 delivers today and is trusted.
TAGS_PARALLAX_MIN_RATIO = _env("TAGS_PARALLAX_MIN_RATIO", 0.12)
# The pooled view must span at least this much, i.e. the rover must actually
# have moved. Guards the case where the jog stalled against something.
TAGS_PARALLAX_MIN_SPREAD_MM = _env("TAGS_PARALLAX_MIN_SPREAD_MM", 100.0)
# Strafe back afterwards, so the manoeuvre leaves the rover where it found it.
TAGS_PARALLAX_RETURN = _env("TAGS_PARALLAX_RETURN", True)
# Settle time after each strafe before grabbing the second view: motion blur
# ruins corner precision, which is the whole basis of the fix.
TAGS_PARALLAX_SETTLE_S = _env("TAGS_PARALLAX_SETTLE_S", 0.5)
# How long to wait for a FRESH detector observation at each viewpoint.
TAGS_PARALLAX_OBS_TIMEOUT_S = _env("TAGS_PARALLAX_OBS_TIMEOUT_S", 2.5)
# The fusion assumes the heading is the same at both viewpoints (the strafe is
# run with hold_yaw). If the T265 says yaw moved more than this between them,
# the assumption is broken and the pair is discarded rather than fused.
TAGS_PARALLAX_MAX_YAW_DRIFT_DEG = _env("TAGS_PARALLAX_MAX_YAW_DRIFT_DEG", 2.0)
# Extra clearance (mm) required beyond the normal footprint check before the
# rover is allowed to strafe into a spot to take the second view.
TAGS_PARALLAX_CLEARANCE_MM = _env("TAGS_PARALLAX_CLEARANCE_MM", 80.0)
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
