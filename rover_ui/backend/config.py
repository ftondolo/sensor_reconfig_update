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
# Emergency stop: while moving, if the rover footprint (car half-extent + this
# margin) overlaps a RAW obstacle, cancel the move and stop immediately. The
# planner keeps the center a full clearance away from this, so on a valid path
# it never trips — it only fires when the plan/target was wrong or the rover
# drifted into a keep-out.
NAV_OBSTACLE_STOP_MARGIN_MM = _env("NAV_OBSTACLE_STOP_MARGIN_MM", 150)
# purely_control wiring (passed through to T265RoverService).
NAV_CMD_VEL_TOPIC = _env("NAV_CMD_VEL_TOPIC", "/cmd_vel")
NAV_MAX_LINEAR = _env("NAV_MAX_LINEAR", 0.28)   # m/s translation cap during moves (initial)
# Start/stop behaviour (passed to T265RoverService). The controller defaults
# (RAMP_STEP 0.03, MIN_LINEAR 0.045) ramp too softly and floor too low for the
# WHEELTEC base — the first ticks sit under the wheels' stiction so it stalls
# on start and on the final crawl. Snappier accel + a higher floor break free
# immediately. RAMP_STEP is m/s added per 20 Hz tick (0.08 -> ~1.6 m/s^2);
# MIN_LINEAR is the speed floor — set just ABOVE the base's real stiction speed.
NAV_RAMP_STEP = _env("NAV_RAMP_STEP", 0.08)
NAV_MIN_LINEAR = _env("NAV_MIN_LINEAR", 0.12)
# Operator-settable speed range for the UI slider (clamps POST /api/nav/speed).
NAV_SPEED_MIN = _env("NAV_SPEED_MIN", 0.05)     # m/s slowest selectable
NAV_SPEED_MAX = _env("NAV_SPEED_MAX", 0.80)     # m/s fastest selectable
NAV_POS_TOL = _env("NAV_POS_TOL", 0.05)         # m arrival tolerance per segment
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
# Vision boxes must agree with the radar bearing within this (deg) to be trusted.
FOLLOW_FUSE_TOL = _env("FOLLOW_FUSE_TOL", 14.0)
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
