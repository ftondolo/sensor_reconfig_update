"""Synthetic checks for the tag-geometry changes. No rover, no camera.

Run:  python3 rover_ui/test_tag_geometry.py

Covers the three things that are easy to get subtly, silently wrong:

  * the ANGULAR-BASELINE gate (spread / range) — that it opens for the
    close-range narrow observations it was meant to rescue, still closes for
    the far-range ones, and never rejects anything the old absolute floor
    accepted;
  * the TWO-VIEW FUSION — that translating the second view's object points back
    along a known baseline really does recover the pose, and beats the
    single-view solve on the degenerate single-column geometry it exists for;
  * the CONSENSUS WINDOW — that a median with trimming survives an outlier the
    old all-must-agree run would have thrown the whole batch away for, and that
    one frame polled repeatedly cannot satisfy an N-frame consensus.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))
sys.path.insert(0, os.path.dirname(_HERE))

import cv2                                   # noqa: E402
import importlib.util                        # noqa: E402


def _load(name, path):
    """Import a single module by path.

    backend/nav/__init__.py pulls in the whole Navigator (and with it the T265
    service, torch, pyrealsense...), none of which this test needs and most of
    which is not installed on a dev laptop. The two modules under test are
    dependency-free apart from numpy and cv2, so load them directly."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TL = _load("_tag_localizer", os.path.join(_HERE, "backend", "detection", "tag_localizer.py"))
tag_fusion = _load("_tag_fusion", os.path.join(_HERE, "backend", "nav", "tag_fusion.py"))

W, H, F = 640, 480, 600.0
K = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1]], float)
D = np.zeros(5)
SZ = 190.0
FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def tag_corners(cx, cy, cz, size=SZ):
    h = size / 2.0
    return np.array([[cx - h, cy + h, cz], [cx + h, cy + h, cz],
                     [cx + h, cy - h, cz], [cx - h, cy - h, cz]], float)


def world_to_cam(yaw):
    return np.array([[math.cos(yaw), 0, math.sin(yaw)],
                     [0, -1, 0],
                     [math.sin(yaw), 0, -math.cos(yaw)]], float)


def make_loc(tags, **kw):
    """A TagLocalizer wired to a synthetic tag table, no ArUco detector."""
    loc = TL.TagLocalizer.__new__(TL.TagLocalizer)
    loc.tags = tags
    loc.min_tags = kw.get("min_tags", 1)
    loc.min_spread_mm = kw.get("min_spread_mm", 400.0)
    loc.min_spread_ratio = kw.get("min_spread_ratio", 0.18)
    loc.min_spread_floor_mm = kw.get("min_spread_floor_mm", 150.0)
    loc.spread_ratio_strict = kw.get("spread_ratio_strict", False)
    loc.enabled = True
    loc._detector = None
    loc.last_reason = loc.last_reason_code = loc.last_geometry = None
    loc.last_observation = None
    loc._last_obj, loc._last_img = {}, {}
    return loc


def project(tags, C, yaw, noise, rng):
    """Render an observe()-shaped dict for a camera at C looking along yaw."""
    Rm = world_to_cam(yaw)
    tv = (-Rm @ np.asarray(C, float)).reshape(3, 1)
    rv, _ = cv2.Rodrigues(Rm)
    img = {}
    for t, ent in tags.items():
        pr, _ = cv2.projectPoints(ent[0], rv, tv, K, D)
        pr = pr.reshape(4, 2)
        if noise:
            pr = pr + rng.normal(0, noise, pr.shape)
        # float32 is what cv2.aruco actually hands back; the degeneracy this
        # code guards against is a knife-edge at exactly that precision.
        img[t] = pr.astype(np.float32).astype(np.float64)
    return {"ids": sorted(tags), "img": img,
            "obj": {t: tags[t][0] for t in tags}, "n": len(tags)}


# --------------------------------------------------------------------- 2a
def pair(spread, height=1200.0):
    return {0: (tag_corners(-spread / 2.0, height, 0.0), SZ, "+z", 0),
            1: (tag_corners(+spread / 2.0, height, 0.0), SZ, "+z", 0)}


def solve_err(tags, R, noise=0.4, trials=200, seed=11):
    rng = np.random.default_rng(seed)
    loc = make_loc(tags)
    C = np.array([0.0, 1150.0, float(R)])
    errs = []
    for _ in range(trials):
        obs = project(tags, C, 0.0, noise, rng)
        o = np.vstack([obs["obj"][t] for t in obs["ids"]])
        i = np.vstack([obs["img"][t] for t in obs["ids"]])
        r = loc._pnp(o, i, K, D)
        if r:
            errs.append(math.hypot(r[1][0] - C[0], r[1][2] - C[2]))
    return float(np.median(errs))


print("\n2a  angular-baseline gate")
rng0 = np.random.default_rng(1)

# the range estimate the gate depends on, against ground truth
for R in (800, 1500, 2500):
    tags = pair(250.0)
    loc = make_loc(tags)
    obs = project(tags, np.array([0.0, 1150.0, float(R)]), 0.0, 0.0, rng0)
    est = loc.range_estimate_mm(obs["ids"], obs["img"], K)
    check("range estimate at %d mm" % R, abs(est - R) / R < 0.06,
          "est %.0f mm (%.1f%% out)" % (est, 100 * abs(est - R) / R))

# the case the change exists for: 250 mm of spread, rejected at every range today
tags = pair(250.0)
loc = make_loc(tags)
for R, want, why in ((1000, True, "close: ratio 0.25"),
                     (2000, False, "mid: ratio 0.125"),
                     (3500, False, "far: ratio 0.071")):
    obs = project(tags, np.array([0.0, 1150.0, float(R)]), 0.0, 0.0, rng0)
    ok, geo = loc.geometry_ok(obs["ids"], obs["img"], K)
    err = solve_err(tags, R)
    check("250 mm spread at %d mm -> %s (%s)" % (R, "accept" if want else "reject", why),
          ok is want, "ratio %.3f, true error %.1f mm" % (geo["ratio"], err))

# nothing the old absolute floor accepted may now be rejected
for spread in (400.0, 650.0, 900.0):
    tags = pair(spread)
    loc = make_loc(tags)
    for R in (1000, 2000, 3500):
        obs = project(tags, np.array([0.0, 1150.0, float(R)]), 0.0, 0.0, rng0)
        ok, _g = loc.geometry_ok(obs["ids"], obs["img"], K)
        check("legacy pass preserved: %.0f mm at %d mm" % (spread, R), ok)

# the hard floor is not negotiable by ratio
tags = pair(80.0)
loc = make_loc(tags)
obs = project(tags, np.array([0.0, 1150.0, 300.0]), 0.0, 0.0, rng0)
ok, geo = loc.geometry_ok(obs["ids"], obs["img"], K)
check("80 mm spread rejected however close", not ok, geo["why"])

# ratio 0 restores the old behaviour exactly
tags = pair(250.0)
loc = make_loc(tags, min_spread_ratio=0.0)
obs = project(tags, np.array([0.0, 1150.0, 900.0]), 0.0, 0.0, rng0)
ok, _g = loc.geometry_ok(obs["ids"], obs["img"], K)
check("min_spread_ratio=0 restores the absolute floor", not ok)

# strict mode closes the far-range hole the floor leaves open
tags = pair(400.0)
loc = make_loc(tags, spread_ratio_strict=True)
obs = project(tags, np.array([0.0, 1150.0, 3500.0]), 0.0, 0.0, rng0)
ok, geo = loc.geometry_ok(obs["ids"], obs["img"], K)
check("strict mode rejects 400 mm at 3.5 m", not ok,
      "ratio %.3f, true error %.0f mm" % (geo["ratio"], solve_err(tags, 3500)))


# --------------------------------------------------------------------- 3c
print("\n3c  two-view fusion (single vertical column)")
COL = {0: (tag_corners(0.0, 900.0, 0.0), SZ, "+z", 0),
       1: (tag_corners(0.0, 1400.0, 0.0), SZ, "+z", 0)}

# exactness first: with no noise the fusion must return the true pose, or the
# object-point translation has a sign error that noise would hide.
loc = make_loc(COL)
C0 = np.array([0.0, 1150.0, 1500.0])
B = 250.0
a = project(COL, C0, 0.0, 0.0, np.random.default_rng(0))
b = project(COL, C0 + np.array([B, 0.0, 0.0]), 0.0, 0.0, np.random.default_rng(0))
fix = loc.solve_multiview([{"obs": a, "delta": (0, 0, 0)},
                           {"obs": b, "delta": (B, 0.0, 0.0)}], K, D,
                          min_ratio=0.0, min_spread_mm=50.0)
check("noiseless fusion returns view A's true pose", fix is not None
      and math.hypot(fix["x"] - C0[0], fix["z"] - C0[2]) < 1.0,
      "err %.2f mm" % math.hypot(fix["x"] - C0[0], fix["z"] - C0[2]) if fix else "no solve")

# a wrong-signed baseline must NOT quietly produce a plausible answer
bad = loc.solve_multiview([{"obs": a, "delta": (0, 0, 0)},
                           {"obs": b, "delta": (-B, 0.0, 0.0)}], K, D,
                          min_ratio=0.0, min_spread_mm=50.0)
check("a sign-flipped baseline is visibly wrong", bad is None or bad["rms_px"] > 2.0,
      "rms %.1f px" % bad["rms_px"] if bad else "refused")

# and the real claim: it beats the single view it is rescuing
print("    range   baseline   ratio    1 view    2 views")
for R, B in ((800, 200), (1200, 240), (1600, 300), (2000, 350), (2500, 350)):
    rng = np.random.default_rng(5)
    C0 = np.array([0.0, 1150.0, float(R)])
    e1, e2, ratio = [], [], None
    for _ in range(200):
        a = project(COL, C0, 0.0, 0.4, rng)
        b = project(COL, C0 + np.array([B, 0.0, 0.0]), 0.0, 0.4, rng)
        o = np.vstack([a["obj"][t] for t in a["ids"]])
        i = np.vstack([a["img"][t] for t in a["ids"]])
        r = loc._pnp(o, i, K, D)
        if r:
            e1.append(math.hypot(r[1][0] - C0[0], r[1][2] - C0[2]))
        f = loc.solve_multiview([{"obs": a, "delta": (0, 0, 0)},
                                 {"obs": b, "delta": (B, 0.0, 0.0)}], K, D,
                                min_ratio=0.0, min_spread_mm=50.0)
        if f:
            e2.append(math.hypot(f["x"] - C0[0], f["z"] - C0[2]))
            ratio = f["spread_ratio"]
    m1, m2 = float(np.median(e1)), float(np.median(e2))
    print("    %5d %9d   %5.3f  %7.1f   %7.1f" % (R, B, ratio, m1, m2))
    check("fusion beats the single view at %d mm" % R, m2 < m1 * 0.85,
          "%.1f -> %.1f mm" % (m1, m2))

# the gate must refuse a pair the rover did not actually separate
tiny = loc.solve_multiview([{"obs": a, "delta": (0, 0, 0)},
                            {"obs": a, "delta": (5.0, 0.0, 0.0)}], K, D,
                           min_ratio=0.12, min_spread_mm=100.0)
check("a 5 mm 'baseline' is refused, not fused", tiny is None, loc.last_reason or "")

# two views of different tags are not two views of the same thing
other = {9: (tag_corners(3000.0, 1200.0, 0.0), SZ, "+z", 1)}
merged = dict(COL); merged.update(other)
loc2 = make_loc(merged)
c = project(other, C0, 0.0, 0.0, np.random.default_rng(0))
check("views sharing no tag are refused",
      loc2.solve_multiview([{"obs": a, "delta": (0, 0, 0)},
                            {"obs": c, "delta": (B, 0, 0)}], K, D) is None)


# --------------------------------------------------------------------- 2b
print("\n2b  consensus window")


class Clock(object):
    """Controllable clock, so ageing can be exercised without sleeping."""

    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t


clk = Clock(104.0)
win = tag_fusion.FixWindow(window_s=60.0, max_n=12, trim_mm=150.0, clock=clk)
for i, (x, z) in enumerate([(1000, 2000), (1010, 1995), (995, 2010), (1005, 2001)]):
    win.add(x, z, 0.0, 3.0, 3, tier="strict", t=100.0 + i, key=i)
c, why = win.consensus(3)
check("four agreeing observations reach consensus", c is not None and c["n"] == 4,
      "centre (%.0f, %.0f)" % (c["x"], c["z"]) if c else why)

win.add(1900, 2900, 0.0, 3.0, 3, tier="loose", t=105.0, key="out")   # one wild outlier
c, why = win.consensus(3)
check("the outlier is trimmed, not fatal", c is not None and c["dropped"] == 1
      and abs(c["x"] - 1002) < 20,
      "centre (%.0f, %.0f), dropped %d" % (c["x"], c["z"], c["dropped"]) if c else why)

win2 = tag_fusion.FixWindow(window_s=60.0, trim_mm=150.0, clock=clk)
for i in range(4):
    win2.add(1000 + i * 400, 2000, 0.0, 3.0, 3, tier="loose", t=100.0 + i, key=i)
c, why = win2.consensus(3)
check("mutually disagreeing observations do NOT reach consensus", c is None, why or "")

win3 = tag_fusion.FixWindow(window_s=60.0, trim_mm=150.0, clock=clk)
first = win3.add(1000, 2000, 0.0, 3.0, 3, t=100.0, key="frame-A")
dup = [win3.add(1000, 2000, 0.0, 3.0, 3, t=100.0, key="frame-A") for _ in range(5)]
c, _w = win3.consensus(3)
check("one frame polled six times cannot fake a 3-frame consensus",
      first and not any(dup) and c is None and len(win3) == 1)

win3.clear("rover moving")
check("moving clears the window", len(win3) == 0)

# ageing: an observation older than the window must stop counting even if no
# new one ever arrives, or a parked rover would keep acting on a stale fix.
win4 = tag_fusion.FixWindow(window_s=6.0, trim_mm=150.0, clock=clk)
for i in range(3):
    win4.add(1000, 2000, 0.0, 3.0, 3, t=clk.t + i, key=i)
c, _w = win4.consensus(3)
check("fresh observations count", c is not None and c["n"] == 3)
clk.t += 30.0
c, why = win4.consensus(3)
check("stale observations age out of the window", c is None, why or "")

x1, z1 = tag_fusion.lateral_offset_xz(1000.0, 2000.0, 0.0, 200.0)
check("lateral offset at yaw 0 goes +x", abs(x1 - 1200) < 1e-6 and abs(z1 - 2000) < 1e-6,
      "(%.0f, %.0f)" % (x1, z1))
x2, z2 = tag_fusion.lateral_offset_xz(1000.0, 2000.0, 90.0, 200.0)
check("lateral offset at yaw 90 goes +z", abs(x2 - 1000) < 1e-6 and abs(z2 - 2200) < 1e-6,
      "(%.0f, %.0f)" % (x2, z2))

# ------------------------------------------------- 2b, through the navigator
print("\n2b  fix intake, end to end through Navigator._consume_fix")
sys.path.insert(0, os.path.dirname(_HERE))          # repo root: rectilinear_mm
try:
    import backend.nav.navigator as NAV             # noqa: E402
except Exception as _e:                             # pragma: no cover
    print("  SKIP  navigator import failed (%s)" % _e)
    NAV = None

if NAV is not None:
    cfg = NAV.config

    class FakeNav(NAV.Navigator):
        """A Navigator with only the fix-intake machinery wired up.

        Built with __new__ so none of the rover, detector or map plumbing has to
        exist: the intake path depends on the pose, the anchor and the window,
        and nothing else."""

        def __init__(self, x=1000.0, z=2000.0):
            import threading
            self._lock = threading.Lock()
            self._pose = {"x": x, "z": z, "yaw_deg": 0.0}
            self._tag_yaw_hist = []
            self._tag_run = []
            self._tag_applied = self._tag_rejected = 0
            self._tag_reject_reason = None
            self._tag_pending = self._tag_last = None
            self._anchor_map = (0.0, 0.0)
            self._drift_dist_mm = self._drift_turn_deg = 0.0
            self._drift_last_pose = None
            self._tag_win = tag_fusion.FixWindow(
                window_s=float(cfg.TAGS_WINDOW_S), max_n=int(cfg.TAGS_WINDOW_N),
                trim_mm=float(cfg.TAGS_TRIM_MM), clock=clk)

        def pose(self):
            return dict(self._pose)

    def fix_at(x, z, rms=2.0, yaw=0.0, n=3, t=None):
        return {"x": x, "z": z, "yaw_deg": yaw, "rms_px": rms, "n_tags": n,
                "ids": list(range(n)), "t": clk.t if t is None else t}

    clk.t = 1000.0
    nv = FakeNav()
    out = nv._consume_fix(fix_at(1040.0, 2000.0, rms=2.0))
    check("strict frame, small correction applies at once",
          out is not None and out["consensus"]["n"] == 1,
          "moved %s mm" % (out["applied_mm"] if out else "-"))

    # A loose frame is one that clears the widened thresholds only. rms limit is
    # TAGS_MAX_RMS_PX + per-tag allowance; go just above it but under the x1.6.
    lim = float(cfg.TAGS_MAX_RMS_PX) + float(cfg.TAGS_RMS_PER_TAG_PX) * 2
    loose_rms = lim * 1.2
    nv = FakeNav()
    applied = []
    for i in range(int(cfg.TAGS_LOOSE_CONSENSUS_N)):
        clk.t += 0.25
        applied.append(nv._consume_fix(fix_at(1040.0 + i, 2000.0, rms=loose_rms)))
    check("a loose frame is held until the consensus is reached",
          all(a is None for a in applied[:-1]) and applied[-1] is not None,
          "applied on frame %d of %d" % (len(applied), cfg.TAGS_LOOSE_CONSENSUS_N))
    check("the applied loose fix records what backed it",
          applied[-1]["tier"] == "loose"
          and applied[-1]["consensus"]["n"] >= int(cfg.TAGS_LOOSE_CONSENSUS_N))

    nv = FakeNav()
    reps = [nv._consume_fix(fix_at(1040.0, 2000.0, rms=loose_rms, t=1000.0))
            for _ in range(6)]
    check("polling one loose frame six times never applies it",
          all(r is None for r in reps))

    nv = FakeNav()
    out = nv._consume_fix(fix_at(1040.0, 2000.0, rms=lim * 2.5))
    check("a frame past even the loose RMS tier is refused outright",
          out is None and nv._tag_rejected == 1, nv._tag_reject_reason or "")

    nv = FakeNav()
    out = nv._consume_fix(fix_at(1040.0, 2000.0, yaw=float(cfg.TAGS_MAX_YAW_ERR_DEG) * 2))
    check("a frame past the loose yaw tier is refused outright", out is None)

    # A large correction still needs corroboration, exactly as before.
    nv = FakeNav()
    big = float(cfg.TAGS_BIG_FIX_MM) + 200.0
    res = []
    for i in range(int(cfg.TAGS_CONFIRM_N)):
        clk.t += 0.25
        res.append(nv._consume_fix(fix_at(1000.0 + big + i, 2000.0, rms=2.0)))
    check("a large correction still needs TAGS_CONFIRM_N agreeing frames",
          all(r is None for r in res[:-1]) and res[-1] is not None)
    check("the eased step is capped at TAGS_FIX_MAX_STEP_MM",
          res[-1]["applied_mm"] <= float(cfg.TAGS_FIX_MAX_STEP_MM) + 1,
          "%d mm of a %d mm residual" % (res[-1]["applied_mm"], res[-1]["resid_mm"]))

    # One bad frame among good ones is discarded, not fatal — the whole reason
    # for a median with trimming rather than an all-must-agree run.
    nv = FakeNav()
    seq = [1000.0 + big, 1000.0 + big + 10, 4000.0, 1000.0 + big + 5,
           1000.0 + big + 8]
    last = None
    for i, x in enumerate(seq):
        clk.t += 0.25
        last = nv._consume_fix(fix_at(x, 2000.0, rms=2.0)) or last
    check("one outlier among four good frames does not void the batch",
          last is not None and last["consensus"]["dropped"] >= 1,
          "dropped %s of %s" % (last["consensus"]["dropped"],
                                last["consensus"]["n"] + last["consensus"]["dropped"])
          if last else "never applied")

    # The window must not survive the rover moving.
    nv = FakeNav()
    clk.t += 0.25
    nv._consume_fix(fix_at(1000.0 + big, 2000.0, rms=2.0))
    nv._tag_win.clear("rover moving")
    clk.t += 0.25
    out = nv._consume_fix(fix_at(1000.0 + big, 2000.0, rms=2.0))
    check("a move resets the corroboration count", out is None)

print("\n%d checks failed" % len(FAILED))
for f in FAILED:
    print("   " + f)
sys.exit(1 if FAILED else 0)
