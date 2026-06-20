#!/usr/bin/env python3
"""Off-hardware tests for two heading-hold features of purely_control.

What this verifies (all in MOCK mode — no T265, no ROS, no real motion):

  A. MID-MOVE yaw correction, one-shot kick
     While driving forward, inject a sudden ~15 deg heading kick. The PI
     heading-hold loop (T265RoverService._yaw_cmd) must drive the heading back
     to the start value before the move ends. We check the kick really showed
     up (peak deviation ~15 deg) and was corrected out (final ~0 deg).

  B. MID-MOVE yaw correction, steady drift
     Apply a steady 8 deg/s external yaw push for the whole move (a mecanum
     base that creeps in yaw while strafing). The integral term must null this
     CONSTANT bias so heading is held during the move and back to ~0 at the end.

  C. END-STAGE correction is TIME-BOUNDED (ARRIVE_SETTLE_TIMEOUT)
     Apply a yaw push so strong the PI loop *cannot* pull heading inside YAW_TOL
     (steady-state error stays a few degrees). Position is reached quickly; the
     rover then rotates in place to fix heading — but only for
     ARRIVE_SETTLE_TIMEOUT seconds, after which the move finishes as "arrived"
     anyway (never spins forever). We run several ARRIVE_SETTLE_TIMEOUT values
     and confirm the measured "reached-position -> move-done" time tracks the
     configured bound, and that the move ends "arrived" (not "timeout") with a
     residual heading error left over (proof it gave up rather than spun on).

The mock integrates commanded velocity into a fake pose and exposes two test
hooks for exactly this purpose:
    _mock_yaw_kick   (rad)    one-shot heading kick, applied on the next pose tick
    _mock_yaw_drift  (rad/s)  continuous external yaw rate

Usage
-----
    # from the demo_v2 root (or anywhere):
    python purely_control/test_yaw_control.py
    # or as a module:
    python -m purely_control.test_yaw_control

    # options:
    python purely_control/test_yaw_control.py --verbose      # per-sample-ish detail
    python purely_control/test_yaw_control.py --only C       # run one group (A|B|C)

Exit code is 0 when every check passes, 1 otherwise. No hardware is required or
used: the test forces MOCK mode even if a real T265 is plugged in, so it only
ever exercises the control logic.
"""
import argparse
import math
import os
import sys
import time

# Make `import purely_control...` work whether run as a script or a module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import purely_control.t265_rover as tr            # noqa: E402
from purely_control.t265_rover import T265RoverService  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _force_mock():
    """Guarantee MOCK mode (and the test hooks) regardless of attached hardware.

    Nulling the module's optional deps makes _open_t265()/_init_ros() fall
    straight into the mock path, which is what creates _mock_yaw_kick/_drift.
    """
    tr.rs = None
    tr.rospy = None
    tr.Twist = None


def deg(rad):
    return math.degrees(rad)


def make_rover(**overrides):
    r = T265RoverService(allow_mock=True, **overrides)
    r.start(wait_for_pose=True, timeout=5.0)
    if not r.using_mock:
        raise SystemExit("FATAL: expected MOCK mode for a control-logic test")
    return r


def set_drift(r, dps):
    """Set a continuous external yaw push, in deg/s (0 to clear)."""
    with r._lock:
        r._mock_yaw_drift = math.radians(dps)


def inject_kick(r, kick_deg):
    """Inject a one-shot heading kick, in deg, applied on the next pose tick."""
    with r._lock:
        r._mock_yaw_kick += math.radians(kick_deg)


def run_move(r, right=0.0, forward=0.0, hold_yaw=None,
             kick_deg=0.0, kick_at=0.0, drift_dps=0.0,
             sample_hz=50, max_wall=30.0):
    """Drive a non-blocking move while sampling heading + remaining distance.

    Returns (MoveResult, samples, pos_tol) where samples is a list of
    (t_rel_s, yaw_dev_rad, remaining_m). yaw_dev is the heading error vs the
    pose captured just before the move (i.e. the start heading that is held).
    """
    start = r.get_pose()
    yaw0, rx0, rf0 = start["yaw"], start["right"], start["forward"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    pos_tol = r.cfg.POS_TOL

    if drift_dps:
        set_drift(r, drift_dps)

    samples = []
    t0 = time.monotonic()
    goal = r.move(right=right, forward=forward, hold_yaw=hold_yaw, blocking=False)
    kicked = False
    dt = 1.0 / sample_hz
    while r.is_busy():
        now = time.monotonic() - t0
        if now > max_wall:
            break
        if kick_deg and not kicked and now >= kick_at:
            inject_kick(r, kick_deg)
            kicked = True
        p = r.get_pose()
        dev = tr._wrap(p["yaw"] - yaw0)
        dxr, dxf = p["right"] - rx0, p["forward"] - rf0
        rt = dxr * c + dxf * s            # travelled, start-heading frame
        ft = -dxr * s + dxf * c
        rem = math.hypot(right - rt, forward - ft)
        samples.append((now, dev, rem))
        time.sleep(dt)

    res = r.wait(goal)
    set_drift(r, 0.0)
    return res, samples, pos_tol


def peak_dev_deg(samples):
    return deg(max((abs(d) for _, d, _ in samples), default=0.0))


def time_position_reached(samples, pos_tol):
    """t_rel of the first sample whose remaining distance is within POS_TOL."""
    for t, _, rem in samples:
        if rem <= pos_tol:
            return t
    return None


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, label, ok, detail=""):
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"    [{mark}] {label}" + (f"   ({detail})" if detail else ""))
        return ok


# --------------------------------------------------------------------------- #
# Test A — mid-move correction of a sudden kick
# --------------------------------------------------------------------------- #
def test_A(ck, verbose=False):
    print("\n== A. mid-move yaw correction — one-shot 15 deg kick ==")
    r = make_rover(MAX_LINEAR=0.25)
    try:
        res, samples, _ = run_move(r, forward=1.0, kick_deg=15.0, kick_at=1.0)
    finally:
        r.shutdown()
    peak = peak_dev_deg(samples)
    final = abs(deg(res.yaw_error))
    tol = deg(r.cfg.YAW_TOL)
    print(f"    result: {res!r}")
    print(f"    peak heading deviation during move: {peak:.1f} deg")
    print(f"    final heading error (vs start):     {final:.2f} deg   (YAW_TOL={tol:.2f})")
    if verbose:
        for t, d, rem in samples[::10]:
            print(f"      t={t:4.1f}s  yaw_dev={deg(d):6.1f}deg  remain={rem*1000:5.0f}mm")
    ck.check("move arrived", res.ok and res.reason == "arrived", res.reason)
    ck.check("kick actually perturbed heading (peak >= 12 deg)", peak >= 12.0,
             f"peak={peak:.1f}")
    ck.check("heading corrected back to ~start (final <= 2 deg)", final <= 2.0,
             f"final={final:.2f}")


# --------------------------------------------------------------------------- #
# Test B — mid-move rejection of a steady drift
# --------------------------------------------------------------------------- #
def test_B(ck, verbose=False):
    print("\n== B. mid-move yaw correction — steady 8 deg/s drift ==")
    r = make_rover(MAX_LINEAR=0.25)
    try:
        res, samples, _ = run_move(r, forward=1.0, drift_dps=8.0)
    finally:
        r.shutdown()
    peak = peak_dev_deg(samples)
    final = abs(deg(res.yaw_error))
    print(f"    result: {res!r}")
    print(f"    max heading deviation held during move: {peak:.1f} deg")
    print(f"    final heading error (vs start):         {final:.2f} deg")
    if verbose:
        for t, d, rem in samples[::10]:
            print(f"      t={t:4.1f}s  yaw_dev={deg(d):6.1f}deg  remain={rem*1000:5.0f}mm")
    ck.check("move arrived", res.ok and res.reason == "arrived", res.reason)
    ck.check("steady drift held bounded during move (peak <= 6 deg)", peak <= 6.0,
             f"peak={peak:.1f}")
    ck.check("constant bias nulled by integral (final <= 2 deg)", final <= 2.0,
             f"final={final:.2f}")


# --------------------------------------------------------------------------- #
# Test C — the end-stage correction is time-bounded
# --------------------------------------------------------------------------- #
def test_C(ck, verbose=False):
    print("\n== C. end-stage in-place correction is bounded by ARRIVE_SETTLE_TIMEOUT ==")
    # 30 deg/s = 0.52 rad/s: below MAX_YAW (0.6) so the loop does NOT spin up,
    # but above what the (integral-capped) PI can null -> heading never reaches
    # YAW_TOL, so the in-place phase always runs to its bound.
    DRIFT_DPS = 30.0
    bounds = [0.0, 0.6, 1.5]
    rows = []
    for bound in bounds:
        r = make_rover(MAX_LINEAR=0.25, ARRIVE_SETTLE_TIMEOUT=bound)
        try:
            res, samples, pos_tol = run_move(r, forward=0.10, drift_dps=DRIFT_DPS)
        finally:
            r.shutdown()
        t_pos = time_position_reached(samples, pos_tol)
        in_place = (res.elapsed - t_pos) if t_pos is not None else float("nan")
        rows.append((bound, res, in_place, t_pos))
        print(f"    ARRIVE_SETTLE_TIMEOUT={bound:.1f}s -> reason={res.reason:8} "
              f"elapsed={res.elapsed:4.2f}s  pos_reached@{(t_pos or 0):4.2f}s  "
              f"in-place={in_place:4.2f}s  residual_yaw={abs(deg(res.yaw_error)):4.1f}deg")
        if verbose:
            for t, d, rem in samples[::10]:
                print(f"      t={t:4.2f}s  yaw_dev={deg(d):6.1f}deg  remain={rem*1000:5.0f}mm")

    yaw_tol_deg = deg(tr.Config().YAW_TOL)

    # All moves must END (as "arrived", never "timeout"/hung) ...
    ck.check("all moves finished as 'arrived' (not timeout/hung)",
             all(res.reason == "arrived" for _, res, _, _ in rows),
             ",".join(res.reason for _, res, _, _ in rows))
    # ... with a residual heading error left over (proves it gave up on yaw) ...
    ck.check("residual heading error remains > YAW_TOL (gave up, didn't spin forever)",
             all(abs(deg(res.yaw_error)) > yaw_tol_deg for _, res, _, _ in rows),
             "residuals=" + ",".join(f"{abs(deg(res.yaw_error)):.1f}" for _, res, _, _ in rows))
    # ... and the in-place time must track the configured bound.
    for bound, res, in_place, _ in rows:
        ck.check(f"in-place time ~= bound for ARRIVE_SETTLE_TIMEOUT={bound:.1f}s",
                 abs(in_place - bound) <= 0.35,
                 f"measured={in_place:.2f}s, bound={bound:.1f}s")
    # Monotonic: a larger bound spends more time correcting.
    in_places = [ip for _, _, ip, _ in rows]
    ck.check("larger bound -> longer in-place correction (monotonic)",
             in_places[0] <= in_places[1] <= in_places[2],
             f"{in_places[0]:.2f} <= {in_places[1]:.2f} <= {in_places[2]:.2f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="purely_control heading-hold tests (mock)")
    ap.add_argument("--only", choices=["A", "B", "C"], help="run only one group")
    ap.add_argument("--verbose", action="store_true", help="print sampled trajectories")
    args = ap.parse_args()

    _force_mock()
    print("purely_control heading-hold tests  —  MOCK mode (no hardware, no ROS)")

    ck = Checker()
    groups = {"A": test_A, "B": test_B, "C": test_C}
    for key in (["A", "B", "C"] if not args.only else [args.only]):
        groups[key](ck, verbose=args.verbose)

    print("\n" + ("ALL CHECKS PASSED" if ck.failures == 0
                  else f"{ck.failures} CHECK(S) FAILED"))
    return 0 if ck.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
