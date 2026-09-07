"""Temporal fusion and parallax geometry for ArUco fixes.

Two ideas live here, both of which exist to move trust OFF the individual frame:

FixWindow (temporal robustness)
    Every fix used to have to pass its own gates alone, so the gates had to be
    strict enough that a single bad frame could not do damage — which meant
    throwing away a great many perfectly good frames to catch a few bad ones.
    A rolling window changes the bargain: loosen the per-frame thresholds a
    little, then require several recent observations to AGREE before the anchor
    actually moves. A bad frame that slips through a looser gate is diluted by
    the frames around it instead of being applied outright, and the accept rate
    goes up rather than down.

    The agreement test is a median with outlier trimming, not a mean with an
    all-must-agree veto. The old confirmation run rejected the whole run if any
    one member disagreed, which is exactly backwards: one outlier among four
    good observations is the case robust statistics are FOR. The median is
    unmoved by it, the trim discards it, and the remaining agreement still
    counts.

    Observations are stored as ABSOLUTE measured rover positions in map mm, not
    as residuals against the current pose. That matters: a residual goes stale
    the moment the anchor is eased, whereas an absolute measurement stays valid
    and can be compared against whatever the pose is when it is finally used.

Parallax geometry
    Small pure helpers for the micro-parallax manoeuvre — where the rover jogs a
    known lateral distance to manufacture a second viewpoint. Kept here, out of
    the navigator, so they can be tested without a rover.
"""
import math
import threading
import time


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def robust_center(points, trim_mm):
    """Component-wise median, then the mean of everything within trim_mm of it.

    Returns (cx, cz, kept_indices). The median locates the consensus while
    ignoring outliers entirely; the trimmed mean then uses ALL the inliers, so
    the answer keeps the noise-averaging benefit of a mean without inheriting a
    mean's sensitivity to the one frame that went wrong.
    """
    if not points:
        return None, None, []
    mx = _median([p[0] for p in points])
    mz = _median([p[1] for p in points])
    keep = [i for i, p in enumerate(points)
            if math.hypot(p[0] - mx, p[1] - mz) <= trim_mm]
    if not keep:
        return mx, mz, []
    cx = sum(points[i][0] for i in keep) / len(keep)
    cz = sum(points[i][1] for i in keep) / len(keep)
    return cx, cz, keep


class FixWindow:
    """Rolling window of recent tag observations, with a robust consensus.

    Every entry is one SOLVE, already through the per-frame gates at some tier:

        "strict" — clears the original thresholds; trustworthy on its own.
        "loose"  — clears the widened thresholds only; may contribute to a
                   consensus but must never move the anchor by itself.

    The window is cleared whenever the rover moves, because a measurement of
    where the rover was is not a measurement of where it is.
    """

    def __init__(self, window_s=6.0, max_n=12, trim_mm=150.0, clock=time.time):
        self.window_s = float(window_s)
        self.max_n = int(max_n)
        self.trim_mm = float(trim_mm)
        # One clock for everything. Observation timestamps come from the
        # detector's time.time(), and ageing compares against the same source —
        # mixing a caller-supplied stamp with a separately-read wall clock is
        # how a window silently empties itself. Injectable so the behaviour can
        # be tested without sleeping.
        self._now = clock
        self._obs = []          # [{t, x, z, yaw, rms, n_tags, tier, key}]
        self._last_key = None
        self.cleared_reason = "init"
        # The navigator fills this from the monitor and drive threads while the
        # web layer reads it for state(); every public method takes the lock so
        # a summary can never catch the list mid-prune.
        self._lock = threading.RLock()

    # -------------------------------------------------------------- mutation
    def clear(self, why=""):
        with self._lock:
            self._obs = []
            self._last_key = None
            self.cleared_reason = why or "cleared"

    def prune(self, now=None):
        with self._lock:
            now = self._now() if now is None else now
            if self.window_s > 0:
                self._obs = [o for o in self._obs if now - o["t"] <= self.window_s]
            if self.max_n > 0:
                del self._obs[:-self.max_n]

    def add(self, x, z, yaw_deg, rms_px, n_tags, tier="strict", t=None, key=None):
        """Record one observation. Returns False if it is a REPEAT.

        De-duplication is not cosmetic. The detector publishes at TAGS_DETECT_HZ
        while the navigator polls faster and from two places (the idle monitor
        and between legs), so the same frame is seen several times. Counting it
        repeatedly would let ONE frame satisfy an N-frame consensus all by
        itself, which would quietly undo the entire point of this class.
        """
        with self._lock:
            k = key if key is not None else t
            if k is not None and k == self._last_key:
                return False
            self._last_key = k
            now = self._now() if t is None else float(t)
            self._obs.append({"t": now, "x": float(x), "z": float(z),
                              "yaw": float(yaw_deg), "rms": float(rms_px),
                              "n_tags": int(n_tags), "tier": str(tier), "key": k})
            self.prune()
            return True

    # ------------------------------------------------------------- consensus
    def consensus(self, need, now=None):
        """Robust agreement over the window, or None with the reason why not.

        Returns {x, z, yaw_deg, n, need, kept, dropped, span_s, scatter_mm,
                 strict, loose, rms_px}.
        """
        with self._lock:
            self.prune(now=now)
            need = max(1, int(need))
            if len(self._obs) < need:
                return None, ("%d of %d agreeing observations so far"
                              % (len(self._obs), need))
            pts = [(o["x"], o["z"]) for o in self._obs]
            cx, cz, keep = robust_center(pts, self.trim_mm)
            if len(keep) < need:
                return None, ("only %d of %d recent observations agree within "
                              "%.0f mm (need %d) — the view is unstable, not the map"
                              % (len(keep), len(self._obs), self.trim_mm, need))
            kept = [self._obs[i] for i in keep]
            ts = [o["t"] for o in kept]
            scatter = max(math.hypot(o["x"] - cx, o["z"] - cz) for o in kept)
            return {"x": cx, "z": cz,
                    "yaw_deg": _median([o["yaw"] for o in kept]),
                    "n": len(kept), "need": need,
                    "kept": len(kept), "dropped": len(self._obs) - len(kept),
                    "span_s": round(max(ts) - min(ts), 2),
                    "scatter_mm": round(scatter, 1),
                    "strict": sum(1 for o in kept if o["tier"] == "strict"),
                    "loose": sum(1 for o in kept if o["tier"] != "strict"),
                    "rms_px": round(_median([o["rms"] for o in kept]), 2),
                    "n_tags": int(_median([o["n_tags"] for o in kept]))}, None

    def summary(self):
        """Small dict for the UI: what is currently in the window."""
        with self._lock:
            self.prune()
            if not self._obs:
                return {"n": 0, "reason": self.cleared_reason}
            pts = [(o["x"], o["z"]) for o in self._obs]
            cx, cz, keep = robust_center(pts, self.trim_mm)
            ts = [o["t"] for o in self._obs]
            return {"n": len(self._obs), "agree": len(keep),
                    "span_s": round(max(ts) - min(ts), 2),
                    "strict": sum(1 for o in self._obs if o["tier"] == "strict"),
                    "loose": sum(1 for o in self._obs if o["tier"] != "strict")}

    def __len__(self):
        with self._lock:
            return len(self._obs)


# --------------------------------------------------------------- parallax
def lateral_offset_xz(x, z, yaw_deg, mm):
    """Map point ``mm`` to the rover's RIGHT of (x, z) at heading yaw_deg.

    Map convention: +x is right and forward is -z at yaw 0, so the body 'right'
    unit vector in map axes is (cos yaw, +sin yaw) in (x, z). Negative mm is to
    the left."""
    g = math.radians(yaw_deg)
    return x + mm * math.cos(g), z + mm * math.sin(g)


def map_delta_to_world_vec(dx_map, dz_map):
    """A map-plane displacement as the 3-vector the localiser's world uses.

    The tag world frame is (X = map x, Y = height, Z = map z), so a ground-plane
    move has no Y component. Exists so the sign convention is stated once, in
    one place, rather than re-derived at the call site."""
    return (float(dx_map), 0.0, float(dz_map))


def baseline_quality(baseline_mm, range_mm, want_ratio):
    """How much angular baseline a jog of this size buys at this range.

    Returns (ratio, enough, want_mm) where want_mm is the jog that WOULD reach
    want_ratio. Used both to size the jog before making it and to explain
    afterwards why a short one was not enough."""
    if not range_mm or range_mm <= 0:
        return None, False, None
    ratio = float(baseline_mm) / float(range_mm)
    want_mm = float(want_ratio) * float(range_mm) if want_ratio else None
    return ratio, (want_ratio is not None and ratio >= float(want_ratio)), want_mm
