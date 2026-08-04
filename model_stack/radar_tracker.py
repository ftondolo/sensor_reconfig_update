"""Stable single-target mmWave tracker (works whether the target moves or not).

Designed for a demo space with ONE human in front of the rover. It does NOT
require motion to detect — a perfectly still person is tracked just like a moving
one. Robustness comes from:

  * POINT ACCUMULATION over a short window -> the sparse OOB cloud (~3-6 pts/frame)
    becomes dense enough to cluster reliably, and the centroid stops jumping.
  * A single persistent TRACK with continuity (associate the nearest candidate
    cluster each frame) + heavy position smoothing -> a steady bearing/range.
  * Doppler is only a *preference* for acquisition (prefer a moving cluster when
    one exists), never a gate -> stillness no longer means "no detection".
  * EGO-MOTION: while the rover is translating its own motion gives static clutter
    a fake Doppler, so we never *acquire* while moving (would grab clutter); the
    existing lock is maintained purely spatially, which is unaffected by ego-motion.

Usage:
    tk = RadarTracker()
    tk.update(points, t, rover_moving=False)   # points: [{x,y,z,v}], t: time.time()
    tgt = tk.target()                          # {range, az_deg, moving} or None
"""
import math
from collections import deque


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _single_link(points, eps):
    """Greedy single-link clustering on (x, y). Fine for the small accumulated cloud."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    eps2 = eps * eps
    for i in range(n):
        xi, yi = points[i]["x"], points[i]["y"]
        for j in range(i + 1, n):
            dx = xi - points[j]["x"]
            dy = yi - points[j]["y"]
            if dx * dx + dy * dy <= eps2:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(points[i])
    return list(groups.values())


def _stats(cl):
    xs = [p["x"] for p in cl]
    ys = [p["y"] for p in cl]
    vs = [abs(p.get("v", 0.0)) for p in cl]
    snrs = [p.get("snr", 0) for p in cl]     # per-point SNR (0.1 dB units), 0 if absent
    xc, yc = _median(xs), _median(ys)        # median centroid -> robust to outliers
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return {"xc": xc, "yc": yc, "n": len(cl), "extent": extent, "max_v": max(vs),
            "range": math.hypot(xc, yc),
            "snr_peak": max(snrs) if snrs else 0, "snr_sum": sum(snrs)}


class RadarTrackerConfig:
    # Region of interest (a person in front of the radar).
    y_min, y_max = 0.3, 6.0      # forward range gate (m)
    x_abs_max = 3.0             # lateral half-width (m)
    z_abs_max = 1.5            # height (m); OOB z is coarse
    # Accumulate points over this window before clustering (denser, steadier).
    accum_sec = 0.35
    # Clustering on the accumulated cloud.
    eps = 0.6                  # neighbour distance (m)
    min_pts = 3               # min points for a candidate cluster
    extent_min, extent_max = 0.10, 1.50   # person-sized
    # Acquisition preference (NOT a gate): prefer a cluster with motion.
    v_move = 0.12
    # Reflection-intensity gate (0.1 dB units) — only ACQUIRE a cluster that
    # reflects strongly enough to be a person, so an empty scene's weak clutter
    # never locks on. 0 = disabled. Maintenance is NOT gated (keep a valid lock).
    min_snr_peak = 0     # the cluster's strongest point must exceed this
    min_snr_sum = 0      # the cluster's total reflected energy must exceed this
    # Continuity / maintenance.
    gate_radius = 1.0          # m: associate a candidate to the existing track within this
    # Acquisition confirmation (ghost gate): a FRESH lock requires the winning
    # moving cluster to persist within acquire_confirm_radius of itself for
    # acquire_confirm_s seconds. A transient multipath ghost stops clustering
    # ~accum_sec after its points cease, so a few-frame ghost never confirms;
    # a real person's continuous returns do. Keep confirm_s above accum_sec.
    # 0 = the old instant single-window acquisition. Maintenance is unaffected.
    acquire_confirm_s = 1.0
    acquire_confirm_radius = 0.6
    hold_timeout = 1.5         # s: keep a spatially-unsupported track this long, then drop
    pos_smooth = 0.3           # EMA: new-position weight (lower = smoother, more lag)
    # Motion confirmation: a lock must have shown motion within this window or it's
    # treated as static clutter and dropped. This is what separates a person (who
    # sways/breathes/walks) from an empty scene's static clutter (SNR can't).
    confirm_hold = 8.0


class RadarTracker:
    def __init__(self, cfg=None):
        self.c = cfg or RadarTrackerConfig()
        self.x = None            # tracked position in the radar frame (m)
        self.y = None
        self.t = 0.0             # time of last position update
        self.acquired = False
        self._moving = False
        self._last_motion_t = 0.0  # last time the lock showed motion (confirmation)
        self._snr_peak = 0       # reflection intensity of the locked cluster (0.1 dB)
        self._snr_sum = 0
        self._buf = deque()      # accumulation: (t, [roi points])
        self._pend = None        # pending acquisition {x, y, t0} awaiting confirmation

    def _roi(self, points):
        c = self.c
        return [p for p in points
                if c.y_min < p["y"] < c.y_max
                and abs(p["x"]) < c.x_abs_max and abs(p["z"]) < c.z_abs_max]

    def _cloud(self, roi, t):
        self._buf.append((t, roi))
        cutoff = t - self.c.accum_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        cloud = []
        for _, pts in self._buf:
            cloud.extend(pts)
        return cloud

    def _candidates(self, cloud):
        c = self.c
        out = []
        for cl in _single_link(cloud, c.eps):
            if len(cl) < c.min_pts:
                continue
            st = _stats(cl)
            if c.extent_min <= st["extent"] <= c.extent_max:
                out.append(st)
        return out

    def _drop(self):
        self.acquired = False
        self.x = self.y = None
        self._pend = None

    def _qualified(self, cands):
        # Clusters eligible for a FRESH lock: MOVING and sufficiently reflective,
        # so an empty/static scene never locks on (clutter reflects as strongly
        # as a person, so SNR alone can't separate them; motion can).
        c = self.c
        return [st for st in cands if st["max_v"] > c.v_move
                and st["snr_peak"] >= c.min_snr_peak and st["snr_sum"] >= c.min_snr_sum]

    def _acquire(self, cands, t):
        """Fresh lock with confirmation ("last X positions" ghost gate): the
        best moving cluster becomes a PENDING candidate and must persist within
        acquire_confirm_radius of itself for acquire_confirm_s before it is
        returned as the track. A transient multipath ghost is only clusterable
        for its own duration + accum_sec, so it never confirms; a real person's
        continuous returns do (the pending point follows them as they walk).
        Most-moving cluster is the human; tiebreak by total SNR."""
        c = self.c
        qual = self._qualified(cands)
        if not qual:
            self._pend = None
            return None
        if c.acquire_confirm_s <= 0:      # legacy instant single-window acquire
            return max(qual, key=lambda st: (st["max_v"], st["snr_sum"]))
        p = self._pend
        if p is not None:
            near = [st for st in qual
                    if math.hypot(st["xc"] - p["x"], st["yc"] - p["y"]) <= c.acquire_confirm_radius]
            if near:
                st = min(near, key=lambda s: math.hypot(s["xc"] - p["x"], s["yc"] - p["y"]))
                p["x"], p["y"] = st["xc"], st["yc"]   # follow a walking person
                if (t - p["t0"]) >= c.acquire_confirm_s:
                    self._pend = None
                    return st
                return None
        best = max(qual, key=lambda st: (st["max_v"], st["snr_sum"]))
        self._pend = {"x": best["xc"], "y": best["yc"], "t0": t}
        return None

    def update(self, points, t, rover_moving=False):
        c = self.c
        cands = self._candidates(self._cloud(self._roi(points), t))

        chosen = None
        if self.acquired and self.x is not None:
            # Maintain spatially — works through target stillness AND ego-motion.
            near = [st for st in cands
                    if math.hypot(st["xc"] - self.x, st["yc"] - self.y) <= c.gate_radius]
            if near:
                chosen = min(near, key=lambda st: math.hypot(st["xc"] - self.x, st["yc"] - self.y))
        elif not rover_moving:
            # acquire only a CONFIRMED moving cluster, only while still
            chosen = self._acquire(cands, t)

        if chosen is not None:
            if not self.acquired or self.x is None:
                self.x, self.y = chosen["xc"], chosen["yc"]
            else:
                self.x = c.pos_smooth * chosen["xc"] + (1 - c.pos_smooth) * self.x
                self.y = c.pos_smooth * chosen["yc"] + (1 - c.pos_smooth) * self.y
            self.t = t
            self.acquired = True
            self._moving = chosen["max_v"] > c.v_move
            self._snr_peak = chosen["snr_peak"]
            self._snr_sum = chosen["snr_sum"]
        elif self.acquired and (t - self.t) >= c.hold_timeout:
            self._drop()                    # lost spatial support entirely

        # Motion confirmation: refresh the timer when the lock shows motion (or
        # while following, which implies the target is moving — ego-motion masks
        # its Doppler). If it hasn't moved for confirm_hold WHILE THE ROVER IS
        # STILL, it's static clutter (or a frozen person) -> drop.
        if self.acquired:
            if rover_moving or (chosen is not None and chosen["max_v"] > c.v_move):
                self._last_motion_t = t
            elif not rover_moving and (t - self._last_motion_t) >= c.confirm_hold:
                self._drop()

    def target(self):
        if not self.acquired or self.x is None:
            return None
        return {
            "range": math.hypot(self.x, self.y),
            "az": math.degrees(math.atan2(self.x, self.y)),   # 0 = ahead, + = right
            "moving": self._moving,
            "snr_peak": self._snr_peak, "snr_sum": self._snr_sum,
        }

    def reset(self):
        self.x = self.y = None
        self.acquired = False
        self._pend = None
        self._buf.clear()
