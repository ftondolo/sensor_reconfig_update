"""ArUco tag localisation: an ABSOLUTE position fix from markers on the panels.

The T265 drifts without bound because nothing in the loop ever observes where
the rover really is. Fiducial tags at surveyed positions close that loop: every
marker whose ID appears in map.json contributes four world<->image corner
correspondences, and one solvePnP over the pooled set returns the camera's pose
in map coordinates directly. That pose is drift-free by construction — it is
measured against the arena, not integrated from motion.

Geometry
--------
World frame is (X = map x, Y = height above the floor, Z = map z), millimetres.
Map z grows "backward" (toward the rover's start), so at yaw 0 the camera looks
along -Z. With the OpenCV camera convention (X right, Y down, Z forward) the
rows of world->camera are:

    camera X (right)   = ( cos yaw, 0,  sin yaw)
    camera Y (down)    = (       0, -1,        0)
    camera Z (forward) = ( sin yaw, 0, -cos yaw)

which is a proper right-handed rotation (X x Y = Z), so solvePnP is well posed
in this frame even though (x, y, z_map) is left-handed on its own.

Tag placement is authored per-obstacle in map.json as an offset from that
obstacle's GROUND-CENTRE, so moving a panel and updating its x/z carries its
tags along with it and nothing needs re-surveying.

Conditioning
------------
What makes a tag observation usable is not how far apart the tags are in
millimetres but how far apart they are IN ANGLE as seen from the camera — the
ANGULAR BASELINE, spread / range. That ratio is what governs how much
perspective difference is available to tell yaw apart from lateral translation.
As spread/range -> 0 the view degenerates toward orthographic, where strafing
sideways and rotating in yaw produce nearly identical image motion and the
solver answers with a confident, wrong, mirrored pose. See _geometry_ok.

Two views separated by a KNOWN baseline are the other way to buy that angle
(see solve_multiview): translating the object points of the second view back
along the baseline turns the pair into one well-conditioned pooled PnP, which
is how a single degenerate tag column is rescued without adding tags.
"""
import math

import numpy as np

try:
    import cv2
    _ARUCO = hasattr(cv2, "aruco")
except Exception:            # pragma: no cover - optional dependency
    cv2 = None
    _ARUCO = False


def _face_axes(facing):
    """(outward normal, in-plane 'right' direction) for a face, in world axes.

    'right' is u x n with u = world up, which is the direction that appears to
    the RIGHT in an image taken from in front of the tag. Corner order then
    matches cv2.aruco's (top-left, top-right, bottom-right, bottom-left)."""
    n = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
         "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}[facing]
    u = np.array([0.0, 1.0, 0.0])
    n = np.array(n)
    r = np.cross(u, n)
    return n, r


def _infer_facing(dx, dz, half_w, half_h):
    """Which face a tag sits on, from how close its offset is to a half-extent."""
    fx = abs(dx) / half_w if half_w > 0 else 0.0
    fz = abs(dz) / half_h if half_h > 0 else 0.0
    if fz >= fx:
        return "+z" if dz >= 0 else "-z"
    return "+x" if dx >= 0 else "-x"


def build_tag_table(map_dict, default_size_mm, panel_boxes=None):
    """map.json -> {tag_id: (4x3 world corner array, size_mm, facing, panel_i)}.

    Per obstacle:
        "tags": [{"id": 0, "dx": -500, "dy": 1000, "dz": 175,
                  "size_mm": 190, "facing": "+z"}]
    dx/dy/dz are millimetres from the obstacle's GROUND-CENTRE
    (x + w/2, 0, z + h/2); dy is height above the floor. size_mm and facing are
    optional (facing is inferred from which offset sits at its half-extent).
    """
    # A top-level "tag_size_mm" in map.json wins over the caller's default, so
    # the physical tag size lives next to the tag definitions it describes.
    # Per-tag "size_mm" still overrides both (one sheet printed differently).
    default_size_mm = float(map_dict.get("tag_size_mm", default_size_mm))
    boxes = panel_boxes
    if boxes is not None and not isinstance(boxes, (list, tuple)):
        # Guard against the easy slip of passing the CONFIG dict here: it would
        # otherwise fail deep inside with a bare KeyError, or worse, silently
        # index something wrong. panel_boxes must be the list from
        # rectilinear_mm.obstacle_boxes(map, cfg, "panel").
        raise TypeError("build_tag_table(panel_boxes=...) expects the list from "
                        "obstacle_boxes(map, cfg, 'panel'), got %s"
                        % type(boxes).__name__)
    table, dupes = {}, []
    for panel_i, ob in enumerate(map_dict.get("obstacles", [])):
        # Tags sit on the PANEL (the thin wall), not on the collision box that
        # includes the stabiliser feet — so the geometry used here must be the
        # bare wall. Offsets stay measured from the panel's GROUND-CENTRE, which
        # is unchanged by the schema: under the midline schema the obstacle
        # anchor already lies on the z midline, so cz is simply that anchor.
        if boxes is not None and panel_i < len(boxes):
            box = boxes[panel_i]
        elif "w" in ob and "h" in ob:
            box = ob                      # legacy map: geometry is on the obstacle
        else:
            # Midline-schema map with no panel_boxes supplied: the obstacle dict
            # alone does not carry its size, so tag positions cannot be built.
            # Fail with the fix rather than a bare KeyError deep in the maths.
            raise ValueError(
                "obstacle %d has no w/h (midline schema), so build_tag_table needs "
                "panel_boxes=obstacle_boxes(map, cfg, 'panel')" % panel_i)
        cx = float(box["x"]) + float(box["w"]) / 2.0
        cz = float(box["z"]) + float(box["h"]) / 2.0
        hw, hh = float(box["w"]) / 2.0, float(box["h"]) / 2.0
        for t in ob.get("tags", []) or []:
            tid = int(t["id"])
            dx, dy, dz = float(t.get("dx", 0.0)), float(t.get("dy", 0.0)), float(t.get("dz", 0.0))
            size = float(t.get("size_mm", default_size_mm))
            facing = t.get("facing") or _infer_facing(dx, dz, hw, hh)
            n, r = _face_axes(facing)
            c = np.array([cx + dx, dy, cz + dz], float)
            u = np.array([0.0, 1.0, 0.0])
            h = size / 2.0
            corners = np.array([c - r * h + u * h,     # top-left
                                c + r * h + u * h,     # top-right
                                c + r * h - u * h,     # bottom-right
                                c - r * h - u * h])    # bottom-left
            if tid in table:
                dupes.append(tid)
            table[tid] = (corners, size, facing, panel_i)
    if dupes:
        raise ValueError("duplicate tag id(s) in map.json: %s" % sorted(set(dupes)))
    return table


def yaw_from_R(R):
    """Map heading (rad) from a world->camera rotation, per the frame above."""
    return math.atan2(R[2, 0], -R[2, 2])


def _detector_params(corner_refine="SUBPIX", win_min=7, win_max=25, win_step=8):
    """ArUco detector parameters, with SUB-PIXEL corner refinement enabled.

    OpenCV defaults to CORNER_REFINE_NONE, which locates a corner only to the
    contour vertex the quad detector found — roughly 0.7-1.0 px on this camera.
    Since PnP error scales directly with corner precision, that was the single
    largest avoidable error in the whole tag pipeline. Measured on rendered
    tags at known poses, 640x480, a 4-tag panel at 2.5 m:

        NONE (old default)  corner err 0.72-0.98 px   pose err 2.7 mm
        SUBPIX              corner err 0.34-0.47 px   pose err 0.4 mm
        CONTOUR             corner err 0.84-0.98 px   pose err 2.7 mm
        APRILTAG            corner err 0.72-0.92 px   pose err 3.2 mm

    SUBPIX roughly halves corner error for a few ms per frame; CONTOUR and
    APRILTAG measured no better than NONE here, so they are not worth the cost.

    Sharper corners also LOWER the reprojection residual, so fixes that were
    being rejected for what was really detector noise now pass the RMS gate —
    the remaining residual becomes a cleaner signal of genuine map error.
    """
    p = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
         else cv2.aruco.DetectorParameters_create())
    mode = getattr(cv2.aruco, "CORNER_REFINE_%s" % str(corner_refine).upper(),
                   cv2.aruco.CORNER_REFINE_SUBPIX)
    p.cornerRefinementMethod = mode
    # Refinement search window, in pixels either side of the initial corner.
    # 5 (the default) is a good match for tags spanning ~20-90 px here: large
    # enough to find the true edge, small enough not to wander onto a
    # neighbouring feature on a small, distant tag.
    p.cornerRefinementWinSize = 5
    # Adaptive-threshold window sweep. The OpenCV default starts at 3 px, which
    # on a perforated pegboard panel is the hole pitch — the binarisation locks
    # onto hole texture rather than the marker and most tags are lost. See
    # config.TAGS_THRESH_WIN_MIN for the measurements behind these values.
    p.adaptiveThreshWinSizeMin = max(3, int(win_min))
    p.adaptiveThreshWinSizeMax = max(int(win_min) + 1, int(win_max))
    p.adaptiveThreshWinSizeStep = max(1, int(win_step))
    return p


def unsharp(gray, amount=0.6, sigma=2.0):
    """Unsharp mask, applied before detection.

    Recovers markers the adaptive threshold would otherwise miss on low-contrast
    or glared surfaces, without costing corner precision (SUBPIX re-finds the
    true edge afterwards). amount <= 0 returns the frame untouched.
    """
    if amount is None or amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), float(sigma))
    return cv2.addWeighted(gray, 1.0 + float(amount), blur, -float(amount), 0)


def _quad_side_px(quad):
    """Mean side length (px) of a 4x2 corner quad — its apparent size."""
    q = np.asarray(quad, float).reshape(4, 2)
    s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return float(np.mean(s))


def _focal_px(K):
    K = np.asarray(K, float)
    return 0.5 * (float(K[0, 0]) + float(K[1, 1]))


def _centre_spread(centres):
    """Largest horizontal (x, z) separation among a list of 3-vectors, mm."""
    best = 0.0
    for i, a in enumerate(centres):
        for b in centres[i + 1:]:
            d = math.hypot(a[0] - b[0], a[2] - b[2])
            if d > best:
                best = d
    return best


class TagLocalizer:
    """Solves the camera's map pose from any tags currently in view."""

    # Machine-readable companions to last_reason, so callers can branch on WHY
    # a solve failed without matching on prose. "degenerate" in particular is
    # the one the navigator acts on: it is the only failure a micro-parallax
    # jog can fix, and it must not be confused with "nothing in view".
    REASON_OK = None
    REASON_NOT_READY = "not_ready"
    REASON_NO_MARKERS = "no_markers"
    REASON_UNKNOWN_IDS = "unknown_ids"
    REASON_FEW_TAGS = "few_tags"
    REASON_DEGENERATE = "degenerate"
    REASON_PNP_FAILED = "pnp_failed"

    def __init__(self, tag_table, dict_name="DICT_4X4_50", min_tags=1,
                 min_spread_mm=400.0, corner_refine="SUBPIX",
                 win_min=7, win_max=25, win_step=8,
                 unsharp_amount=0.6, unsharp_sigma=2.0,
                 min_spread_ratio=0.18, min_spread_floor_mm=150.0,
                 spread_ratio_strict=False):
        self.tags = tag_table or {}
        self.min_tags = int(min_tags)
        # Absolute PASS threshold: this much horizontal spread is accepted at
        # any range, so nothing that was accepted before this change is
        # rejected now. The ratio below only ever ADDS acceptances.
        self.min_spread_mm = float(min_spread_mm)
        # Angular baseline: spread / range. This is the real conditioning
        # number (see _geometry_ok). 0 disables the ratio path entirely and
        # restores the old absolute-floor-only behaviour exactly.
        self.min_spread_ratio = float(min_spread_ratio)
        # Hard floor the ratio can never argue past. Below this the two tags
        # are physically almost one tag, and a range UNDER-estimate (a tag
        # partly occluded, so it measures small... and therefore far) must not
        # be able to talk a near-zero spread through the gate.
        self.min_spread_floor_mm = float(min_spread_floor_mm)
        # When True the ratio is a REQUIREMENT rather than an alternative
        # route: a wide-but-distant observation is rejected too. The measured
        # sweep in _geometry_ok shows the current absolute floor passing a
        # 400 mm pair at 3.5 m, which carries ~208 mm of error — worse than
        # anything the ratio gate lets through. Default False so this change
        # only ever ADDS acceptances; turn it on once the ratio reported in
        # each fix has been watched for a session.
        self.spread_ratio_strict = bool(spread_ratio_strict)
        self.unsharp_amount = float(unsharp_amount)
        self.unsharp_sigma = float(unsharp_sigma)
        # Why the last solve() returned None. Four quite different faults used
        # to collapse into one "no tags" message, sending the operator after the
        # wrong cause — a frame showing two tags in a single column was reported
        # identically to a frame showing nothing at all.
        self.last_reason = None
        self.last_reason_code = None
        # Geometry of the last gate decision, kept whether it passed or failed:
        # {spread_mm, range_mm, ratio, need_ratio}. The operator can watch a
        # marginal view get better as the rover moves, instead of only being
        # told "no".
        self.last_geometry = None
        self.enabled = bool(_ARUCO and self.tags)
        self._detector = None
        self._last_obj = {}
        self._last_img = {}
        self.last_observation = None
        if not self.enabled:
            return
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = _detector_params(corner_refine, win_min, win_max, win_step)
        if hasattr(cv2.aruco, "ArucoDetector"):          # OpenCV >= 4.7
            self._detector = cv2.aruco.ArucoDetector(d, params)
        else:                                            # pragma: no cover - legacy
            self._dict, self._params = d, params

    def _detect(self, gray):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:                                            # pragma: no cover - legacy
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict,
                                                      parameters=self._params)
        return corners, ids

    def _spread(self, ids):
        """Largest horizontal separation between any two tag centres (mm)."""
        return _centre_spread([self.tags[i][0].mean(axis=0) for i in ids])

    # ------------------------------------------------------------------ 2a
    def range_estimate_mm(self, ids, img_by_tag, K):
        """Rough camera-to-tag range (mm) from APPARENT tag size in pixels.

        range ~= f * size_mm / side_px, taken per tag and median-combined. This
        deliberately does NOT use the pose solve: the whole point of the gate is
        to decide whether that solve can be trusted, so it cannot be an input.
        Depth from apparent size is the one quantity a degenerate single-column
        view still gets right — the ambiguity there is yaw against lateral
        translation, not scale.

        A tag seen obliquely projects SMALLER than a face-on one, so this
        over-estimates range, which under-estimates the angular baseline and
        makes the gate stricter. Erring toward rejection is the safe direction.
        Returns None when no tag gives a usable measurement.
        """
        f = _focal_px(K)
        if f <= 0:
            return None
        est = []
        for t in ids:
            ent = self.tags.get(int(t))
            quad = img_by_tag.get(int(t))
            if ent is None or quad is None:
                continue
            side = _quad_side_px(quad)
            if side > 1e-6:
                est.append(f * float(ent[1]) / side)
        if not est:
            return None
        return float(np.median(est))

    def _geometry_ok(self, spread_mm, range_mm):
        """Is this observation well enough conditioned to solve?

        The old gate was an ABSOLUTE spread floor, which rejects a genuinely
        good close-range observation identically to a genuinely bad far-range
        one. What actually governs the error is the ANGULAR baseline,
        spread / range: how much real perspective difference exists between the
        tags. As that ratio tends to zero the view tends to orthographic, where
        a sideways translation and a yaw rotation produce almost the same image
        motion, so the solver cannot separate them and answers with a confident
        mirrored pose that reprojects at ~0.3 px — invisible to the RMS gate and
        repeatable enough to survive a confirmation run.

        Measured, not merely argued. Two tags at one height, 0.4 px of corner
        noise (the SUBPIX level), 200 trials each, median lateral pose error:

            spread  range   ratio    error
             150 mm  0.8 m  0.188    4.5 mm
             250 mm  1.0 m  0.250    5.6 mm      <- rejected today, for nothing
             250 mm  2.0 m  0.125   48.5 mm
             250 mm  3.5 m  0.071  287.4 mm      <- correctly bad
             400 mm  2.0 m  0.200   28.1 mm
             400 mm  3.5 m  0.114  208.0 mm      <- ACCEPTED today
             650 mm  2.0 m  0.325   20.3 mm      <- accepted today
             650 mm  3.5 m  0.186  105.8 mm      <- accepted today
             900 mm  2.5 m  0.360   24.3 mm

        Read down the ratio column: the error is very nearly a function of the
        ratio alone, and barely of the spread or the range on their own. That
        is the whole justification for the change.

        The default threshold of 0.18 is calibrated against what the CURRENT
        gate already tolerates rather than picked for taste: a 650 mm pair at
        3.5 m is ratio 0.186 and ~106 mm of error, and it passes today. So
        anything at ratio >= 0.18 is no worse conditioned than an observation
        the system already trusts.

        Two modes:
          * permissive (default) — accept if EITHER the absolute spread clears
            min_spread_mm (so nothing previously accepted is now rejected) OR
            the ratio clears min_spread_ratio with the spread still above the
            hard floor. Strictly widens the accept set.
          * strict (spread_ratio_strict=True) — the ratio must clear, full
            stop. Note the table above: the absolute floor is the MORE
            permissive rule at long range, so strict mode also closes the
            400 mm-at-3.5 m hole.

        Returns (ok, ratio_or_None, why).
        """
        ratio = (spread_mm / range_mm) if (range_mm and range_mm > 0) else None
        if self.spread_ratio_strict and self.min_spread_ratio > 0.0:
            if ratio is None:
                return False, None, ("range could not be estimated, and strict "
                                     "angular-baseline gating is on")
            if spread_mm < self.min_spread_floor_mm:
                return False, ratio, (
                    "spans only %.0f mm, under the %.0f mm hard floor"
                    % (spread_mm, self.min_spread_floor_mm))
            if ratio < self.min_spread_ratio:
                return False, ratio, (
                    "angular baseline %.3f (%.0f mm of spread at %.0f mm range) "
                    "is under %.3f — too little perspective to tell yaw from "
                    "sideways motion" % (ratio, spread_mm, range_mm,
                                         self.min_spread_ratio))
            return True, ratio, ("angular baseline %.3f clears %.3f"
                                 % (ratio, self.min_spread_ratio))
        if spread_mm >= self.min_spread_mm:
            return True, ratio, "absolute spread %.0f mm" % spread_mm
        if self.min_spread_ratio <= 0.0:
            return False, ratio, ("spans only %.0f mm (need %.0f)"
                                  % (spread_mm, self.min_spread_mm))
        if ratio is None:
            return False, None, ("spans only %.0f mm (need %.0f) and the range "
                                 "could not be estimated to judge the angular "
                                 "baseline" % (spread_mm, self.min_spread_mm))
        if spread_mm < self.min_spread_floor_mm:
            return False, ratio, (
                "spans only %.0f mm, under the %.0f mm hard floor — at that "
                "separation the tags are effectively one tag whatever the range"
                % (spread_mm, self.min_spread_floor_mm))
        if ratio >= self.min_spread_ratio:
            return True, ratio, ("angular baseline %.3f (%.0f mm at %.0f mm) "
                                 "clears %.3f" % (ratio, spread_mm, range_mm,
                                                  self.min_spread_ratio))
        return False, ratio, (
            "angular baseline %.3f (%.0f mm of spread at %.0f mm range) is under "
            "%.3f — too little perspective to tell yaw from sideways motion"
            % (ratio, spread_mm, range_mm, self.min_spread_ratio))

    def geometry_ok(self, ids, img_by_tag, K):
        """Public gate over a set of observed tag ids. Returns (ok, info)."""
        spread = self._spread(ids) if len(ids) > 1 else 0.0
        rng = self.range_estimate_mm(ids, img_by_tag, K)
        if len(ids) <= 1:
            # A single tag has no spread to judge; min_tags decides that case.
            info = {"spread_mm": 0.0, "range_mm": rng, "ratio": None,
                    "need_ratio": self.min_spread_ratio, "why": "single tag"}
            return True, info
        ok, ratio, why = self._geometry_ok(spread, rng)
        info = {"spread_mm": round(spread, 1),
                "range_mm": (round(rng) if rng else None),
                "ratio": (round(ratio, 4) if ratio else None),
                "need_ratio": self.min_spread_ratio, "why": why}
        return ok, info

    # ------------------------------------------------------------ observation
    def observe(self, frame):
        """Detect known tags and return the RAW correspondences, ungated.

        {"ids": [...], "img": {id: 4x2}, "obj": {id: 4x3}, "n": int}

        This is deliberately separate from solve(): a view too degenerate to
        solve on its own is still a perfectly good observation to pair with a
        second one taken from a different place (see solve_multiview). Throwing
        the pixels away at detection time is what made that impossible.
        """
        self.last_reason = None
        self.last_reason_code = self.REASON_OK
        if not self.enabled or frame is None:
            self.last_reason = "localiser not ready"
            self.last_reason_code = self.REASON_NOT_READY
            return None
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = unsharp(gray, self.unsharp_amount, self.unsharp_sigma)
        corners, ids = self._detect(gray)
        if ids is None or len(ids) == 0:
            self.last_reason = "no markers found in frame"
            self.last_reason_code = self.REASON_NO_MARKERS
            return None
        img, obj, seen = {}, {}, []
        for c, i in zip(corners, ids.flatten()):
            ent = self.tags.get(int(i))
            if ent is None:
                continue                                  # unknown ID -> ignore
            obj[int(i)] = np.asarray(ent[0], float)
            img[int(i)] = np.asarray(c, float).reshape(4, 2)
            seen.append(int(i))
        if not seen:
            # Markers WERE decoded, they just are not ones the map knows about.
            # Completely different fix from "nothing detected": check the ids in
            # map.json, or whether a stray marker is in shot.
            self.last_reason = ("saw marker id(s) %s — none are in map.json"
                                % sorted(int(i) for i in ids.flatten()))
            self.last_reason_code = self.REASON_UNKNOWN_IDS
            return None
        out = {"ids": sorted(seen), "img": img, "obj": obj, "n": len(seen)}
        self.last_observation = out
        return out

    # ----------------------------------------------------------------- solve
    def _pnp(self, obj, img, K, dist):
        d = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
        Km = np.asarray(K, float)
        ok, rvec, tvec = cv2.solvePnP(obj, img, Km, d, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).ravel()
        proj, _ = cv2.projectPoints(obj, rvec, tvec, Km, d)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return R, C, rms, rvec, tvec

    def solve(self, frame, K, dist=None):
        """Return {x, z, yaw_deg, n_tags, ids, rms_px, ...} for the CAMERA in
        map mm, or None when no usable fix is available.

        All visible known tags are pooled into a single PnP problem, so the
        solve is identical whether one tag is in view or eight — it simply
        becomes better conditioned as more appear.
        """
        if K is None:
            self.last_reason = "localiser not ready"
            self.last_reason_code = self.REASON_NOT_READY
            return None
        obs = self.observe(frame)
        if obs is None:
            return None
        return self.solve_obs(obs, K, dist)

    def solve_obs(self, obs, K, dist=None):
        """The gate + PnP half of solve(), over an observation already taken.

        Split out so a caller can hold on to the raw correspondences (which stay
        useful even when this refuses them — see solve_multiview) instead of
        having to re-detect the frame to get them back."""
        if obs is None:
            return None
        if K is None:
            self.last_reason = "localiser not ready"
            self.last_reason_code = self.REASON_NOT_READY
            return None
        seen = obs["ids"]
        if len(seen) < self.min_tags:
            self.last_reason = ("only %d known tag(s) %s, need %d"
                                % (len(seen), seen, self.min_tags))
            self.last_reason_code = self.REASON_FEW_TAGS
            return None
        # Conditioning gate. Tags stacked in a single vertical column (e.g. only
        # the left-hand pair of a panel in frame) give eight corners confined to
        # one tag-width horizontally, so sideways position and heading are
        # under-determined: the solver returns a confident, wrong, mirrored pose
        # that reprojects at ~0.3 px, so neither the residual nor a confirmation
        # run can catch it. Verified knife-edge: the same view solves correctly
        # in float64 and flips at the float32 precision cv2.aruco returns. Two
        # tags at different heights are NOT a substitute for two at different x.
        #
        # What counts as "enough" is now the ANGULAR baseline, not a fixed
        # millimetre floor — see _geometry_ok.
        ok, geo = self.geometry_ok(seen, obs["img"], K)
        self.last_geometry = geo
        if not ok:
            # This is NOT "no tags": the camera is looking straight at them. The
            # geometry is simply unusable, and saying so points at the fix (move
            # so both columns of a panel are in shot, get closer, or take a
            # micro-parallax pair — see solve_multiview).
            self.last_reason = "tags %s: %s" % (seen, geo["why"])
            self.last_reason_code = self.REASON_DEGENERATE
            return None
        self._last_obj = {t: obs["obj"][t] for t in seen}
        self._last_img = {t: obs["img"][t] for t in seen}
        objm = np.vstack([obs["obj"][t] for t in seen]).astype(np.float64)
        imgm = np.vstack([obs["img"][t] for t in seen]).astype(np.float64)
        res = self._pnp(objm, imgm, K, dist)
        if res is None:
            self.last_reason = "solvePnP failed on %d tag(s) %s" % (len(seen), seen)
            self.last_reason_code = self.REASON_PNP_FAILED
            return None
        R, C, rms, _rv, _tv = res
        return {"x": float(C[0]), "z": float(C[2]), "y": float(C[1]),
                "yaw_deg": math.degrees(yaw_from_R(R)),
                "n_tags": len(seen), "ids": sorted(seen), "rms_px": rms,
                "spread_mm": round(geo["spread_mm"], 1),
                "range_mm": geo["range_mm"],
                "spread_ratio": geo["ratio"],
                "views": 1}

    # ------------------------------------------------------------------- 3c
    def solve_multiview(self, views, K, dist=None, min_ratio=None,
                        min_spread_mm=None):
        """Fuse observations taken from KNOWN-OFFSET viewpoints into one solve.

        ``views`` is a list of {"obs": <observe() result>, "delta": (dx, dy, dz)}
        where delta is the camera's world displacement for that view RELATIVE TO
        VIEW 0 (so view 0's delta is (0, 0, 0)). The returned pose is the camera
        at view 0.

        Why this works, and why it needs no new solver: for a camera whose
        rotation R is the same in both views and whose centre moves by D,

            R(X - (C0 + D)) = R((X - D) - C0)

        so an observation taken from the displaced viewpoint is EXACTLY an
        observation of the world point X - D taken from view 0. Translating the
        object points back along the known baseline therefore lets both views'
        correspondences go into one ordinary pooled solvePnP.

        The pooled object cloud now spans the baseline as well as the tags, so a
        single vertical tag column — hopeless on its own — becomes a
        well-conditioned problem: the manufactured baseline supplies the
        parallax the tag layout does not. This is stereo triangulation with the
        rover's own body as the stereo rig.

        Assumptions, all of which the caller must honour:
          * the heading is the SAME at every view (the rover strafes with
            hold_yaw, so it is);
          * delta is the MEASURED displacement, not the commanded one (the T265
            is trustworthy over a sub-second, sub-metre move even though it
            drifts over minutes);
          * at least one tag is seen in common, so the views are of the same
            thing.
        Returns None (with last_reason set) if any of that fails.
        """
        self.last_reason = None
        self.last_reason_code = self.REASON_OK
        if not self.enabled or K is None:
            self.last_reason = "localiser not ready"
            self.last_reason_code = self.REASON_NOT_READY
            return None
        views = [v for v in (views or []) if v and v.get("obs")]
        if len(views) < 2:
            self.last_reason = "multiview needs 2 usable observations, got %d" % len(views)
            self.last_reason_code = self.REASON_FEW_TAGS
            return None
        common = set(views[0]["obs"]["ids"])
        for v in views[1:]:
            common &= set(v["obs"]["ids"])
        if not common:
            self.last_reason = ("the two views share no tag (%s vs %s) — they are "
                                "not looking at the same thing"
                                % (views[0]["obs"]["ids"], views[-1]["obs"]["ids"]))
            self.last_reason_code = self.REASON_UNKNOWN_IDS
            return None
        obj_parts, img_parts, centres, all_ids = [], [], [], []
        for v in views:
            obs = v["obs"]
            D = np.asarray(v.get("delta", (0.0, 0.0, 0.0)), float).reshape(3)
            for t in obs["ids"]:
                P = np.asarray(obs["obj"][t], float) - D    # X - D, see docstring
                obj_parts.append(P)
                img_parts.append(np.asarray(obs["img"][t], float).reshape(4, 2))
                centres.append(P.mean(axis=0))
                all_ids.append(int(t))
        # Conditioning of the AUGMENTED cloud. The baseline shows up here
        # naturally: the same tag seen from two places contributes two centres
        # |D| apart, so the spread the gate sees is the parallax that was
        # manufactured, exactly as if a second tag had been bolted to the wall.
        spread = _centre_spread(centres)
        rng = self.range_estimate_mm(views[0]["obs"]["ids"], views[0]["obs"]["img"], K)
        need = self.min_spread_ratio if min_ratio is None else float(min_ratio)
        floor = (self.min_spread_floor_mm if min_spread_mm is None
                 else float(min_spread_mm))
        base = float(np.linalg.norm(
            np.asarray(views[-1].get("delta", (0, 0, 0)), float)))
        ratio = (spread / rng) if (rng and rng > 0) else None
        geo = {"spread_mm": round(spread, 1),
               "range_mm": (round(rng) if rng else None),
               "ratio": (round(ratio, 4) if ratio else None),
               "need_ratio": need, "baseline_mm": round(base, 1)}
        self.last_geometry = geo
        # The pooled cloud is gated on its OWN terms, with a floor and a ratio
        # the caller sets: a manufactured baseline is better conditioned than the
        # same millimetres of tag spread would be, because the same tag is seen
        # in both views (twice the corners, so ~sqrt(2) less corner noise) and
        # the baseline runs exactly across the line of sight, which is the
        # optimal direction, rather than partly along it as panel-mounted tags
        # usually do.
        if spread < floor:
            self.last_reason = (
                "the %.0f mm jog left the pooled view spanning only %.0f mm "
                "(floor %.0f) — the rover barely moved, or the second view lost "
                "the tags" % (base, spread, floor))
            self.last_reason_code = self.REASON_DEGENERATE
            return None
        if need > 0 and ratio is not None and ratio < need:
            self.last_reason = (
                "even with the %.0f mm baseline the angular baseline is only "
                "%.3f at %.0f mm range (need %.3f) — jog further, or get closer "
                "to the tags" % (base, ratio, rng, need))
            self.last_reason_code = self.REASON_DEGENERATE
            return None
        objm = np.vstack(obj_parts).astype(np.float64)
        imgm = np.vstack(img_parts).astype(np.float64)
        res = self._pnp(objm, imgm, K, dist)
        if res is None:
            self.last_reason = "solvePnP failed on the %d-view pair" % len(views)
            self.last_reason_code = self.REASON_PNP_FAILED
            return None
        R, C, rms, _rv, _tv = res
        # Keep view 0's correspondences as the audit subject: the audit asks
        # whether the MAP is right, which is a per-view question.
        self._last_obj = {t: views[0]["obs"]["obj"][t] for t in views[0]["obs"]["ids"]}
        self._last_img = {t: views[0]["obs"]["img"][t] for t in views[0]["obs"]["ids"]}
        return {"x": float(C[0]), "z": float(C[2]), "y": float(C[1]),
                "yaw_deg": math.degrees(yaw_from_R(R)),
                "n_tags": len(set(all_ids)), "ids": sorted(set(all_ids)),
                "rms_px": rms,
                "spread_mm": geo["spread_mm"], "range_mm": geo["range_mm"],
                "spread_ratio": geo["ratio"],
                "views": len(views), "baseline_mm": round(base, 1)}

    # ----------------------------------------------------------------- audit
    def audit(self, obj_by_tag, img_by_tag, K, dist=None):
        """Diagnose WHY a solve disagrees with the map. Returns
        {scale, scale_rms, base_rms, per_tag: {id: residual_px}}.

        Two failure modes look identical in the RMS number but need opposite
        fixes, so they are separated here:

        * a wrong tag SIZE is a uniform scale error — every tag's corners are
          too big or too small by the same ratio. Fitting a single scale factor
          collapses the residual, and the fitted ratio IS the correction. A
          ratio near 1.5 is the classic ArUco convention slip: DICT_4X4_50 is
          4 data cells plus a one-cell black border each side = 6x6, and the
          "marker size" cv2.aruco expects is that FULL black square, so
          measuring only the 4x4 data area under-reports by exactly 6/4.
        * a MISPLACED or MIS-IDENTIFIED tag is a local error — one tag carries
          nearly all the residual while the rest agree, and rescaling does not
          help. The per-tag residuals name the offender directly.
        """
        ids = sorted(obj_by_tag)
        d = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
        Km = np.asarray(K, float)

        def fit(scale):
            obj, img = [], []
            for t in ids:
                P = obj_by_tag[t]
                c = P.mean(axis=0)
                obj.append(c + (P - c) * scale)     # resize about the tag centre
                img.append(img_by_tag[t])
            obj = np.vstack(obj).astype(np.float64)
            img = np.vstack(img).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(obj, img, Km, d, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None, None, None
            proj, _ = cv2.projectPoints(obj, rv, tv, Km, d)
            err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            return float(np.sqrt(np.mean(err ** 2))), err, (rv, tv)

        base_rms, base_err, _ = fit(1.0)
        if base_rms is None:
            return None
        # Coarse-to-fine scale search. A flat 1%-step sweep over 0.40..2.20 was
        # 181 solvePnP calls -- 212 ms -- and because every real frame here sits
        # above the audit trigger, it ran essentially always, costing more than
        # a whole core on the Jetson and starving the detector it was meant to
        # diagnose. RMS(scale) is smooth and single-minimum, so a 10% sweep
        # followed by a 1% refinement around the winner finds the same answer in
        # ~40 calls.
        best_s, best_rms = 1.0, base_rms
        for k in range(40, 225, 10):                # 0.40x .. 2.20x, 10% steps
            sc = k / 100.0
            r, _e, _p = fit(sc)
            if r is not None and r < best_rms:
                best_s, best_rms = sc, r
        lo = max(40, int(best_s * 100) - 10)
        for k in range(lo, lo + 21):                # +/-10%, 1% steps
            sc = k / 100.0
            r, _e, _p = fit(sc)
            if r is not None and r < best_rms:
                best_s, best_rms = sc, r
        per = {}
        if base_err is not None:
            for i, t in enumerate(ids):
                per[t] = round(float(np.sqrt(np.mean(base_err[i * 4:(i + 1) * 4] ** 2))), 2)
        out = {"scale": round(best_s, 3), "scale_rms": round(best_rms, 2),
               "base_rms": round(base_rms, 2), "per_tag": per}

        # Per-PANEL cross-check. "Largest residual" only ever names the tag in
        # the minority group, so when a whole panel has been moved the blame
        # lands on the innocent majority's odd tag out and shifts as the view
        # changes. Solving each panel INDEPENDENTLY and comparing where each
        # says the camera is measures the panel displacement directly — and it
        # catches the dangerous case the RMS gate cannot: a wholly-displaced
        # panel is self-consistent, so viewed alone it produces a low residual
        # and a confidently wrong fix.
        by_panel = {}
        for t in ids:
            ent = self.tags.get(t)
            pi = ent[3] if ent is not None and len(ent) > 3 else None
            by_panel.setdefault(pi, []).append(t)
        cams = {}
        for pi, tids in by_panel.items():
            if len(tids) < 2:
                continue                     # one tag alone cannot fix a camera
            # Same conditioning guard the main solve uses, including the angular
            # baseline: a panel contributing only ONE VERTICAL COLUMN (e.g. its
            # two -x tags) leaves sideways position unconstrained, and its solo
            # solve lands anywhere — that is what produced "panels disagree by
            # 1586 mm" from a pair that was merely stacked. Skip such panels
            # rather than report nonsense. A close-range narrow pair now passes
            # here too, so more panels can be cross-checked than before.
            pok, _pgeo = self.geometry_ok(tids, img_by_tag, K)
            if not pok:
                continue
            o = np.vstack([obj_by_tag[t] for t in tids]).astype(np.float64)
            im = np.vstack([img_by_tag[t] for t in tids]).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(o, im, Km, d, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            Rm, _ = cv2.Rodrigues(rv)
            C = (-Rm.T @ tv).ravel()
            cams[pi] = (float(C[0]), float(C[2]))
        out["per_panel_cam"] = {str(k): [round(v[0]), round(v[1])] for k, v in cams.items()}
        out["n_panels"] = len(by_panel)
        # Residual WITHIN each panel, i.e. how well that panel's own tags agree
        # with each other. Non-zero here means the tag offsets on that panel are
        # wrong; it is independent of whether panels agree with one another.
        intra = {}
        for pi, tids in by_panel.items():
            if len(tids) < 2:
                continue
            e = [per[t] for t in tids if t in per]
            if e:
                intra[str(pi)] = round(float(np.sqrt(np.mean(np.square(e)))), 2)
        out["intra_panel_rms"] = intra
        if len(cams) > 1:
            keys = sorted(cams)
            worst, pair = 0.0, None
            for i, a_ in enumerate(keys):
                for b_ in keys[i + 1:]:
                    dd = math.hypot(cams[a_][0] - cams[b_][0], cams[a_][1] - cams[b_][1])
                    if dd > worst:
                        worst, pair = dd, (a_, b_)
            out["panel_disagree_mm"] = round(worst)
            out["panel_pair"] = list(pair) if pair else None
        return out
