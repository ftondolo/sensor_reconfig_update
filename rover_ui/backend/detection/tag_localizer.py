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
    """map.json -> {tag_id: (4x3 world corner array, size_mm, facing)}.

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


class TagLocalizer:
    """Solves the camera's map pose from any tags currently in view."""

    def __init__(self, tag_table, dict_name="DICT_4X4_50", min_tags=1,
                 min_spread_mm=400.0, corner_refine="SUBPIX",
                 win_min=7, win_max=25, win_step=8,
                 unsharp_amount=0.6, unsharp_sigma=2.0):
        self.tags = tag_table or {}
        self.min_tags = int(min_tags)
        self.min_spread_mm = float(min_spread_mm)
        self.unsharp_amount = float(unsharp_amount)
        self.unsharp_sigma = float(unsharp_sigma)
        # Why the last solve() returned None. Four quite different faults used
        # to collapse into one "no tags" message, sending the operator after the
        # wrong cause — a frame showing two tags in a single column was reported
        # identically to a frame showing nothing at all.
        self.last_reason = None
        self.enabled = bool(_ARUCO and self.tags)
        self._detector = None
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
        cs = [self.tags[i][0].mean(axis=0) for i in ids]
        best = 0.0
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                d = math.hypot(a[0] - b[0], a[2] - b[2])
                if d > best:
                    best = d
        return best

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
            # Same degeneracy guard the main solve uses. A panel contributing
            # only ONE VERTICAL COLUMN (e.g. its two -x tags) leaves sideways
            # position unconstrained, and its solo solve lands anywhere: that is
            # what produced "panels disagree by 1586 mm" from a pair that was
            # merely stacked. Skip such panels rather than report nonsense.
            if self._spread(tids) < self.min_spread_mm:
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

    def solve(self, frame, K, dist=None):
        """Return {x, z, yaw_deg, n_tags, ids, rms_px} for the CAMERA in map mm,
        or None when no usable fix is available.

        All visible known tags are pooled into a single PnP problem, so the
        solve is identical whether one tag is in view or eight — it simply
        becomes better conditioned as more appear.
        """
        self.last_reason = None
        if not self.enabled or frame is None or K is None:
            self.last_reason = "localiser not ready"
            return None
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = unsharp(gray, self.unsharp_amount, self.unsharp_sigma)
        corners, ids = self._detect(gray)
        if ids is None or len(ids) == 0:
            self.last_reason = "no markers found in frame"
            return None
        obj, img, seen = [], [], []
        for c, i in zip(corners, ids.flatten()):
            ent = self.tags.get(int(i))
            if ent is None:
                continue                                  # unknown ID -> ignore
            obj.append(ent[0])
            img.append(c.reshape(4, 2))
            seen.append(int(i))
        if not seen:
            # Markers WERE decoded, they just are not ones the map knows about.
            # Completely different fix from "nothing detected": check the ids in
            # map.json, or whether a stray marker is in shot.
            self.last_reason = ("saw marker id(s) %s — none are in map.json"
                                % sorted(int(i) for i in ids.flatten()))
            return None
        if len(seen) < self.min_tags:
            self.last_reason = ("only %d known tag(s) %s, need %d"
                                % (len(seen), seen, self.min_tags))
            return None
        # Reject observations with no horizontal SPREAD. Tags stacked in a
        # single vertical column (e.g. only the left-hand pair of a panel in
        # frame) give eight corners confined to one tag-width horizontally, so
        # the sideways position and heading are under-determined: the solver
        # returns a confident, wrong, mirrored pose that reprojects at ~0.3 px,
        # so neither the residual nor a confirmation run can catch it. Verified
        # knife-edge: the same view solves correctly in float64 and flips at the
        # float32 precision cv2.aruco actually returns. Two tags at different
        # heights are NOT a substitute for two at different x.
        if len(seen) > 1 and self._spread(seen) < self.min_spread_mm:
            # This is NOT "no tags": the camera is looking straight at them. The
            # geometry is simply unusable, and saying so points at the fix
            # (move so both columns of a panel are in shot).
            self.last_reason = (
                "tags %s span only %.0f mm horizontally (need %.0f) — they are "
                "one vertical column, so sideways position is undetermined"
                % (seen, self._spread(seen), self.min_spread_mm))
            return None
        obj_list, img_list = obj, img
        obj = np.vstack(obj).astype(np.float64)
        img = np.vstack(img).astype(np.float64)
        d = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
        self._last_obj = {t: o for t, o in zip(seen, obj_list)}
        self._last_img = {t: i for t, i in zip(seen, img_list)}
        ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), d,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            self.last_reason = "solvePnP failed on %d tag(s) %s" % (len(seen), seen)
            return None
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).ravel()                         # camera centre in world
        proj, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(K, float), d)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return {"x": float(C[0]), "z": float(C[2]), "y": float(C[1]),
                "yaw_deg": math.degrees(yaw_from_R(R)),
                "n_tags": len(seen), "ids": sorted(seen), "rms_px": rms,
                "spread_mm": round(self._spread(seen), 1)}