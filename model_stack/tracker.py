"""Lightweight IOU + centroid tracker with constant-velocity prediction.

Purpose: in the async detect_view pipeline the detector runs at ~8-13 fps but the
render loop runs at ~30 fps. This tracker carries detection boxes between detector
updates by predicting each track's position with its estimated velocity, so the
view stays smooth without re-running the models every frame.

Deliberately tiny and dependency-free (numpy only) — for a handful of person
boxes, greedy IOU association is plenty; no Kalman/Hungarian needed.
"""
import numpy as np

_MAX_EXTRAP = 0.12   # cap velocity extrapolation to 120 ms (avoids overshoot jitter)


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    def __init__(self, tid, box, t, score=0.0, contrib=None):
        self.id = tid
        self.box = np.asarray(box, np.float32)   # last observed [x1,y1,x2,y2]
        self.disp = self.box.copy()               # smoothed box shown to the UI
        self.vel = np.zeros(4, np.float32)        # box-corner velocity (px/s)
        self.t = t                                # time of last observation
        self.hits = 1
        self.misses = 0
        self.score = score
        self.contrib = contrib                    # last per-box [rgb,therm,radar], may be None

    def predict(self, t):
        """Smoothed box, lightly extrapolated (capped) along its velocity."""
        dt = max(0.0, min(t - self.t, _MAX_EXTRAP))
        return self.disp + self.vel * dt


class IOUTracker:
    def __init__(self, iou_thresh=0.3, max_misses=12, max_age=0.8,
                 vel_smooth=0.75, pos_smooth=0.35, min_hits=3):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses      # consecutive detector misses before drop
        self.max_age = max_age            # s since last obs before a track goes stale
        self.vel_smooth = vel_smooth      # higher = steadier velocity estimate
        self.pos_smooth = pos_smooth      # EMA factor for the displayed box (lower = smoother)
        self.min_hits = min_hits          # detections required before a box is drawn
        self.tracks = []
        self._next = 1

    def update(self, boxes, scores, t, contribs=None):
        """Associate a fresh detector frame (boxes Nx4, scores N) at time t.

        contribs (optional Nx3) attaches per-box modality contributions to the
        matched tracks so they survive between gate runs."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if scores is None or len(scores) != len(boxes):
            scores = np.ones(len(boxes), np.float32)
        else:
            scores = np.asarray(scores, np.float32).reshape(-1)

        def _contrib(i):
            return contribs[i] if (contribs is not None and i < len(contribs)) else None

        assigned = set()
        for tr in self.tracks:
            pred = tr.predict(t)
            best, best_iou = -1, self.iou_thresh
            for i, b in enumerate(boxes):
                if i in assigned:
                    continue
                v = _iou(pred, b)
                if v >= best_iou:
                    best_iou, best = v, i
            if best >= 0:
                b = boxes[best]; assigned.add(best)
                dt = max(1e-3, t - tr.t)
                new_vel = (b - tr.box) / dt
                tr.vel = self.vel_smooth * tr.vel + (1 - self.vel_smooth) * new_vel
                tr.box, tr.t = b, t
                tr.disp = self.pos_smooth * b + (1 - self.pos_smooth) * tr.disp
                tr.hits += 1; tr.misses = 0
                tr.score = float(scores[best])
                c = _contrib(best)
                if c is not None:
                    tr.contrib = c
            else:
                tr.misses += 1

        for i, b in enumerate(boxes):
            if i in assigned:
                continue
            self.tracks.append(Track(self._next, b, t, float(scores[i]), _contrib(i)))
            self._next += 1

        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= self.max_misses and (t - tr.t) <= self.max_age]
        return self.tracks

    def predict_boxes(self, t, w=None, h=None):
        """Predicted boxes, scores, contribs for live tracks at time t (clamped)."""
        out, sc, co = [], [], []
        for tr in self.tracks:
            if (t - tr.t) > self.max_age or tr.hits < self.min_hits:
                continue
            b = tr.predict(t)
            if w is not None and h is not None:
                b = np.array([np.clip(b[0], 0, w - 1), np.clip(b[1], 0, h - 1),
                              np.clip(b[2], 0, w - 1), np.clip(b[3], 0, h - 1)], np.float32)
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            out.append(b); sc.append(tr.score)
            co.append(tr.contrib if tr.contrib is not None else [np.nan, np.nan, np.nan])
        return (np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32),
                np.asarray(sc, np.float32) if sc else np.zeros(0, np.float32),
                np.asarray(co, np.float32) if co else np.zeros((0, 3), np.float32))
