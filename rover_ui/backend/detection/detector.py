"""In-process multimodal detector for the audience view.

The heavy detection stack (RGB + thermal YOLOv8, the shared gated-fusion
ResNet18 that yields per-detection sensor *contributions*, and the DQN adaptive
resolution controller) lives in the sibling ``jetson_deploy`` project. Rather
than run that as a second process — which would fight the rover_ui sensor
threads for the cameras — we import it lazily and feed it the frames the FLIR
and D435i threads already produce.

Outputs, like every other sensor:
  * ``self.latest``    — an annotated RGB JPEG (boxes + contribution labels),
                         streamed to the browser as MJPEG (``/stream/detect``).
  * ``self.telemetry`` — structured detection data (boxes, per-detection
                         contributions, aggregate contribution, adaptive
                         settings, fps) pushed over the telemetry WebSocket.

If torch / ultralytics / the weights are unavailable (e.g. a dev laptop), the
thread raises out of :meth:`open` and the base class drops into :meth:`read_mock`,
which fabricates plausible boxes and contributions so the audience UI is fully
exercisable without the GPU stack. The badge reads MOCK in that case, LIVE when
the real models are running — exactly like the sensors.
"""
import math
import os
import sys
import threading
import time

import numpy as np

from ..sensors.base import SensorThread, LatestValue
from .. import config

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _install_numpy_compat() -> None:
    """Let numpy-1.x unpickle checkpoints saved under numpy 2.x.

    numpy 2.0 moved its C internals from ``numpy.core`` to ``numpy._core``.
    Checkpoints pickled there reference ``numpy._core.*``; on a numpy-1.x runtime
    those modules don't exist and unpickling raises ModuleNotFoundError. Aliasing
    the private paths to the public 1.x ones fixes it. No-op on numpy >= 2.0.
    """
    import sys
    import numpy
    if hasattr(numpy, "_core"):
        return
    for sub in ("", ".multiarray", ".numeric", ".umath", ".overrides", "._multiarray_umath"):
        src = "numpy.core" + sub
        dst = "numpy._core" + sub
        try:
            __import__(src)
            sys.modules[dst] = sys.modules[src]
        except Exception:
            pass


def _pick_weights(weights_dir, stem):
    """Prefer the TensorRT .engine (FP16, ~1.8x faster) when present + enabled."""
    eng = os.path.join(weights_dir, stem + ".engine")
    if config.DETECT_USE_TRT and os.path.exists(eng):
        return eng
    return os.path.join(weights_dir, stem + ".pt")


class DetectorThread(SensorThread):
    name = "detector"

    def __init__(self, rgb_sensor, thermal_sensor, radar_sensor=None, allow_mock: bool = True):
        super().__init__(allow_mock)
        self._rgb_sensor = rgb_sensor
        self._thermal_sensor = thermal_sensor
        self._radar_sensor = radar_sensor
        self.telemetry = LatestValue()
        # Annotated streams for the other modality panels (same boxes as RGB).
        self.latest_thermal = LatestValue()
        self.latest_depth = LatestValue()

        # mmWave presence -> bearing/distance line on the RGB view. Algorithm
        # (clustering + hysteresis) is reused from jetson_deploy/radar_presence.py.
        self._radar_eval = None       # evaluate_frame fn (set in open())
        self._radar_conf = None       # confidence fn
        self._rcfg = None             # PresenceConfig
        self._rwin = None             # deque of qualify flags
        self._rpresent = False
        self._rlast_best = None
        self._rinfo = None            # {range, az, conf} or None
        self._raz = None              # smoothed azimuth (deg)
        self._rlast_seq = -1
        self._radar_tracker = None    # single-moving-target tracker (set in open())
        self._radar_last_t = 0.0      # time of last radar frame (staleness guard)
        self._rover_moving_getter = None  # () -> bool: is the rover translating?
        self._rlock = threading.Lock()

        self._interval = 1.0 / max(config.DETECT_FPS, 1)
        self._next_t = 0.0
        self._fps = 0.0            # detection fps
        self._render_fps = 0.0     # display (render-loop) fps
        self._mock_t0 = None

        # Real-model handles (populated in open()).
        self._torch = None
        self._device = None
        self._rgb_yolo = None
        self._therm_yolo = None
        self._gate = None
        self._controller = None
        self._res_tiers = None
        self._fps_tiers = None

        # Async render pipeline (Phase 2): the heavy models run in read_once at
        # their own rate and feed a tracker; a separate render thread draws
        # tracker-predicted boxes onto the latest RGB frame at DETECT_RENDER_FPS,
        # so /stream/detect stays smooth (~30 fps) regardless of detection speed.
        self._tracker = None
        self._clahe = None            # fallback thermal enhancement (set in open())
        self._thermal_pre = None      # FLIROneProPreprocessor (set in open())
        self._therm_disp = None       # cached preprocessed thermal for the panel
        self._render_i = 0            # render-loop counter (throttles side panels)
        # Smoothed single target (prevents teleport across vision<->radar handoff).
        self._sm = None               # {cx, cy, bw, bh, range, score, contrib, radar, t}
        self._last_depth = None       # (range_m, t) last good D435 depth, for the
                                      # depth->mmwave hold (see _assemble_dets)
        self._disp_dets = []          # smoothed det(s) the render loop draws
        self._disp_target = None      # smoothed {x, az, range, source} for control
        # Debounced displayed resolution (shows the DQN's adaptive res, no flicker).
        self._scale_deb = {"rgb": {"cand": 1.0, "cand_t": 0.0, "disp": 1.0},
                           "therm": {"cand": 1.0, "cand_t": 0.0, "disp": 1.0}}
        self._frame_i = 0
        self._gate_every = max(1, int(config.DETECT_GATE_EVERY))
        self._render_interval = 1.0 / max(1.0, float(config.DETECT_RENDER_FPS))
        self._render_thread = None
        self._render_stop = threading.Event()
        self._slock = threading.Lock()      # guards _last_settings
        self._last_settings = None

    def set_rover_state_getter(self, fn):
        """fn() -> bool, True while the rover is translating (ego-motion). Lets
        the radar tracker drop Doppler and track spatially while moving."""
        self._rover_moving_getter = fn

    # ---- lifecycle -----------------------------------------------------
    def open(self) -> None:
        if not config.DETECT_ENABLE:
            raise RuntimeError("detection disabled (DETECT_ENABLE=0)")
        if cv2 is None:
            raise RuntimeError("opencv-python not installed")

        jd = config.JETSON_DEPLOY_DIR
        weights = os.path.join(jd, "weights")
        for w in ("yolov8_rgb.pt", "yolov8_thermal.pt", "shared_gated.pt", "dqn_agent.pt"):
            if not os.path.exists(os.path.join(weights, w)):
                raise RuntimeError(f"missing detector weight: {w} in {weights}")

        # Lazy, expensive imports — only when we actually have the stack.
        import torch  # noqa: F401
        import torchvision

        # The gate/DQN checkpoints were pickled under numpy 2.x (module path
        # ``numpy._core``), but the Jetson runs numpy 1.x. Alias the private
        # module paths to their 1.x equivalents so torch.load can unpickle them
        # without a risky numpy upgrade. No-op when numpy already has _core.
        _install_numpy_compat()

        # Jetson GPU has no torchvision NMS CUDA kernel; patch with a pure-PyTorch
        # implementation before ultralytics imports it. (Same patch run_live uses.)
        def _nms_pure(boxes, scores, iou_threshold):
            if len(boxes) == 0:
                return torch.zeros(0, dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                xx1 = torch.max(x1[i], x1[order[1:]]); yy1 = torch.max(y1[i], y1[order[1:]])
                xx2 = torch.min(x2[i], x2[order[1:]]); yy2 = torch.min(y2[i], y2[order[1:]])
                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                order = order[1:][iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        torchvision.ops.nms = _nms_pure
        torchvision.ops.boxes.nms = _nms_pure

        if jd not in sys.path:
            sys.path.insert(0, jd)
        from ultralytics import YOLO
        from models.shared_gated_detector import SharedGatedDetector
        from front_end.adaptive_rl.dqn_agent import QNetwork
        from front_end.adaptive_rl.environment import (
            STATE_DIM, NUM_ACTIONS, RES_TIERS, FPS_TIERS, action_to_tiers,
        )

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._res_tiers = RES_TIERS
        self._fps_tiers = FPS_TIERS

        self._rgb_yolo = YOLO(_pick_weights(weights, "yolov8_rgb"))
        self._therm_yolo = YOLO(_pick_weights(weights, "yolov8_thermal"))

        ckpt = torch.load(os.path.join(weights, "shared_gated.pt"),
                          map_location=self._device, weights_only=False)
        gate = SharedGatedDetector(
            backbone_name=ckpt["backbone"], pretrained=False,
            num_modalities=ckpt["num_modalities"],
            fpn_channels=ckpt["fpn_channels"], temperature=ckpt["temperature"],
        ).to(self._device)
        gate.load_state_dict(ckpt["model"])
        gate.eval()
        self._gate = gate

        self._controller = _AdaptiveController(
            os.path.join(weights, "dqn_agent.pt"), QNetwork, STATE_DIM, NUM_ACTIONS,
            RES_TIERS, FPS_TIERS, action_to_tiers, torch, self._device,
        )

        # Tracker carries boxes between (decimated) detection frames.
        from tracker import IOUTracker
        self._tracker = IOUTracker(min_hits=int(config.DETECT_TRACK_MIN_HITS))
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # Full FLIR One Pro domain-gap preprocessor (palette->value, MSX bilateral,
        # CLAHE, histogram-match to training mean~130/std~60).
        try:
            from thermal_preprocess import FLIROneProPreprocessor
            self._thermal_pre = FLIROneProPreprocessor(clahe_clip=2.0)
            # This FLIR relay outputs pure grayscale with no MSX (edge density
            # ~0.02), so skip the bilateral MSX filter — it is the single biggest
            # CPU cost on the detection thread and buys nothing here.
            self._thermal_pre._remove_msx = lambda g: g
        except Exception as exc:
            self._thermal_pre = None
            print(f"[detector] thermal_preprocess unavailable ({exc}); using CLAHE only")
        self._frame_i = 0
        self._last_settings = self._controller.decide()

        # mmWave target for the RGB bearing/distance line. Default = the single-
        # moving-target tracker (Doppler isolates the one human; spatial gate +
        # hold keeps the lock on stops). Set RADAR_USE_TRACKER=0 to fall back to
        # the old per-frame presence clustering.
        self._rinfo = None
        self._radar_last_t = 0.0
        self._rlast_seq = -1
        if self._radar_sensor is not None and config.RADAR_USE_TRACKER:
            from radar_tracker import RadarTracker, RadarTrackerConfig
            tc = RadarTrackerConfig()
            tc.accum_sec = float(config.RADAR_ACCUM_SEC)
            tc.v_move = float(config.RADAR_V_MOVE)
            tc.gate_radius = float(config.RADAR_GATE_RADIUS)
            tc.hold_timeout = float(config.RADAR_HOLD_TIMEOUT)
            tc.confirm_hold = float(config.RADAR_CONFIRM_HOLD)
            tc.pos_smooth = float(config.RADAR_POS_SMOOTH)
            tc.min_snr_peak = float(config.RADAR_MIN_SNR_PEAK)
            tc.min_snr_sum = float(config.RADAR_MIN_SNR_SUM)
            self._radar_tracker = RadarTracker(tc)
            self._radar_eval = None
        elif self._radar_sensor is not None:
            from collections import deque
            from radar_presence import PresenceConfig, evaluate_frame, confidence
            self._rcfg = PresenceConfig()
            self._rcfg.eps = float(config.RADAR_EPS)
            self._rcfg.min_pts = int(config.RADAR_MIN_PTS)
            self._rcfg.enter_need = int(config.RADAR_ENTER_NEED)
            self._rcfg.release_win = int(config.RADAR_RELEASE_WIN)
            self._radar_eval = evaluate_frame
            self._radar_conf = confidence
            self._rwin = deque(maxlen=self._rcfg.release_win)
            self._rpresent = False
            self._rlast_best = None
            self._raz = None

        # Launch the async render loop that drives /stream/detect at ~30 fps.
        self._render_stop.clear()
        if self._render_thread is None or not self._render_thread.is_alive():
            self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
            self._render_thread.start()

    def close(self) -> None:
        # Stop the render loop so the base class's mock path can own self.latest.
        self._render_stop.set()
        rt = self._render_thread
        if rt is not None and rt.is_alive() and rt is not threading.current_thread():
            rt.join(timeout=1.0)
        self._render_thread = None
        # Drop references so a retry rebuilds cleanly; free GPU memory if we can.
        self._rgb_yolo = self._therm_yolo = self._gate = self._controller = None
        self._tracker = None
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- pacing --------------------------------------------------------
    def _pace(self) -> None:
        now = time.monotonic()
        if now < self._next_t:
            time.sleep(max(0.0, self._next_t - now))
        self._next_t = time.monotonic() + self._interval

    # ---- real inference (DETECTION step; render loop owns self.latest) ----
    def read_once(self):
        # Runs the heavy models as fast as they go and feeds the tracker. We
        # return None so the SensorThread base does NOT publish at detection
        # rate — _render_loop publishes self.latest at ~30 fps instead.
        rgb, _ = self._rgb_sensor.raw.get()
        therm_gray, _ = self._thermal_sensor.raw.get()
        if rgb is None or therm_gray is None:
            time.sleep(0.01)
            return None

        t0 = time.time()
        torch = self._torch
        rgb = np.ascontiguousarray(rgb)
        if therm_gray.ndim == 2:
            therm_bgr = cv2.cvtColor(therm_gray, cv2.COLOR_GRAY2BGR)
        else:
            therm_bgr = therm_gray
        h, w = rgb.shape[:2]
        if therm_bgr.shape[:2] != (h, w):
            therm_bgr = cv2.resize(therm_bgr, (w, h))

        settings = self._controller.decide()  # DQN tiers still drive the panel/HUD

        # RGB YOLO every frame at FULL resolution. The DQN's downscale tier is
        # kept for the demo panel, but we don't shrink the detection input: the
        # fixed-shape TRT engine letterboxes to 640 either way (so downscaling
        # saved no real compute) and a changing input size made the box leap.
        rd = self._rgb_yolo.predict(rgb, verbose=False, conf=config.DETECT_CONF, iou=0.5, classes=[0])
        rgb_boxes = rd[0].boxes.xyxy.cpu().numpy() if len(rd[0].boxes) > 0 else np.zeros((0, 4))
        rgb_scores = rd[0].boxes.conf.cpu().numpy() if len(rd[0].boxes) > 0 else np.zeros(0)
        rgb_boxes, rgb_scores = self._filter_small(rgb_boxes, rgb_scores, w, h)

        # Thermal YOLO + gate are decimated to every Nth frame.
        contribs = None
        if self._frame_i % self._gate_every == 0:
            # Thermal: full-res 640x480 + CLAHE (skip the DQN downscale so the
            # weak FLIR feed gets its best chance); boxes already in RGB coords.
            therm_in = self._preprocess_thermal(therm_bgr) if config.DETECT_THERMAL_ENHANCE else therm_bgr
            self._therm_disp = therm_in          # cache for the panel (avoid recompute)
            td = self._therm_yolo.predict(therm_in, verbose=False, conf=config.DETECT_THERMAL_CONF, iou=0.5, classes=[0])
            therm_boxes = td[0].boxes.xyxy.cpu().numpy() if len(td[0].boxes) > 0 else np.zeros((0, 4))
            therm_scores = td[0].boxes.conf.cpu().numpy() if len(td[0].boxes) > 0 else np.zeros(0)
            therm_boxes, therm_scores = self._filter_small(therm_boxes, therm_scores, w, h)
            # Prefer the STABLE RGB box; only fall back to the weak/misaligned
            # thermal box when RGB has nothing (e.g. RGB blocked). Mixing them
            # frame-to-frame is the main source of box jitter on this hardware.
            if len(rgb_boxes) > 0:
                boxes, scores = rgb_boxes, rgb_scores
            else:
                boxes, scores = therm_boxes, therm_scores
            if len(boxes) > 0:
                sz = config.DETECT_IMG_SIZE
                rgb_t = torch.from_numpy(cv2.resize(rgb, (sz, sz))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                therm_t = torch.from_numpy(cv2.resize(therm_bgr, (sz, sz))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                # Mask out a blocked/dead modality (covered lens / black feed) so
                # the gate doesn't credit a sensor that sees nothing -> its
                # contribution is forced to exactly 0%. Radar is always present.
                rg = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                tg = cv2.cvtColor(therm_bgr, cv2.COLOR_BGR2GRAY)
                rgb_alive = 1.0 if (rg.std() > config.DETECT_ALIVE_STD and rg.mean() > config.DETECT_ALIVE_MEAN) else 0.0
                therm_alive = 1.0 if (tg.std() > config.DETECT_ALIVE_STD and tg.mean() > config.DETECT_ALIVE_MEAN) else 0.0
                mmask = torch.tensor([[rgb_alive, therm_alive, 1.0]], dtype=torch.float32, device=self._device)
                with torch.no_grad():
                    _, _, gates = self._gate(rgb_t.to(self._device), therm_t.to(self._device), modality_mask=mmask)
                contribs = self._gate.extract_contributions(gates, boxes, h, w)
        else:
            boxes, scores = rgb_boxes, rgb_scores

        # Fold in the freshest mmWave frame (radar-only fallback). Policy:
        # radar points produced while the ROVER ITSELF is translating
        # DO NOT COUNT — ego-motion Doppler-shifts static clutter, so the
        # tracker is reset and radar contributes nothing until the rover is
        # stationary again and a MOVING cluster is freshly acquired. Static
        # returns are filtered in software (|v| < RADAR_MIN_POINT_V) on top of
        # the hardware clutterRemoval enabled in the chirp cfg.
        if self._radar_sensor is not None:
            rv, rseq = self._radar_sensor.latest.get()
            if rv is not None and rseq != self._rlast_seq:
                self._rlast_seq = rseq
                moving = bool(self._rover_moving_getter()) if self._rover_moving_getter else False
                min_v = float(config.RADAR_MIN_POINT_V)
                pts = [p for p in rv.get("points", [])
                       if abs(p.get("v", 0.0)) >= min_v]
                if moving:
                    if self._radar_tracker is not None:
                        self._radar_tracker.reset()
                    self._reset_presence()
                    with self._rlock:
                        self._rinfo = None
                        self._raz = None
                elif self._radar_tracker is not None:
                    now_r = time.time()
                    self._radar_tracker.update(pts, now_r, rover_moving=False)
                    self._radar_last_t = now_r
                    tgt = self._radar_tracker.target()
                    with self._rlock:
                        self._rinfo = ({"range": tgt["range"], "az": tgt["az"],
                                        "conf": 0.9 if tgt["moving"] else 0.7,
                                        "snr_peak": tgt.get("snr_peak"), "snr_sum": tgt.get("snr_sum")}
                                       if tgt else None)
                elif self._radar_eval is not None:
                    self._radar_last_t = time.time()
                    self._ingest_radar(pts)
        rinfo = self._get_rinfo() if self._radar_sensor is not None else None
        radar_only = (len(boxes) == 0 and rinfo is not None)

        # DQN state: real contribs, or Radar-100% when only the radar sees a person.
        if contribs is not None and len(contribs):
            dqn_contribs, avg_conf = contribs, float(scores.mean())
        elif radar_only:
            dqn_contribs = np.array([[0.0, 0.0, 1.0]], np.float32)
            avg_conf = float(rinfo.get("conf", 0.5))
        else:
            dqn_contribs = np.zeros((0, 3))
            avg_conf = float(scores.mean()) if len(scores) else 0.5
        n_dets = len(boxes) if len(boxes) else (1 if radar_only else 0)
        settings = self._controller.update(dqn_contribs, avg_conf, n_dets)

        now = time.time()
        self._tracker.update(boxes, scores, now, contribs)
        with self._slock:
            self._last_settings = settings

        inst = 1.0 / max(now - t0, 1e-6)
        self._fps = inst if self._fps == 0 else 0.2 * inst + 0.8 * self._fps

        # Single radar-validated detection -> SMOOTHED so it never teleports across
        # a vision<->radar handoff. Shared with the render loop so display, the
        # per-detection list, and the control target are all the same smooth box.
        pb, ps, pc = self._tracker.predict_boxes(now, w, h)
        self._fill_contribs(pc)
        raw_dets = self._assemble_dets(pb, ps, pc, w, h, rinfo)
        sdet = self._smooth_target(raw_dets[0] if raw_dets else None, w, h, now)
        dets = [sdet] if sdet else []
        target = self._target_from_det(sdet, w)
        with self._slock:
            self._disp_dets = dets
            self._disp_target = target
        tel = self._telemetry(dets, settings, w, h, mock=False)
        tel["target"] = target
        tel["radar_snr"] = ({"peak": rinfo.get("snr_peak"), "sum": rinfo.get("snr_sum")}
                            if rinfo else None)
        # Radar track in the RADAR frame (range + raw azimuth, same convention
        # as the point cloud: x=range*sin(az), y=range*cos(az)) so the audience
        # mmWave plot can box the tracked target. target.az is camera-frame and
        # not usable for that; rinfo.az is the raw radar bearing.
        tel["radar_track"] = ({"range": round(float(rinfo["range"]), 3),
                               "az": round(float(rinfo["az"]), 1)}
                              if rinfo and rinfo.get("range") is not None
                              and rinfo.get("az") is not None else None)
        tel["dbg"] = {"vbox": int(len(boxes)),
                      "vmax": round(float(scores.max()), 2) if len(scores) else 0.0,
                      "tracks": int(len(pb)),
                      "radar": rinfo is not None}
        self.telemetry.set(tel)

        self._frame_i += 1
        return None

    # ---- async render loop (drives /stream/detect at ~30 fps) ----------
    def _fill_contribs(self, pc):
        """Replace not-yet-gated (nan) track contribs with the running EMA."""
        if len(pc) and self._controller is not None:
            ema = self._controller.contrib_ema
            for i in range(len(pc)):
                if np.isnan(pc[i]).any():
                    pc[i] = ema
        return pc

    def _filter_small(self, boxes, scores, w, h):
        """Drop boxes smaller than DETECT_MIN_BOX_FRAC of the frame (object FPs)."""
        if len(boxes) == 0:
            return boxes, scores
        min_area = config.DETECT_MIN_BOX_FRAC * w * h
        keep = [i for i in range(len(boxes))
                if (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) >= min_area]
        if len(keep) == len(boxes):
            return boxes, scores
        if not keep:
            return np.zeros((0, 4)), np.zeros(0)
        idx = np.asarray(keep, int)
        return boxes[idx], scores[idx]

    def _preprocess_thermal(self, therm_bgr):
        """Bridge the FLIR One Pro -> FLIR ADAS domain gap before the thermal
        YOLO. Uses thermal_preprocess.FLIROneProPreprocessor when available
        (palette->value, MSX bilateral, CLAHE, histogram-match), else CLAHE."""
        if self._thermal_pre is not None:
            out = self._thermal_pre.process(therm_bgr)
            if out is None:
                return therm_bgr
            if config.DETECT_THERMAL_INVERT:
                g = 255 - cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            return out
        g = cv2.cvtColor(therm_bgr, cv2.COLOR_BGR2GRAY) if therm_bgr.ndim == 3 else therm_bgr
        if self._clahe is not None:
            g = self._clahe.apply(g)
        if config.DETECT_THERMAL_INVERT:
            g = 255 - g
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def _render_loop(self):
        last = None
        while not self._render_stop.is_set() and not self._stop.is_set():
            t_start = time.time()
            rgb, _ = self._rgb_sensor.raw.get()
            if rgb is None or self._tracker is None:
                self._render_stop.wait(0.02); continue
            with self._slock:
                settings = self._last_settings
                dets = self._disp_dets          # smoothed single det (read_once owns it)
                tgt = self._disp_target
            if settings is None:
                self._render_stop.wait(0.02); continue
            rgb = np.ascontiguousarray(rgb)
            h, w = rgb.shape[:2]
            try:
                self.latest.set(self._render(rgb, dets, settings, mock=False, tgt=tgt))
            except Exception:
                self._render_stop.wait(0.02); continue
            # Side panels (thermal + depth) are secondary -> ~10 fps, and reuse
            # the detector's already-preprocessed thermal (no extra bilateral).
            self._render_i += 1
            if self._render_i % 3 == 0:
                try:
                    tb = self._therm_disp
                    if tb is None:
                        tv, _ = self._thermal_sensor.raw.get()
                        tb = cv2.cvtColor(tv, cv2.COLOR_GRAY2BGR) if (tv is not None and tv.ndim == 2) else tv
                    if tb is not None:
                        tb = tb.copy()
                        if tb.shape[:2] != (h, w):
                            tb = cv2.resize(tb, (w, h))
                        if config.DETECT_SHOW_RES:
                            tb = self._degrade(tb, self._debounced_scale(
                                "therm", settings.get("therm_scale", 1.0), time.time()))
                        self.latest_thermal.set(self._encode_panel(tb, dets, "THERMAL"))
                except Exception:
                    pass
                try:
                    dj, _ = self._rgb_sensor.depth_jpeg.get()
                    if dj is not None:
                        dimg = cv2.imdecode(np.frombuffer(dj, np.uint8), cv2.IMREAD_COLOR)
                        if dimg is not None:
                            if dimg.shape[:2] != (h, w):
                                dimg = cv2.resize(dimg, (w, h))
                            self.latest_depth.set(self._encode_panel(dimg, dets, "DEPTH"))
                except Exception:
                    pass
            now = time.time()
            if last is not None:
                inst = 1.0 / max(now - last, 1e-6)
                self._render_fps = inst if self._render_fps == 0 else 0.1 * inst + 0.9 * self._render_fps
            last = now
            sleep = self._render_interval - (time.time() - t_start)
            if sleep > 0:
                self._render_stop.wait(sleep)

    # ---- mock fallback -------------------------------------------------
    def read_mock(self):
        if cv2 is None:
            return None
        self._pace()
        if self._mock_t0 is None:
            self._mock_t0 = time.time()
        t = time.time() - self._mock_t0

        rgb, _ = self._rgb_sensor.raw.get()
        if rgb is None:
            rgb = np.full((config.D435_HEIGHT, config.D435_WIDTH, 3), 30, dtype="uint8")
            cv2.putText(rgb, "no RGB frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (120, 120, 120), 2)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]

        # One drifting subject box, tracking the mock RGB rectangle's motion.
        cx = w * (0.5 + 0.3 * math.sin(t * 0.5))
        cy = h * 0.55
        bw, bh = w * 0.16, h * 0.42
        box = np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]])
        score = np.array([0.72 + 0.12 * math.sin(t * 1.3)])

        # Contributions drift between an "RGB-dominant day" and a "thermal-
        # dominant night" so the audience bars visibly move.
        phase = 0.5 + 0.5 * math.sin(t * 0.25)   # 0 = night, 1 = day
        r = 0.25 + 0.45 * phase
        th = 0.65 - 0.45 * phase
        ra = max(0.05, 1.0 - r - th)
        s = r + th + ra
        contribs = np.array([[r / s, th / s, ra / s]])

        settings = self._mock_settings(phase)
        self._fps = config.DETECT_FPS

        dets = self._build_dets(box, score, contribs)
        jpeg = self._render(rgb, dets, settings, mock=True)
        self.telemetry.set(self._telemetry(dets, settings, w, h, mock=True))
        return jpeg

    # ---- shared helpers ------------------------------------------------
    def _build_dets(self, boxes, scores, contribs):
        dets = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i][:4]]
            c = contribs[i] if i < len(contribs) else np.array([0.0, 0.0, 0.0])
            dets.append({
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(float(scores[i]) if i < len(scores) else 0.0, 3),
                "contrib": [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)],
            })
        return dets

    @staticmethod
    def _box_color(d):
        """BGR box color = the DOMINANT sensor's color, matching the audience
        Sensor Contribution bars (RGB=blue, Thermal=green, Radar=orange).
        A radar-only detection is always orange; otherwise the box takes the
        color of whichever modality contributed most to it."""
        BLUE = (247, 129, 47)     # --rgb  #2f81f7
        GREEN = (80, 185, 63)     # --radar palette (thermal shown green) #3fb950
        ORANGE = (62, 136, 240)   # --thermal palette (radar shown orange) #f0883e
        if d.get("radar"):
            return ORANGE
        c = d.get("contrib") or [0.0, 0.0, 0.0]
        idx = max(range(3), key=lambda i: c[i]) if any(c) else 0
        return (BLUE, GREEN, ORANGE)[idx]

    def _draw_boxes(self, canvas, dets, labels=True):
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            radar = d.get("radar")
            color = self._box_color(d)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if labels:
                if radar:
                    label = "PERSON (radar)"
                else:
                    c = d["contrib"]
                    label = f"R:{c[0]:.0%} T:{c[1]:.0%} Ra:{c[2]:.0%}"
                cv2.rectangle(canvas, (x1, max(y1 - 18, 0)), (x1 + 9 * len(label), max(y1, 18)),
                              (0, 0, 0), -1)
                cv2.putText(canvas, label, (x1 + 2, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _target_bearing(self, dets, rinfo, w, h):
        """Unified target for the bearing line + rover control. Aligns to the
        vision detection when there is one (radar supplies range); otherwise
        falls back to the radar azimuth — the rover-control mode used when RGB
        and thermal both fail. Returns {x, az(deg), range, source} or None."""
        hfov = float(config.DETECT_RGB_HFOV)
        vis = [d for d in dets if not d.get("radar")]
        radar_rng = rinfo.get("range") if rinfo else None
        # Radar bearing in the CAMERA frame (+ = right), so it's directly
        # comparable to a vision-box bearing and drives a consistent strafe.
        radar_az = None
        if rinfo is not None and rinfo.get("az") is not None:
            radar_az = (-rinfo["az"] if config.DETECT_RADAR_FLIP else rinfo["az"])

        def vis_az(d):
            cx = (d["box"][0] + d["box"][2]) / 2.0
            return (cx / w - 0.5) * hfov

        def vis_target(d):
            cx = (d["box"][0] + d["box"][2]) / 2.0
            return {"x": float(cx), "az": round(vis_az(d), 1),
                    "range": radar_rng, "source": "vision"}

        if vis:
            if radar_az is not None:
                # Radar = the anchor (the one real moving human). Only accept a
                # vision box whose bearing AGREES with the radar; reject the rest
                # (wrong RGB/thermal detections can't steer the rover).
                tol = float(config.FOLLOW_FUSE_TOL)
                ok = [d for d in vis if abs(vis_az(d) - radar_az) <= tol]
                if ok:
                    return vis_target(min(ok, key=lambda d: abs(vis_az(d) - radar_az)))
                # no vision box matches the radar -> fall through to the radar target
            else:
                # No radar lock to cross-check -> trust the largest vision box.
                return vis_target(max(vis, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])))
        if radar_az is not None:
            frac = max(-1.0, min(1.0, radar_az / (hfov / 2.0)))
            return {"x": float(w * 0.5 * (1 + frac)), "az": round(radar_az, 1),
                    "range": radar_rng, "source": "radar"}
        return None

    def _draw_bearing(self, canvas, tgt):
        """Vertical bearing line at the (smoothed) target center."""
        if tgt is None:
            return
        h, w = canvas.shape[:2]
        col = int(max(0, min(w - 1, tgt["x"])))
        radar_only = tgt["source"] == "radar"
        color = (255, 180, 60) if radar_only else (120, 255, 120)  # amber vs green
        cv2.line(canvas, (col, 22), (col, h), color, 2)
        rng = tgt["range"]
        if radar_only:
            lab = ("mmWave %.1f m → rover" % rng) if rng is not None else "mmWave → rover"
        else:
            lab = ("target %.1f m" % rng) if rng is not None else "target"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        lx = max(2, min(col - tw // 2, w - tw - 4))
        cv2.rectangle(canvas, (lx - 3, 26), (lx + tw + 3, 26 + th + 8), (0, 0, 0), -1)
        cv2.putText(canvas, lab, (lx, 26 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _encode(self, canvas):
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if not ok:
            raise RuntimeError("detector JPEG encode failed")
        return buf.tobytes()

    def _encode_panel(self, img, dets, tag):
        """Boxes (no contrib labels) + a corner tag, for the thermal/depth panels."""
        self._draw_boxes(img, dets, labels=False)
        cv2.rectangle(img, (0, 0), (len(tag) * 11 + 10, 20), (0, 0, 0), -1)
        cv2.putText(img, tag, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        return self._encode(img)

    def _debounced_scale(self, key, target, t):
        """Follow the DQN's chosen resolution tier, but only change the SHOWN
        resolution after the new tier has held RESOLUTION_DEBOUNCE seconds -> the
        displayed resolution steps slowly instead of flickering frame-to-frame."""
        d = self._scale_deb[key]
        if target == d["disp"]:
            d["cand"], d["cand_t"] = target, t          # nothing pending
        elif target != d["cand"]:
            d["cand"], d["cand_t"] = target, t          # new candidate -> start timer
        elif t - d["cand_t"] >= float(config.RESOLUTION_DEBOUNCE):
            d["disp"] = target                          # held long enough -> commit
        return d["disp"]

    @staticmethod
    def _degrade(img, scale):
        """Down- then up-sample so the frame visibly carries the lower resolution
        (blocky), while staying the same display size. Overlays are drawn AFTER."""
        if scale >= 0.999:
            return img
        h, w = img.shape[:2]
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _render(self, rgb, dets, settings, mock, tgt=None):
        if config.DETECT_SHOW_RES:
            rgb = self._degrade(rgb, self._debounced_scale("rgb", settings.get("rgb_scale", 1.0), time.time()))
        canvas = rgb.copy()
        self._draw_boxes(canvas, dets, labels=True)
        save = (1.0 - settings.get("compute_cost", 1.0)) * 100.0
        hud = (f"{'MOCK ' if mock else ''}det {len(dets)} | "
               f"{self._fps:4.1f}/{self._render_fps:4.1f} fps det/disp | "
               f"RGB {settings['rgb_res']} / T {settings['therm_res']} | save {save:.0f}%")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 255), 1)
        self._draw_bearing(canvas, tgt)
        return self._encode(canvas)

    # ---- mmWave presence (drives the RGB bearing/distance line) --------
    def _reset_presence(self):
        """Clear the legacy presence-clustering state (rover started moving)."""
        if self._rwin is not None:
            self._rwin.clear()
        self._rpresent = False
        self._rlast_best = None

    def _ingest_radar(self, points):
        c = self._rcfg
        qual, best, _ = self._radar_eval(points, c)
        if best is not None:
            self._rlast_best = best
        self._rwin.append(qual)
        if not self._rpresent and sum(list(self._rwin)[-c.enter_win:]) >= c.enter_need:
            self._rpresent = True
        elif self._rpresent and sum(self._rwin) == 0:
            self._rpresent = False
            self._rlast_best = None
        conf = self._radar_conf(self._rwin, best, c)
        with self._rlock:
            if self._rpresent:
                b = best or self._rlast_best
                az = b["az"] if b else None
                if az is not None:   # smooth the bearing so the line stops jittering
                    a = float(config.RADAR_AZ_SMOOTH)
                    self._raz = az if self._raz is None else (1 - a) * self._raz + a * az
                    az = self._raz
                self._rinfo = {"range": (b["range"] if b else None), "az": az, "conf": conf}
            else:
                self._rinfo = None
                self._raz = None

    def _get_rinfo(self):
        with self._rlock:
            # Stale guard: no radar frame in 2 s -> drop the bearing (safety).
            if self._radar_last_t and time.time() - self._radar_last_t > 2.0:
                return None
            return dict(self._rinfo) if self._rinfo else None

    def _radar_det(self, w, h, rinfo):
        """Synthetic detection from a radar-only presence (no YOLO box): an
        approximate box at the target azimuth, contribution = Radar 100%."""
        az, rng = rinfo.get("az"), rinfo.get("range")
        if az is not None:
            a = -az if config.DETECT_RADAR_FLIP else az
            frac = max(-1.0, min(1.0, a / (float(config.DETECT_RGB_HFOV) / 2.0)))
            cx = w * 0.5 * (1.0 + frac)
        else:
            cx = w * 0.5
        bw, bh = w * 0.14, h * 0.55
        x1, x2 = max(0.0, cx - bw / 2), min(float(w), cx + bw / 2)
        y1 = h * 0.30
        y2 = min(float(h), y1 + bh)
        return {"box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(float(rinfo.get("conf", 0.5)), 3),
                "contrib": [0.0, 0.0, 1.0], "radar": True,
                "range": rng, "range_src": "radar"}

    def _assemble_dets(self, pb, ps, pc, w, h, rinfo):
        """Return the ONE correct detection (a 0- or 1-element list).

        Policy — VISION FIRST, mmWave only as a fallback:
          * any vision (RGB/thermal) box -> trust the most-confident one if it
            clears DETECT_CONF_NORADAR. The radar never selects or rejects
            vision boxes; its measured range is still fused in (vision has no
            range), but that is distance fusion, not detection.
          * NO vision box -> radar-only detection. rinfo is only ever non-None
            here while the rover itself is stationary AND a moving cluster was
            acquired (see the ingestion gate in read_once), so a radar-driven
            target can never come from the rover's own ego-motion.
        """
        n = len(pb)
        chosen = None
        if n > 0:
            best = max(range(n), key=lambda i: float(ps[i]) if i < len(ps) else 0.0)
            if (float(ps[best]) if best < len(ps) else 0.0) >= float(config.DETECT_CONF_NORADAR):
                chosen = best

        if chosen is not None:
            dets = self._build_dets(pb[chosen:chosen + 1], ps[chosen:chosen + 1], pc[chosen:chosen + 1])
            # Distance for a vision detection: D435 depth is the baseline (same
            # view as RGB, aligned per-pixel). When depth drops out we fall back
            # to the radar (mmWave) range — BUT if that radar range is wildly
            # different from the last depth (a spurious jump on the handoff), we
            # keep using the last depth value for a short hold instead of
            # teleporting the distance. Radar is accepted once it's close again,
            # or after DETECT_DEPTH_HOLD_S, or if there was no recent depth.
            for d in dets:
                dr = self._depth_range(d["box"], w, h)
                now = time.time()
                radar_rng = rinfo.get("range") if rinfo is not None else None
                if dr is not None:
                    d["range"] = round(dr, 3)
                    d["range_src"] = "depth"
                    self._last_depth = (dr, now)
                elif (self._last_depth is not None
                      and now - self._last_depth[1] <= float(config.DETECT_DEPTH_HOLD_S)
                      and radar_rng is not None
                      and abs(radar_rng - self._last_depth[0]) > float(config.DETECT_DEPTH_HOLD_JUMP_M)):
                    d["range"] = round(self._last_depth[0], 3)
                    d["range_src"] = "depth_hold"
                elif radar_rng is not None:
                    d["range"] = round(radar_rng, 3)
                    d["range_src"] = "radar"
                else:
                    d["range"] = None
                    d["range_src"] = None
            return dets
        if rinfo is not None and rinfo.get("az") is not None:
            return [self._radar_det(w, h, rinfo)]
        return []

    def _depth_range(self, box, w, h):
        """Person distance (m) from the color-ALIGNED D435 depth: median over
        the box's upper-torso region (middle 50% width, 15-55% height — skips
        the head edge and the legs/floor). None when depth is unavailable,
        resolution-mismatched, too holey, or out of the trustworthy range."""
        buf = getattr(self._rgb_sensor, "depth_raw", None)
        if buf is None:
            return None
        val, _ = buf.get()
        if val is None:
            return None
        draw, scale = val
        if draw.shape[0] != h or draw.shape[1] != w:
            return None
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        sx1 = int(max(0.0, x1 + 0.25 * bw)); sx2 = int(min(float(w), x2 - 0.25 * bw))
        sy1 = int(max(0.0, y1 + 0.15 * bh)); sy2 = int(min(float(h), y1 + 0.55 * bh))
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        patch = draw[sy1:sy2, sx1:sx2]
        valid = patch[patch > 0]                  # 0 = depth hole
        if valid.size < 30 or valid.size < 0.1 * patch.size:
            return None
        rng = float(np.median(valid)) * float(scale)
        if not (0.15 <= rng <= float(config.D435_DEPTH_MAX_M)):
            return None
        return rng

    @staticmethod
    def _step(cur, tgt, alpha, max_step):
        """EMA toward tgt, but cap the actual move so a teleport becomes a slide."""
        move = alpha * (tgt - cur)
        if move > max_step:
            move = max_step
        elif move < -max_step:
            move = -max_step
        return cur + move

    def _smooth_target(self, raw, w, h, t):
        """Smooth the single selected detection's box across frames AND source
        switches so it never teleports. Returns a smoothed det (or None). Coasts
        on the last box for TARGET_COAST seconds when detection briefly drops."""
        a = float(config.TARGET_SMOOTH)
        max_step = float(config.TARGET_MAX_STEP) * w
        if raw is not None:
            x1, y1, x2, y2 = raw["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = (x2 - x1), (y2 - y1)
            if self._sm is None:
                self._sm = {"cx": cx, "cy": cy, "bw": bw, "bh": bh, "range": raw.get("range"),
                            "range_src": raw.get("range_src"),
                            "score": raw.get("score", 0.0), "contrib": raw.get("contrib", [0, 0, 0]),
                            "radar": bool(raw.get("radar")), "t": t}
            else:
                s = self._sm
                s["cx"] = self._step(s["cx"], cx, a, max_step)
                s["cy"] = self._step(s["cy"], cy, a, max_step)
                s["bw"] = 0.25 * bw + 0.75 * s["bw"]
                s["bh"] = 0.25 * bh + 0.75 * s["bh"]
                r = raw.get("range")
                if r is not None:
                    s["range"] = r if s["range"] is None else 0.3 * r + 0.7 * s["range"]
                    s["range_src"] = raw.get("range_src")
                s["score"] = raw.get("score", 0.0)
                s["contrib"] = raw.get("contrib", s["contrib"])
                s["radar"] = bool(raw.get("radar"))
                s["t"] = t
        elif self._sm is None or (t - self._sm["t"]) > float(config.TARGET_COAST):
            self._sm = None
            self._last_depth = None     # target gone: don't hold its depth onto the next one
            return None
        # else: coast — keep the last smoothed box for a moment.
        s = self._sm
        cx, cy, bw, bh = s["cx"], s["cy"], s["bw"], s["bh"]
        return {"box": [round(cx - bw / 2, 1), round(cy - bh / 2, 1),
                        round(cx + bw / 2, 1), round(cy + bh / 2, 1)],
                "score": round(float(s["score"]), 3), "contrib": s["contrib"],
                "radar": s["radar"]}

    def _target_from_det(self, sdet, w):
        """Control target {x, az, range, source} from the smoothed det center."""
        if sdet is None:
            return None
        cx = (sdet["box"][0] + sdet["box"][2]) / 2.0
        return {"x": float(cx), "az": round((cx / w - 0.5) * float(config.DETECT_RGB_HFOV), 1),
                "range": self._sm["range"] if self._sm else None,
                "range_src": self._sm.get("range_src") if self._sm else None,
                "source": "radar" if sdet.get("radar") else "vision"}

    def _telemetry(self, dets, settings, w, h, mock):
        if dets:
            agg = np.mean([d["contrib"] for d in dets], axis=0)
            contrib = [round(float(agg[0]), 3), round(float(agg[1]), 3), round(float(agg[2]), 3)]
        else:
            contrib = [0.0, 0.0, 0.0]
        return {
            "t": time.time(),
            "mock": mock,
            "num": len(dets),
            "detections": dets,
            "contrib": contrib,
            "fps": round(self._fps, 1),
            "w": w, "h": h,
            "adaptive": {
                "rgb_res": settings["rgb_res"],
                "therm_res": settings["therm_res"],
                "rgb_fps": settings["rgb_fps"],
                "therm_fps": settings["therm_fps"],
                "savings": round((1.0 - settings.get("compute_cost", 1.0)) * 100.0, 0),
            },
        }

    def _mock_settings(self, phase):
        # In bright (day) scenes the agent can downscale thermal; at night it
        # keeps thermal high. Mirrors the real controller's qualitative behavior.
        if phase > 0.6:
            rgb_res, therm_res, cost = "full", "1/2", 0.62
            rfps, tfps = 30, 10
        elif phase < 0.4:
            rgb_res, therm_res, cost = "1/2", "full", 0.62
            rfps, tfps = 10, 30
        else:
            rgb_res, therm_res, cost = "3/4", "3/4", 0.56
            rfps, tfps = 20, 20
        return {"rgb_scale": 1.0, "therm_scale": 1.0, "rgb_res": rgb_res,
                "therm_res": therm_res, "rgb_fps": rfps, "therm_fps": tfps,
                "compute_cost": cost}


class _AdaptiveController:
    """Thin wrapper around the trained DQN policy (mirrors run_live.py).

    Holds the contribution / confidence / detection-count EMAs that form the
    9-dim state, runs the Q-network, and applies a DETECT_TIER_HOLD_FRAMES
    hysteresis before committing to a new resolution/framerate tier.
    """

    def __init__(self, model_path, QNetwork, state_dim, num_actions,
                 res_tiers, fps_tiers, action_to_tiers, torch, device):
        self._torch = torch
        self._device = device
        self._res_tiers = res_tiers
        self._fps_tiers = fps_tiers
        self._action_to_tiers = action_to_tiers

        self.q_net = QNetwork(state_dim, num_actions).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_net.eval()

        self.contrib_ema = np.array([0.4, 0.5, 0.1], dtype=np.float32)
        self.conf_ema = 0.5
        self.n_dets_ema = 2.0
        self.rgb_tier = 0
        self.therm_tier = 0
        self._pending = None
        self._pending_count = 0

    def _settings(self):
        r, t = self.rgb_tier, self.therm_tier
        return {
            "rgb_scale": self._res_tiers[r][0], "rgb_res": self._res_tiers[r][1],
            "rgb_fps": self._fps_tiers[r],
            "therm_scale": self._res_tiers[t][0], "therm_res": self._res_tiers[t][1],
            "therm_fps": self._fps_tiers[t],
            "compute_cost": (self._res_tiers[r][2] + self._res_tiers[t][2]) / 2.0,
        }

    def decide(self):
        """Current committed settings (used to size this frame's inputs)."""
        return self._settings()

    def update(self, contribs, avg_conf, n_dets):
        torch = self._torch
        if contribs is not None and len(contribs) > 0:
            self.contrib_ema = 0.3 * contribs.mean(axis=0).astype(np.float32) + 0.7 * self.contrib_ema
        self.conf_ema = 0.3 * avg_conf + 0.7 * self.conf_ema
        self.n_dets_ema = 0.3 * n_dets + 0.7 * self.n_dets_ema

        state = np.array([
            self.contrib_ema[0], self.contrib_ema[1], self.contrib_ema[2],
            self.conf_ema, self.n_dets_ema / 20.0,
            self.rgb_tier / 3.0, self.therm_tier / 3.0,
            self._fps_tiers[self.rgb_tier] / 30.0, self._fps_tiers[self.therm_tier] / 30.0,
        ], dtype=np.float32)

        with torch.no_grad():
            q = self.q_net(torch.tensor(state).unsqueeze(0).to(self._device))
            action = int(q.argmax(1).item())

        if action != self._pending:
            self._pending = action
            self._pending_count = 1
        else:
            self._pending_count += 1
        # Hysteresis: the DQN must pick the SAME new action this many frames in
        # a row before the tier commits (~ frames / DETECT_FPS seconds). Keeps
        # the audience tier boxes / displayed resolution from flapping.
        if self._pending_count >= int(config.DETECT_TIER_HOLD_FRAMES):
            self.rgb_tier, self.therm_tier = self._action_to_tiers(action)

        return self._settings()
