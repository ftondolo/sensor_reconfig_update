"""Intel RealSense sensors via pyrealsense2.

Two independent producers share this module:

* :class:`D435iRGB`  - color stream from the D435i, encoded to JPEG for MJPEG.
* :class:`T265Pose`  - 6-DoF pose from the T265, published as a dict for the
  trajectory view and consumed by the return-to-start controller.

They use separate pipelines (and ideally separate processes/threads) so the
high-rate pose stream is never throttled by RGB encoding.
"""
import math
import threading
import time

import numpy as np

from .base import SensorThread, LatestValue
from .. import config

_DEPTH_COLORMAPS = {
    "JET": 2, "INFERNO": 14, "PLASMA": 15, "TURBO": 20, "HOT": 11,
}


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover
    rs = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class D435iRGB(SensorThread):
    name = "d435i_rgb"

    def __init__(self, allow_mock: bool = True):
        super().__init__(allow_mock)
        self._pipe = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY] if cv2 else []
        self._depth_enabled = bool(config.D435_ENABLE_DEPTH)
        self._depth_cmap = _DEPTH_COLORMAPS.get(config.D435_DEPTH_COLORMAP, 2)
        self._depth_scale = 0.001  # m per unit; refined from the device at open()
        # Side-channel buffers: the raw BGR color frame (for the detector), a
        # colormapped depth JPEG (for the audience MJPEG stream), and the
        # color-ALIGNED raw 16-bit depth as (uint16 array, meters_per_unit) so
        # the detector can read a person's distance inside an RGB box.
        self.raw = LatestValue()
        self.depth_jpeg = LatestValue()
        self.depth_raw = LatestValue()
        # Depth->color registration via a precomputed remap LUT (built once from
        # the camera intrinsics/extrinsics at a nominal depth). Each frame is then
        # a cheap cv2.remap (~3ms) instead of rs.align (~110ms), so depth aligns
        # to RGB at FULL fps. Approximate (assumes ~nominal depth for parallax).
        self._remap_x = None
        self._remap_y = None

    def open(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        if cv2 is None:
            raise RuntimeError("opencv-python not installed")
        pipe = rs.pipeline()
        cfg = rs.config()
        if config.D435_SERIAL:
            cfg.enable_device(config.D435_SERIAL)
        cfg.enable_stream(
            rs.stream.color, config.D435_WIDTH, config.D435_HEIGHT,
            rs.format.bgr8, config.D435_FPS,
        )
        if self._depth_enabled:
            cfg.enable_stream(
                rs.stream.depth, config.D435_WIDTH, config.D435_HEIGHT,
                rs.format.z16, config.D435_FPS,
            )
        profile = pipe.start(cfg)
        if self._depth_enabled:
            try:
                self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            except Exception:
                self._depth_scale = 0.001
            # Build the cheap depth->color remap once (so the box lands on the
            # person in depth, at full fps). Falls back to unaligned on failure.
            try:
                self._build_align_remap(profile)
            except Exception as exc:
                self._remap_x = self._remap_y = None
                print(f"[D435i] depth-align remap unavailable ({exc}); depth unaligned")
        self._pipe = pipe

    def _build_align_remap(self, profile, nominal_z=1.5):
        """Precompute, for every COLOR pixel, the DEPTH pixel it samples (assuming
        a nominal scene depth). cv2.remap with these LUTs registers depth->color."""
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci, di = cprof.get_intrinsics(), dprof.get_intrinsics()
        ext = cprof.get_extrinsics_to(dprof)                 # color -> depth
        Rm = np.array(ext.rotation, np.float64).reshape(3, 3).T
        t = np.array(ext.translation, np.float64)
        us, vs = np.meshgrid(np.arange(ci.width), np.arange(ci.height))
        z = float(nominal_z)
        Pc = np.stack([(us - ci.ppx) / ci.fx * z,
                       (vs - ci.ppy) / ci.fy * z,
                       np.full(us.shape, z)], axis=-1)        # 3D in color frame
        Pd = np.einsum("ij,hwj->hwi", Rm, Pc) + t            # -> depth frame
        zd = np.where(np.abs(Pd[..., 2]) < 1e-6, 1e-6, Pd[..., 2])
        self._remap_x = (di.fx * Pd[..., 0] / zd + di.ppx).astype(np.float32)
        self._remap_y = (di.fy * Pd[..., 1] / zd + di.ppy).astype(np.float32)

    def _depth_colormap(self, depth_raw: np.ndarray) -> np.ndarray:
        meters = depth_raw.astype("float32") * self._depth_scale
        far = max(config.D435_DEPTH_MAX_M, 0.1)
        norm = np.clip(meters / far, 0.0, 1.0)
        vis = cv2.applyColorMap((norm * 255).astype("uint8"), self._depth_cmap)
        vis[depth_raw == 0] = (0, 0, 0)   # holes -> black
        return vis

    def _encode_depth(self, depth_aligned: np.ndarray) -> bytes:
        """Colormap an (already color-aligned) 16-bit depth frame to a JPEG."""
        vis = self._depth_colormap(depth_aligned)
        ok, buf = cv2.imencode(".jpg", vis, self._encode_params)
        return buf.tobytes() if ok else b""

    def _publish_depth(self, draw: np.ndarray) -> None:
        """Align a raw 16-bit depth frame to the color view (cheap remap on the
        single channel), publish it for the detector, then colormap for MJPEG."""
        if self._remap_x is not None:
            draw = cv2.remap(draw, self._remap_x, self._remap_y, cv2.INTER_NEAREST,
                             borderValue=0)
        self.depth_raw.set((draw, self._depth_scale))
        jpeg = self._encode_depth(draw)
        if jpeg:
            self.depth_jpeg.set(jpeg)

    def read_once(self):
        frames = self._pipe.wait_for_frames(2000)
        color = frames.get_color_frame()
        if not color:
            return None
        img = np.asanyarray(color.get_data())
        self.raw.set(img)                       # RGB published immediately (fast)
        if self._depth_enabled:
            depth = frames.get_depth_frame()
            if depth:
                self._publish_depth(np.asanyarray(depth.get_data()))
        ok, buf = cv2.imencode(".jpg", img, self._encode_params)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def read_mock(self):
        if cv2 is None:
            return None
        h, w = config.D435_HEIGHT, config.D435_WIDTH
        t = time.time()
        img = np.zeros((h, w, 3), dtype="uint8")
        img[:] = (40, 40, 40)
        # A drifting rectangle standing in for a tracked subject.
        cx = int(w * (0.5 + 0.3 * math.sin(t * 0.5)))
        cy = int(h * 0.55)
        cv2.rectangle(img, (cx - 40, cy - 90), (cx + 40, cy + 90), (0, 180, 255), -1)
        cv2.putText(img, "D435i MOCK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        self.raw.set(img)
        if self._depth_enabled:
            # Synthetic depth: nearer (warm) at the drifting subject, far elsewhere.
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.hypot(xx - cx, yy - cy) / float(max(h, w))
            meters = np.clip(0.6 + d * config.D435_DEPTH_MAX_M, 0, config.D435_DEPTH_MAX_M)
            raw = (meters / max(self._depth_scale, 1e-6)).astype("uint16")
            self._publish_depth(raw)
        ok, buf = cv2.imencode(".jpg", img, self._encode_params)
        time.sleep(1.0 / max(config.D435_FPS, 1))
        return buf.tobytes() if ok else None

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None


def _quat_to_yaw(x, y, z, w) -> float:
    """Yaw (rotation about vertical axis) from a quaternion, in radians.

    The T265 reports pose in a right-handed system with -Z forward and +Y up,
    so heading is the rotation about the Y axis.
    """
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return math.atan2(siny_cosp, cosy_cosp)


class T265Pose(SensorThread):
    name = "t265_pose"

    def __init__(self, allow_mock: bool = True):
        super().__init__(allow_mock)
        self._pipe = None
        self._interval = 1.0 / max(config.T265_POSE_HZ, 1)
        self._mock_t0 = None
        # Re-origin state for "Set Start Point": capture the next sample as the
        # origin, then report every sample relative to it.
        self._origin = None        # (x0, y0, yaw0) in the reported frame
        self._want_zero = False
        self._origin_lock = threading.Lock()

    def zero_origin(self) -> None:
        """Make the *current* pose the (0, 0, 0) reference.

        Called by 'Set Start Point'. The next sample is captured as the origin
        and all later samples are reported relative to it, rotated into the
        rover's heading-at-start frame so the rover starts at (0,0,0) facing
        +y. Convention: x=right(+), y=forward(+), yaw=CCW+ (clockwise decreases).
        """
        with self._origin_lock:
            self._want_zero = True

    def _apply_origin(self, s: dict) -> dict:
        """Re-express a sample relative to the recorded origin pose."""
        with self._origin_lock:
            if self._want_zero:
                self._origin = (s["x"], s["y"], s["yaw"])
                self._want_zero = False
            origin = self._origin
        if origin is None:
            return s
        x0, y0, yaw0 = origin
        dx, dy = s["x"] - x0, s["y"] - y0
        c, sn = math.cos(yaw0), math.sin(yaw0)
        s["x"] = dx * c + dy * sn        # right, in the start-heading frame
        s["y"] = -dx * sn + dy * c       # forward, in the start-heading frame
        s["yaw"] = _wrap(s["yaw"] - yaw0)
        return s

    def open(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        pipe = rs.pipeline()
        cfg = rs.config()
        if config.T265_SERIAL:
            cfg.enable_device(config.T265_SERIAL)
        cfg.enable_stream(rs.stream.pose)
        pipe.start(cfg)
        self._pipe = pipe

    def _sample(self, x, y, z, yaw, confidence) -> dict:
        # Rover ground-frame convention used everywhere downstream:
        #   x = right (+), y = forward (+), yaw = CCW+ (clockwise decreases).
        # T265 device frame: +x right, +y up, +z back  =>  forward = -z.
        return {
            "t": time.time(),
            "x": x,         # right (T265 +x)
            "y": -z,        # forward (T265 -z)
            "z": y,         # up (unused by the 2D plot, kept for 3D later)
            "yaw": yaw,
            "confidence": confidence,
        }

    def read_once(self):
        frames = self._pipe.wait_for_frames(2000)
        pose = frames.get_pose_frame()
        if not pose:
            return None
        data = pose.get_pose_data()
        tr = data.translation
        rot = data.rotation
        yaw = _quat_to_yaw(rot.x, rot.y, rot.z, rot.w)
        time.sleep(self._interval)
        return self._apply_origin(
            self._sample(tr.x, tr.y, tr.z, yaw, data.tracker_confidence))

    def read_mock(self):
        if self._mock_t0 is None:
            self._mock_t0 = time.time()
        t = time.time() - self._mock_t0
        # Drive a gentle figure-eight so the trajectory view has something to show.
        x = 1.5 * math.sin(t * 0.2)
        y = 1.0 * math.sin(t * 0.4)
        yaw = 0.4 * math.sin(t * 0.2)
        time.sleep(self._interval)
        # mock maps to the same projected frame the real path uses.
        return self._apply_origin(
            {"t": time.time(), "x": x, "y": y, "z": 0.0, "yaw": yaw, "confidence": 3})

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None
