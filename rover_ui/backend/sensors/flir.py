"""FLIR One Pro thermal camera, read as a V4L2 device via OpenCV.

Produces JPEG-encoded frames (colormapped if the source is single channel) so
the HTTP layer can stream them as MJPEG with no per-request re-encoding.
"""
import time

import numpy as np

from .base import SensorThread, LatestValue
from .. import config

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may be absent in dev
    cv2 = None

_COLORMAPS = {
    "INFERNO": getattr(cv2, "COLORMAP_INFERNO", 14) if cv2 else 14,
    "JET": getattr(cv2, "COLORMAP_JET", 2) if cv2 else 2,
    "HOT": getattr(cv2, "COLORMAP_HOT", 11) if cv2 else 11,
    "PLASMA": getattr(cv2, "COLORMAP_PLASMA", 15) if cv2 else 15,
}


class FlirThermal(SensorThread):
    name = "flir_thermal"

    def __init__(self, allow_mock: bool = True):
        super().__init__(allow_mock)
        self._cap = None
        # Side-channel buffer holding the latest *raw* thermal frame (grayscale,
        # pre-colormap) for the detector. The MJPEG path still uses self.latest.
        self.raw = LatestValue()
        self._colormap = _COLORMAPS.get(config.FLIR_COLORMAP, _COLORMAPS["INFERNO"])
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY] if cv2 else []
        self._frame_interval = 1.0 / max(config.FLIR_FPS, 1)
        self._next_frame_t = 0.0

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python not installed")
        # Parse "/dev/videoN" -> index N is not reliable across drivers; pass the
        # path directly, which V4L2 backend accepts.
        cap = cv2.VideoCapture(config.FLIR_DEVICE, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open FLIR device {config.FLIR_DEVICE}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FLIR_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FLIR_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FLIR_FPS)
        # Small internal buffer keeps latency low (we always want the newest frame).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def _encode(self, frame: np.ndarray) -> bytes:
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            gray = frame if frame.ndim == 2 else frame[:, :, 0]
            # Normalize for a stable colormap regardless of raw range.
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            frame = cv2.applyColorMap(norm, self._colormap)
        ok, buf = cv2.imencode(".jpg", frame, self._encode_params)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def read_once(self):
        # Pace to the configured FPS to avoid spinning the CPU.
        now = time.monotonic()
        if now < self._next_frame_t:
            time.sleep(max(0.0, self._next_frame_t - now))
        self._next_frame_t = time.monotonic() + self._frame_interval

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("FLIR read failed")
        # Stash the raw grayscale thermal frame for the detector (modality input
        # must be grayscale, not the colormapped display frame).
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.raw.set(gray)
        return self._encode(frame)

    def read_mock(self):
        if cv2 is None:
            return None
        # Moving warm blob on a cool background to mimic a person.
        h, w = config.FLIR_HEIGHT, config.FLIR_WIDTH
        t = time.time()
        yy, xx = np.mgrid[0:h, 0:w]
        cx = int(w * (0.5 + 0.3 * np.sin(t * 0.6)))
        cy = int(h * (0.5 + 0.2 * np.cos(t * 0.4)))
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = np.exp(-dist2 / (2 * (min(h, w) * 0.12) ** 2))
        bg = 0.25 + 0.05 * np.sin(t + xx * 0.01)
        field = np.clip(bg + 0.75 * blob, 0, 1)
        gray = (field * 255).astype("uint8")
        self.raw.set(gray)
        time.sleep(self._frame_interval)
        return self._encode(gray)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
