"""TI IWR6843ISK mmWave radar reader.

The IWR6843ISK streams detected-object data over its high-speed UART as a
sequence of TLV (type-length-value) packets framed by a fixed magic word. We
parse the detected-points TLV into an (x, y, z, velocity) point cloud and
publish the latest frame for the UI's 2D radar plot.

Only the parsing needed for the demo is implemented; additional TLV types
(range profile, heatmaps) can be added without touching the rest of the system.
"""
import math
import struct
import time

from .base import SensorThread
from .. import config

try:
    import serial
except Exception:  # pragma: no cover
    serial = None

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7   # per-point SNR + noise (int16 each, 0.1 dB units), same order as points


class MmWaveRadar(SensorThread):
    name = "mmwave_radar"

    def __init__(self, allow_mock: bool = True):
        super().__init__(allow_mock)
        self._cli = None
        self._data = None
        self._buf = bytearray()
        self._last_frame_t = 0.0   # for stall detection (monotonic)

    def open(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial not installed")
        # Optionally push a chirp config to the CLI port.
        if config.MMWAVE_CONFIG_FILE:
            self._cli = serial.Serial(config.MMWAVE_CLI_PORT, config.MMWAVE_CLI_BAUD, timeout=1)
            self._send_config(config.MMWAVE_CONFIG_FILE)
        self._data = serial.Serial(config.MMWAVE_DATA_PORT, config.MMWAVE_DATA_BAUD, timeout=0.1)
        self._buf = bytearray()
        self._last_frame_t = time.monotonic()

    def _send_config(self, path: str) -> None:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                self._cli.write((line + "\n").encode())
                time.sleep(0.01)
                self._cli.reset_input_buffer()

    def read_once(self):
        chunk = self._data.read(4096)
        if chunk:
            self._buf.extend(chunk)
        frame = self._parse_buffer()
        now = time.monotonic()
        if frame is not None:
            self._last_frame_t = now
        elif now - self._last_frame_t > config.MMWAVE_STALL_TIMEOUT:
            # Data UART silent too long -> IWR stalled. Raise so the run loop
            # reopens the ports + re-sends the cfg (auto-recovery) instead of
            # sitting dead. (Empty scenes still emit frames, so this is a true stall.)
            raise RuntimeError(
                f"mmwave data stalled: no frame for {now - self._last_frame_t:.1f}s")
        return frame  # may be None until a full frame is buffered

    def _parse_buffer(self):
        """Extract the most recent complete frame from the rolling buffer."""
        result = None
        while True:
            idx = self._buf.find(MAGIC_WORD)
            if idx < 0:
                # Keep a tail in case the magic word straddles reads.
                if len(self._buf) > len(MAGIC_WORD):
                    del self._buf[:-len(MAGIC_WORD)]
                break
            if idx > 0:
                del self._buf[:idx]
            # Header is 40 bytes (magic 8 + 9 uint32 fields).
            if len(self._buf) < 40:
                break
            # magic(8s) + 8 u32: version, totalPacketLen, platform, frameNumber,
            # timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber.
            header = struct.unpack("<8s8I", self._buf[:40])
            total_len = header[2]  # totalPacketLen (header[3] is platform)
            num_tlvs = header[7]
            if total_len < 40 or total_len > 65536:
                # Corrupt header; skip past this magic word.
                del self._buf[:len(MAGIC_WORD)]
                continue
            if len(self._buf) < total_len:
                break  # wait for the rest of the frame
            packet = bytes(self._buf[:total_len])
            del self._buf[:total_len]
            parsed = self._parse_packet(packet, num_tlvs)
            if parsed is not None:
                result = parsed  # keep the newest complete frame
        return result

    def _parse_packet(self, packet: bytes, num_tlvs: int):
        offset = 40
        points = []
        side = []   # (snr, noise) per point, parallel to points
        for _ in range(num_tlvs):
            if offset + 8 > len(packet):
                break
            tlv_type, tlv_len = struct.unpack("<2I", packet[offset:offset + 8])
            offset += 8
            payload = packet[offset:offset + tlv_len]
            offset += tlv_len
            if tlv_type == TLV_DETECTED_POINTS:
                # Each point: x, y, z, velocity as float32 (16 bytes).
                count = len(payload) // 16
                for i in range(count):
                    x, y, z, v = struct.unpack_from("<4f", payload, i * 16)
                    points.append({"x": x, "y": y, "z": z, "v": v})
            elif tlv_type == TLV_SIDE_INFO:
                # Per-point SNR + noise as int16 (4 bytes/point), same order as points.
                count = len(payload) // 4
                for i in range(count):
                    snr, noise = struct.unpack_from("<2h", payload, i * 4)
                    side.append((snr, noise))
        # Attach SNR (0.1 dB units) to each point when the side-info TLV is present.
        for i, p in enumerate(points):
            if i < len(side):
                p["snr"] = side[i][0]
                p["noise"] = side[i][1]
        return {"t": time.time(), "points": points, "num": len(points)}

    def read_mock(self):
        t = time.time()
        n = 18
        points = []
        for i in range(n):
            ang = (i / n) * 1.4 - 0.7  # ~80 deg FoV
            r = 2.0 + 1.0 * math.sin(t * 0.7 + i)
            cluster = abs(ang) < 0.25  # fake a person dead ahead
            if cluster:
                r = 1.8 + 0.1 * math.sin(t * 3 + i)
            x = r * math.sin(ang)
            y = r * math.cos(ang)
            v = (0.3 if cluster else 0.0) * math.sin(t * 2 + i)
            points.append({"x": x, "y": y, "z": 0.0, "v": v})
        time.sleep(0.05)
        return {"t": t, "points": points, "num": len(points)}

    def close(self) -> None:
        for port in (self._data, self._cli):
            if port is not None:
                try:
                    port.close()
                except Exception:
                    pass
        self._data = None
        self._cli = None
