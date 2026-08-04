"""FastAPI backend for the rover demo UI (map navigation).

Movement is map-based:

  * a pre-stored 6000 x 6000 mm BEV map (map.json) with marked obstacle regions;
  * the detector's fused target (bearing + radar range) is projected onto that
    map continuously;
  * the rover goal = the target shifted 2 m toward the rover side (adjusted off
    any obstacle region), planned around obstacles with rectilinear_mm and
    executed leg-by-leg through purely_control.T265RoverService.move_axis().

The T265 is owned exclusively by the Navigator's T265RoverService — do not start
a second T265 pose sensor thread alongside it.

Endpoints:
  GET  /                       -> operator console (map + nav controls)
  GET  /audience               -> audience display (big BEV map + sensors)
  GET  /api/status             -> snapshot of sensor + nav status
  GET  /api/map                -> static map payload (size, obstacles, car)
  GET  /stream/flir|d435|depth|detect|detect_thermal|detect_depth -> MJPEG
  WS   /ws/telemetry           -> {mmwave, detection, nav, status} pushed
  POST /api/nav/go             -> navigate to the current detection target
  POST /api/nav/goto           -> {x, z} navigate to a manual map target (mm)
  POST /api/nav/cancel         -> cancel the active navigation (stops the rover)
  POST /api/nav/auto           -> {enabled} toggle auto-navigate on detections
  POST /api/nav/follow         -> {enabled} toggle live-follow (re-plan on target move)
  POST /api/nav/ignore_obstacles -> {enabled} ignore the rover's own position when it
                                    overlaps an obstacle/clearance zone (start/continue)
  POST /api/nav/speed          -> {mps} set the navigation speed cap (clamped)
  POST /api/nav/jog_speed      -> {mps} set the Drive-pad hold-to-move speed
  POST /api/nav/confirm_n      -> {n} set the ghost-guard confirm-count (1-10, clamped)
  POST /api/nav/standoff       -> {mm} set the target approach distance (clamped)
  POST /api/nav/reset_pose     -> {x?, z?} anchor the current pose to a map cell
  POST /api/nav/nudge          -> {axis, mm} small open-map setup move
"""
import asyncio
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .sensors.flir import FlirThermal
from .sensors.realsense import D435iRGB
from .sensors.mmwave import MmWaveRadar
from .detection.detector import DetectorThread
from .nav.navigator import Navigator

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app = FastAPI(title="Rover Demo UI — map navigation")

# ---- shared singletons (created at startup) ----------------------------
sensors = {}
detector: DetectorThread = None  # type: ignore
nav: Navigator = None  # type: ignore


@app.on_event("startup")
def _startup():
    global detector, nav
    flir = FlirThermal(allow_mock=config.ALLOW_MOCK)
    d435 = D435iRGB(allow_mock=config.ALLOW_MOCK)
    mmwave = MmWaveRadar(allow_mock=config.ALLOW_MOCK)
    sensors.update({
        "flir_thermal": flir,
        "d435i_rgb": d435,
        "mmwave_radar": mmwave,
    })
    for s in sensors.values():
        s.start()
    detector = DetectorThread(rgb_sensor=d435, thermal_sensor=flir,
                              radar_sensor=mmwave, allow_mock=config.ALLOW_MOCK)
    detector.start()
    nav = Navigator(detector=detector)
    nav.start()
    # Radar tracker drops Doppler gating while the rover itself is translating.
    detector.set_rover_state_getter(nav.is_moving)


@app.on_event("shutdown")
def _shutdown():
    if nav is not None:
        nav.stop()
    if detector is not None:
        detector.stop()
    for s in sensors.values():
        s.stop()


# ---- MJPEG streaming ----------------------------------------------------
def _mjpeg_generator(buffer):
    boundary = b"--frame"
    last_seq = -1
    while True:
        frame, seq = buffer.wait_for_next(last_seq, timeout=1.0)
        if frame is None:
            continue
        last_seq = seq
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
               + frame + b"\r\n")


def _mjpeg_response(buffer):
    return StreamingResponse(
        _mjpeg_generator(buffer),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/flir")
def stream_flir():
    return _mjpeg_response(sensors["flir_thermal"].latest)


@app.get("/stream/d435")
def stream_d435():
    return _mjpeg_response(sensors["d435i_rgb"].latest)


@app.get("/stream/depth")
def stream_depth():
    return _mjpeg_response(sensors["d435i_rgb"].depth_jpeg)


@app.get("/stream/detect")
def stream_detect():
    return _mjpeg_response(detector.latest)


@app.get("/stream/detect_thermal")
def stream_detect_thermal():
    return _mjpeg_response(detector.latest_thermal)


@app.get("/stream/detect_depth")
def stream_detect_depth():
    return _mjpeg_response(detector.latest_depth)


# ---- status / map -------------------------------------------------------
def _full_status():
    snap = {name: s.status() for name, s in sensors.items()}
    if detector is not None:
        snap["detector"] = detector.status()
    return {"sensors": snap}


@app.get("/api/status")
def api_status():
    out = _full_status()
    out["nav"] = nav.state() if nav else {}
    return out


@app.get("/api/map")
def api_map():
    return nav.map_payload()


# ---- telemetry websocket -------------------------------------------------
@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    mmwave_sensor = sensors["mmwave_radar"]
    last_mm_seq = -1
    last_det_seq = -1
    try:
        while True:
            payload = {}
            mm, last_mm_seq = _maybe(mmwave_sensor.latest, last_mm_seq)
            if mm is not None:
                payload["mmwave"] = _filter_mmwave(mm)
            if detector is not None:
                det, last_det_seq = _maybe(detector.telemetry, last_det_seq)
                if det is not None:
                    payload["detection"] = det
            if nav is not None:
                payload["nav"] = nav.state()
            payload["status"] = _full_status()
            await ws.send_json(payload)
            await asyncio.sleep(0.066)  # ~15 Hz: nav/map data is low-rate
    except WebSocketDisconnect:
        return
    except Exception:
        return


def _maybe(buffer, last_seq):
    value, seq = buffer.get()
    if seq != last_seq and value is not None:
        return value, seq
    return None, last_seq


def _filter_mmwave(mm):
    """Clean the radar point cloud for display: it's only meaningful while the
    rover is stationary (ego-motion fakes Doppler on static clutter), so blank
    it while the rover is moving; and drop sub-MMWAVE_DISPLAY_MIN_V points as
    micro-jitter."""
    if nav is not None and nav.is_moving():
        return {**mm, "points": [], "num": 0, "suppressed": "moving"}
    vmin = float(config.MMWAVE_DISPLAY_MIN_V)
    pts = [p for p in mm.get("points", []) if abs(p.get("v", 0.0)) >= vmin]
    return {**mm, "points": pts, "num": len(pts)}


# ---- navigation endpoints -------------------------------------------------
class GotoBody(BaseModel):
    x: float
    z: float


class AutoBody(BaseModel):
    enabled: bool


class FollowBody(BaseModel):
    enabled: bool


class IgnoreObstaclesBody(BaseModel):
    enabled: bool


class SpeedBody(BaseModel):
    mps: float


class JogSpeedBody(BaseModel):
    mps: float


class StandoffBody(BaseModel):
    mm: float


class ConfirmNBody(BaseModel):
    n: int


class ResetPoseBody(BaseModel):
    x: Optional[float] = None
    z: Optional[float] = None


class SetStartBody(BaseModel):
    x: float
    z: float


class NudgeBody(BaseModel):
    axis: str     # "x" (right+) or "z" (back+; forward = negative)
    mm: float


class JogBody(BaseModel):
    x: int = 0    # -1 = left, +1 = right
    z: int = 0    # -1 = forward, +1 = back


@app.post("/api/nav/go")
def nav_go():
    ok, msg = nav.navigate_to_detection()
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg, "nav": nav.state()}, status_code=code)


@app.post("/api/nav/goto")
def nav_goto(body: GotoBody):
    ok, msg = nav.navigate_to_target(body.x, body.z)
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg, "nav": nav.state()}, status_code=code)


@app.post("/api/nav/cancel")
def nav_cancel():
    nav.cancel()
    return {"ok": True, "nav": nav.state()}


@app.post("/api/nav/home")
def nav_home():
    """Return the rover to the configured start cell (disables follow/auto)."""
    ok, msg = nav.return_to_start()
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg, "nav": nav.state()}, status_code=code)


@app.post("/api/nav/auto")
def nav_auto(body: AutoBody):
    nav.set_auto(body.enabled)
    return {"ok": True, "auto": body.enabled}


@app.post("/api/nav/follow")
def nav_follow(body: FollowBody):
    """Toggle live-follow: continuously re-target the detection and re-plan,
    overriding any navigation in progress (mutually exclusive with auto)."""
    nav.set_follow(body.enabled)
    return {"ok": True, "follow": body.enabled, "nav": nav.state()}


@app.post("/api/nav/ignore_obstacles")
def nav_ignore_obstacles(body: IgnoreObstaclesBody):
    """Toggle the operator override: when enabled, the rover's OWN current
    position being inside an obstacle/clearance zone no longer blocks starting
    a plan or continuing a move (obstacle avoidance everywhere else is
    unaffected). Takes effect immediately, including mid-move."""
    applied = nav.set_ignore_obstacles(body.enabled)
    return {"ok": True, "ignore_obstacles": applied, "nav": nav.state()}


@app.post("/api/nav/speed")
def nav_speed(body: SpeedBody):
    """Set the navigation speed cap (m/s), clamped to NAV_SPEED_MIN/MAX.
    Takes effect on the next control tick (even mid-move)."""
    applied = nav.set_speed(body.mps)
    return {"ok": True, "speed": applied}


@app.post("/api/nav/jog_speed")
def nav_jog_speed(body: JogSpeedBody):
    """Set the Drive-pad (hold-to-move) speed (m/s), clamped to
    NAV_JOG_SPEED_MIN/MAX (also bounded by the nav speed cap at drive time)."""
    applied = nav.set_jog_speed(body.mps)
    return {"ok": True, "jog_speed": applied}


@app.post("/api/nav/confirm_n")
def nav_confirm_n(body: ConfirmNBody):
    """Set how many mutually-consistent detector frames (1-10, clamped) must
    accumulate before a new/relocated detection target is trusted (the
    ghost-detection guard in Navigator.project_detection). Takes effect on
    the very next detection, no restart needed."""
    applied = nav.set_confirm_n(body.n)
    return {"ok": True, "confirm_n": applied, "nav": nav.state()}


@app.post("/api/nav/standoff")
def nav_standoff(body: StandoffBody):
    """Set how far short of the target the rover stops (mm), clamped to
    [NAV_STANDOFF_MIN_MM, NAV_STANDOFF_MAX_MM].

    This is the SOFT preference used by Navigator.compute_goal — obstacle
    avoidance still wins, so the achieved distance can differ when the
    preferred spot is blocked. Applies to the next goal computation, which in
    AUTO/FOLLOW is the very next cycle."""
    applied = nav.set_standoff(body.mm)
    return {"ok": True, "standoff_mm": applied, "nav": nav.state()}


@app.post("/api/nav/reload_map")
def nav_reload_map():
    """Hot-reload an edited map.json / config.json (refused mid-navigation)."""
    ok, msg = nav.reload_files()
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg, "map": nav.map_payload() if ok else None},
                        status_code=code)


@app.post("/api/nav/set_start")
def nav_set_start(body: SetStartBody):
    """Change the configured start cell (map.json rover_start) and persist it.
    Returns the refreshed map payload so the UI can redraw the start marker."""
    ok, msg = nav.set_start_cell(body.x, body.z)
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg, "map": nav.map_payload() if ok else None},
                        status_code=code)


@app.post("/api/nav/reset_pose")
def nav_reset_pose(body: ResetPoseBody):
    ok = nav.reset_pose(body.x, body.z)
    if not ok:
        return JSONResponse({"ok": False, "error": "no T265 pose available yet"},
                            status_code=409)
    return {"ok": True, "nav": nav.state()}


@app.post("/api/nav/jog")
def nav_jog(body: JogBody):
    """Hold-to-move: the UI calls this every ~200 ms while a drive button is
    held; the rover's dead-man stops it when the calls cease."""
    if body.x not in (-1, 0, 1) or body.z not in (-1, 0, 1):
        return JSONResponse({"ok": False, "error": "x/z must be -1, 0 or 1"}, status_code=422)
    ok, msg = nav.jog(body.x, body.z)
    code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": msg}, status_code=code)


@app.post("/api/nav/jog_stop")
def nav_jog_stop():
    nav.jog_stop()
    return {"ok": True}


@app.post("/api/nav/nudge")
def nav_nudge(body: NudgeBody):
    if body.axis not in ("x", "z"):
        return JSONResponse({"ok": False, "error": "axis must be 'x' or 'z'"}, status_code=422)
    if abs(body.mm) > 1000:
        return JSONResponse({"ok": False, "error": "nudge capped at 1000 mm"}, status_code=422)
    if nav.rover.is_busy():
        return JSONResponse({"ok": False, "error": "rover is busy"}, status_code=409)
    res = nav.rover.move_axis(body.axis, body.mm, units="mm")
    return {"ok": bool(res), "reason": res.reason if res else "no result"}


# ---- frontend -------------------------------------------------------------
@app.get("/")
def index():
    """Operator console (map + navigation controls + sensor thumbnails)."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/audience")
def audience():
    """Audience display (large BEV map with rover/target + sensor feeds)."""
    return FileResponse(os.path.join(FRONTEND_DIR, "audience.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
