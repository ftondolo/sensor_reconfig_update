#!/usr/bin/env python3
"""Standalone live presence detector for the TI IWR6843ISK.

Reuses rover_ui's MmWaveRadar reader (TLV detected-points parser) and runs a
pure point-cloud presence test — the "no visual detection -> ask the radar"
fallback for the detection loop. No camera, no calibration: walk in and out of
the FoV and watch the decision so we can tune the thresholds against the real
sensor.

Radar coordinate convention (TI): y = forward range (m), x = lateral (m),
z = height (m), v = radial Doppler velocity (m/s).

Usage:
    python radar_presence.py                 # run until Ctrl-C
    python radar_presence.py --secs 30       # run 30 s then summarize
    python radar_presence.py --mock          # use synthetic points (no hw)
"""
import argparse
import math
import os
import sys
import time
from collections import deque

# Reuse the verified rover_ui radar reader (only needed by the standalone
# main() below; this file is otherwise imported as a library).
ROVER_UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rover_ui")
if ROVER_UI not in sys.path:
    sys.path.insert(0, ROVER_UI)


# --------------------------------------------------------------------------
# Presence algorithm
# --------------------------------------------------------------------------
class PresenceConfig:
    # Region of interest (a person standing in front of the radar)
    y_min, y_max = 0.3, 6.0          # forward range gate (m)
    x_abs_max = 3.0                  # lateral half-width gate (m)
    z_abs_max = 1.5                  # height gate (m); OOB z is coarse, keep loose
    # Clustering (single-link on x,y)
    eps = 0.40                       # neighbour distance (m)
    min_pts = 3                      # min points to call a cluster
    # Human gate (cluster spatial extent, diagonal of its xy bbox)
    extent_min, extent_max = 0.10, 1.30
    doppler_thresh = 0.05            # |v| above this = "moving" (m/s)
    # Temporal hysteresis (tuned on live walk data 2026-06-02):
    # the person's cluster only forms ~50% of frames, so use an asymmetric
    # window — quick-ish to enter, slow to release — to kill ON/OFF chatter.
    enter_win = 6                    # look back this many frames to enter
    enter_need = 3                   # qualifying frames within enter_win to fire
    release_win = 12                 # clear only after this many frames with none (~1.2s)


def _filter_roi(points, c):
    out = []
    for p in points:
        x, y, z = p["x"], p["y"], p["z"]
        if c.y_min < y < c.y_max and abs(x) < c.x_abs_max and abs(z) < c.z_abs_max:
            out.append(p)
    return out


def _cluster(points, eps):
    """Greedy single-link clustering on (x, y). Fine for the handful of OOB points."""
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


def _cluster_stats(cl):
    xs = [p["x"] for p in cl]
    ys = [p["y"] for p in cl]
    vs = [abs(p["v"]) for p in cl]
    xc, yc = sum(xs) / len(xs), sum(ys) / len(ys)
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    rng = math.hypot(xc, yc)
    az = math.degrees(math.atan2(xc, yc))  # 0 = straight ahead, + = right
    return {
        "n": len(cl), "range": rng, "az": az, "extent": extent,
        "max_v": max(vs), "xc": xc, "yc": yc,
    }


def evaluate_frame(points, c):
    """Return (instant_qualifies, best_cluster_stats_or_None, n_roi)."""
    roi = _filter_roi(points, c)
    if len(roi) < c.min_pts:
        return False, None, len(roi)
    clusters = _cluster(roi, c.eps)
    best, qualifies = None, False
    for cl in clusters:
        if len(cl) < c.min_pts:
            continue
        st = _cluster_stats(cl)
        if c.extent_min <= st["extent"] <= c.extent_max:
            if best is None or st["n"] > best["n"]:
                best = st
                qualifies = True
    # If nothing passed the extent gate, still surface the densest cluster for tuning.
    if best is None:
        dense = max(clusters, key=len)
        if len(dense) >= c.min_pts:
            best = _cluster_stats(dense)
    return qualifies, best, len(roi)


def confidence(win, best, c):
    """Crude 0..1 confidence from recent qualify rate + density + motion."""
    recent = list(win)[-c.enter_win:]
    rate = sum(recent) / max(1, len(recent))
    dens = min(1.0, (best["n"] / 8.0)) if best else 0.0
    motion = 1.0 if (best and best["max_v"] > c.doppler_thresh) else 0.0
    return round(0.5 * rate + 0.35 * dens + 0.15 * motion, 2)


# --------------------------------------------------------------------------
# Live loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=0.0, help="run duration (0 = until Ctrl-C)")
    ap.add_argument("--mock", action="store_true", help="use synthetic radar points")
    args = ap.parse_args()

    from backend.sensors.mmwave import MmWaveRadar
    from backend import config as rcfg

    c = PresenceConfig()
    print(f"[cfg] CLI={rcfg.MMWAVE_CLI_PORT} DATA={rcfg.MMWAVE_DATA_PORT} "
          f"cfg={os.path.basename(rcfg.MMWAVE_CONFIG_FILE)}")
    print(f"[roi] {c.y_min}<y<{c.y_max}  |x|<{c.x_abs_max}  |z|<{c.z_abs_max}  "
          f"eps={c.eps} min_pts={c.min_pts} extent[{c.extent_min},{c.extent_max}] "
          f"enter={c.enter_need}/{c.enter_win} release={c.release_win}")

    radar = MmWaveRadar(allow_mock=args.mock)
    radar.start()

    win = deque(maxlen=c.release_win)
    present = False
    last_seq = 0
    t0 = time.time()
    n_frames = 0
    n_present = 0
    first_data = None
    last_data_t = None
    last_present = None
    try:
        while True:
            if args.secs and time.time() - t0 > args.secs:
                break
            val, seq = radar.latest.wait_for_next(last_seq, timeout=1.0)
            if val is None:
                st = radar.status()
                stall = time.time() - (last_data_t or t0)
                print(f"[{time.time()-t0:5.1f}s] no-frame status={st['status']} "
                      f"stall={stall:.1f}s {st['detail']} {st['error']}", flush=True)
                if st["status"] == "error" or stall > 5.0:
                    print("  !! radar not delivering frames (disconnect/stall) — aborting",
                          flush=True)
                    break
                continue
            last_seq = seq
            n_frames += 1
            last_data_t = time.time()
            if first_data is None:
                first_data = time.time() - t0
            pts = val["points"]
            qual, best, n_roi = evaluate_frame(pts, c)
            win.append(qual)
            # asymmetric hysteresis: enter on enter_need within enter_win,
            # release only after release_win frames with no qualifier.
            if not present and sum(list(win)[-c.enter_win:]) >= c.enter_need:
                present = True
            elif present and sum(win) == 0:
                present = False
            if present:
                n_present += 1
            conf = confidence(win, best, c)

            if present and not last_present:
                print(f"  >>> PERSON ENTERED  (t={time.time()-t0:.1f}s)")
            if not present and last_present:
                print(f"  <<< scene clear     (t={time.time()-t0:.1f}s)")
            last_present = present

            flag = "PERSON" if present else "  -   "
            if best:
                bstr = (f"clust n={best['n']:2d} R={best['range']:4.2f}m "
                        f"az={best['az']:+5.1f} ext={best['extent']:4.2f} "
                        f"vmax={best['max_v']:4.2f}")
            else:
                bstr = "clust none"
            print(f"[{time.time()-t0:5.1f}s] {flag} conf={conf:.2f} "
                  f"pts={val['num']:2d} roi={n_roi:2d} {bstr}  q={int(qual)}", flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        radar.stop() if hasattr(radar, "stop") else radar._stop.set()
        radar.close()

    dur = time.time() - t0
    print("\n===== summary =====")
    print(f"duration       : {dur:.1f}s")
    print(f"radar frames   : {n_frames}  ({n_frames/dur:.1f} fps)" if dur else "")
    print(f"first data at  : {first_data}")
    print(f"frames PRESENT : {n_present}  ({100*n_present/max(1,n_frames):.0f}% of frames)")
    print(f"final status   : {radar.status()}")


if __name__ == "__main__":
    main()
