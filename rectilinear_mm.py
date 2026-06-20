"""
plan_rectilinear_path -- rectilinear path planning, returns a list of (axis, mm) segments
=========================================================================================
Inputs:
  map_file    : path to map.json, or an already-loaded dict
  config_file : path to config.json, or an already-loaded dict
  start, goal : (x, z) coordinates of the car CENTER, in mm
Output:
  list[(axis, mm)] -- one tuple per segment:
    axis = 'x'  lateral move;       positive = right, negative = left
    axis = 'z'  forward/back move;  negative = forward, positive = backward
  Returns None when there is no feasible path / start or goal is blocked.

Coordinate convention: the car heading always points "up" (forward) within the
map. x is positive to the right; z is positive downward (backward), so moving
"forward (up)" means z decreases = negative. Map and config are entirely in mm.

This file has zero third-party dependencies (stdlib only) and can be copied out
and used on its own.
"""
import json
import math
import heapq

EPS = 1e-7


# ----------------- geometry -----------------
def _inflate(box, mx, mz):
    x1, z1, x2, z2 = box
    return (x1 - mx, z1 - mz, x2 + mx, z2 + mz)


def _inside(px, pz, b):
    return b[0] + EPS < px < b[2] - EPS and b[1] + EPS < pz < b[3] - EPS


def _seg_hits(ax, az, bx, bz, b):
    x1, z1, x2, z2 = b[0] + EPS, b[1] + EPS, b[2] - EPS, b[3] - EPS
    if x1 >= x2 or z1 >= z2:
        return False
    dx, dz = bx - ax, bz - az
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x1), (dx, x2 - ax), (-dz, az - z1), (dz, z2 - az)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
    return t0 < t1


# ----------------- Hanan-grid rectilinear planning -----------------
def _plan(obstacles, car_w, car_l, start, goal, clearance, turn_penalty):
    mx = car_w / 2.0 + clearance
    mz = car_l / 2.0 + clearance
    boxes = [_inflate(o, mx, mz) for o in obstacles]

    def free(p):
        return not any(_inside(p[0], p[1], b) for b in boxes)

    if not free(start):
        return None, "start_blocked"
    if not free(goal):
        return None, "goal_blocked"

    xs = sorted({start[0], goal[0]} | {v for b in boxes for v in (b[0], b[2])})
    zs = sorted({start[1], goal[1]} | {v for b in boxes for v in (b[1], b[3])})

    idx, nodes = {}, []
    for x in xs:
        for z in zs:
            if free((x, z)):
                idx[(x, z)] = len(nodes)
                nodes.append((x, z))

    adj = [[] for _ in nodes]

    def hits(ax, az, bx, bz):
        return any(_seg_hits(ax, az, bx, bz, b) for b in boxes)

    for z in zs:
        for i in range(len(xs) - 1):
            a, b = (xs[i], z), (xs[i + 1], z)
            if a in idx and b in idx and not hits(a[0], a[1], b[0], b[1]):
                w = xs[i + 1] - xs[i]
                adj[idx[a]].append((idx[b], w, 1))
                adj[idx[b]].append((idx[a], w, 1))
    for x in xs:
        for j in range(len(zs) - 1):
            a, b = (x, zs[j]), (x, zs[j + 1])
            if a in idx and b in idx and not hits(a[0], a[1], b[0], b[1]):
                w = zs[j + 1] - zs[j]
                adj[idx[a]].append((idx[b], w, 2))
                adj[idx[b]].append((idx[a], w, 2))

    src, dst = idx[tuple(start)], idx[tuple(goal)]
    INF = float("inf")
    dist, prev = {(src, 0): 0.0}, {}
    pq = [(0.0, src, 0)]
    while pq:
        d, u, ud = heapq.heappop(pq)
        if d > dist.get((u, ud), INF):
            continue
        if u == dst:
            break
        for v, w, vd in adj[u]:
            turn = turn_penalty if (ud != 0 and ud != vd) else 0.0
            nd = d + w + turn
            st = (v, vd)
            if nd < dist.get(st, INF):
                dist[st] = nd
                prev[st] = (u, ud)
                heapq.heappush(pq, (nd, v, vd))

    best = min((dist.get((dst, dd), INF), dd) for dd in (0, 1, 2))
    if best[0] == INF:
        return None, "no_path"

    st, raw = (dst, best[1]), []
    while st in prev or st[0] == src:
        raw.append(nodes[st[0]])
        if st[0] == src:
            break
        st = prev[st]
    raw.reverse()

    path = [raw[0]]
    for p in raw[1:]:
        if len(path) >= 2:
            a, b = path[-2], path[-1]
            if (a[0] == b[0] == p[0]) or (a[1] == b[1] == p[1]):
                path[-1] = p
                continue
        path.append(p)
    return path, "ok"


def _load(x):
    if isinstance(x, dict):
        return x
    with open(x, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------- public API -----------------
def plan_rectilinear_path(map_file, config_file, start, goal):
    m = _load(map_file)
    c = _load(config_file)

    obstacles = [(o["x"], o["z"], o["x"] + o["w"], o["z"] + o["h"])
                 for o in m.get("obstacles", [])]
    car_w = c["car"]["width"]
    car_l = c["car"]["length"]
    clearance = c.get("clearance", 0)
    turn_penalty = c.get("turn_penalty", 0)

    path, status = _plan(obstacles, car_w, car_l,
                         tuple(start), tuple(goal), clearance, turn_penalty)
    if path is None:
        return None

    pts = [(round(p[0]), round(p[1])) for p in path]
    segs = []
    for (x1, z1), (x2, z2) in zip(pts, pts[1:]):
        if x2 != x1:
            segs.append(("x", x2 - x1))      # right positive / left negative
        elif z2 != z1:
            segs.append(("z", z2 - z1))      # forward negative (up) / backward positive (down)

    merged = []
    for ax, d in segs:
        if merged and merged[-1][0] == ax:
            merged[-1] = (ax, merged[-1][1] + d)
        else:
            merged.append((ax, d))
    return [(a, d) for a, d in merged if d != 0]


# ----------------- self-test -----------------
if __name__ == "__main__":
    sample_map = {
        "format": "pathlab-map", "version": 1, "units": "mm",
        "size": {"width": 10000, "depth": 8000},
        "obstacles": [
            {"x": 2200, "z": 1600, "w": 1600, "h": 2400},
            {"x": 4600, "z": 3000, "w": 1800, "h": 2200},
            {"x": 5200, "z": 800,  "w": 1300, "h": 1600},
            {"x": 7000, "z": 3400, "w": 1500, "h": 1400},
        ],
    }
    sample_config = {
        "format": "pathlab-config", "version": 1, "units": "mm",
        "car": {"width": 800, "length": 1200},
        "clearance": 300, "turn_penalty": 1500,
    }
    start = (600, 7200)     # bottom-left
    goal = (9200, 700)      # top-right

    # equivalently, write to files first and pass the paths
    with open("sample_map.json", "w", encoding="utf-8") as f:
        json.dump(sample_map, f, ensure_ascii=False, indent=2)
    with open("sample_config.json", "w", encoding="utf-8") as f:
        json.dump(sample_config, f, ensure_ascii=False, indent=2)

    segments = plan_rectilinear_path("sample_map.json", "sample_config.json", start, goal)
    print("start", start, " goal", goal)
    print("rectilinear segments (axis, mm):")
    desc = {"x": lambda d: "right" if d > 0 else "left",
            "z": lambda d: "forward" if d < 0 else "backward"}
    total = 0
    for ax, d in segments:
        total += abs(d)
        print(f"  ('{ax}', {d:>6})   # {desc[ax](d)} {abs(d)} mm")
    print(f"{len(segments)} segments, total travel {total} mm")
