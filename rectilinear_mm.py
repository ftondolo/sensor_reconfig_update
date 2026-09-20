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

The route is chosen to MAXIMISE ITS MINIMUM DISTANCE to any obstacle rather
than merely to clear them by config.json's `clearance`, accepting a modestly
longer path in exchange for a wider safety margin. Because feasibility is
monotonic in clearance (inflating obstacles only shrinks free space), the
largest clearance that still admits a path is exactly that max-min distance,
and a binary search over `clearance` finds it -- reusing the planner unchanged.
`clearance` remains the floor: the search never returns less, and never fails
where a single-shot plan would have succeeded. plan_rectilinear_path_ex()
additionally reports the clearance actually achieved. See config.json keys
`max_clearance` (cap, default 400 mm) and `clearance_tolerance` (search
resolution, default 10 mm); set max_clearance <= clearance to disable.

Coordinate convention: the car heading always points "up" (forward) within the
map. x is positive to the right; z is positive downward (backward), so moving
"forward (up)" means z decreases = negative. Map and config are entirely in mm.

Footprint rule: every test is on the car's axis-aligned RECTANGLE centred on
the point, never on the point alone. Obstacles are grown by the car's
half-width (x) and half-length (z) -- the Minkowski sum -- so testing the
centre against a grown box is exactly "does the rectangle overlap the box",
and a straight centre segment hitting a grown box is exactly "does the moving
rectangle sweep into the box". footprint_gap() gives the matching distance.

After the max-min search, a final pass prefers routes that stay far from
obstacles EVERYWHERE, not just at the tightest point: each leg costs more the
closer the car's swept rectangle comes to an obstacle than
`preferred_clearance` (weight `clearance_weight`). The start/goal pose only
lowers the inflation of the specific obstacle it is close to
(`endpoint_clearance` is the floor for how close an endpoint may be).

This file has zero third-party dependencies (stdlib only) and can be copied out
and used on its own.
"""
import json
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


def footprint_gap(ax, az, bx, bz, obstacles, car_w, car_l):
    """Clearance between the car rectangle swept from centre (ax, az) to
    (bx, bz) along an AXIS-ALIGNED move (a == b for a single pose) and the
    nearest obstacle box (x1, z1, x2, z2), in mm.

    Measured the same way the planner inflates boxes (per axis, square
    corners), so "footprint_gap(p) < c" <=> "p is inside the obstacle grown by
    the car half-extents + c". Negative = the rectangle overlaps an obstacle.
    Returns +inf when there are no obstacles."""
    hw, hl = car_w / 2.0, car_l / 2.0
    rx1, rx2 = min(ax, bx) - hw, max(ax, bx) + hw
    rz1, rz2 = min(az, bz) - hl, max(az, bz) + hl
    best = float("inf")
    for (x1, z1, x2, z2) in obstacles:
        g = max(x1 - rx2, rx1 - x2, z1 - rz2, rz1 - z2)
        if g < best:
            best = g
    return best


# ----------------- Hanan-grid rectilinear planning -----------------
def _plan(obstacles, car_w, car_l, start, goal, clearance, turn_penalty,
          ignore_start_obstacle=False, map_size=None, wall_clearance=0.0,
          inflation=None, edge_cost=None, extra_xs=(), extra_zs=(),
          midlines=False):
    """Shortest (length + turn_penalty per turn [+ edge_cost]) rectilinear
    path on the Hanan grid of the grown obstacles.

    inflation : optional per-obstacle clearance (mm), overriding `clearance`.
    edge_cost : optional f(a, b) -> multiplier (>= 1) applied to a leg's length.
    extra_xs/zs, midlines : extra candidate grid lines, so a route can run
                away from obstacle edges (e.g. down the middle of a corridor)."""
    infl = inflation if inflation is not None else [clearance] * len(obstacles)
    boxes = [_inflate(o, car_w / 2.0 + e, car_l / 2.0 + e)
             for o, e in zip(obstacles, infl)]

    def free(p):
        return not any(_inside(p[0], p[1], b) for b in boxes)

    # Keep the rover INSIDE the arena. The edges are real walls and the rover
    # has a physical footprint, so testing only its CENTRE against the map
    # rectangle was wrong: a centre exactly on an edge leaves HALF THE CAR
    # outside the room. That let a route hug an edge -- or detour around
    # obstacles through coordinates the rover cannot physically occupy -- while
    # every node still passed the check. Test the FOOTPRINT instead.
    #
    # `wall_clearance` optionally holds the body further off the walls, exactly
    # as `clearance` does for obstacles (0 = flush, the default, which preserves
    # reachability on tight arenas).
    #
    # `slack` accommodates a start or goal that ALREADY protrudes -- e.g. the
    # rover parked in a corner with its centre on the corner point. Refusing to
    # plan from such a pose would strand it, so the route may protrude by as
    # much as the start/goal already does, and never more. When both sit
    # properly inside, slack is 0 and containment is strict.
    half_w = car_w / 2.0 + wall_clearance
    half_l = car_l / 2.0 + wall_clearance

    def _excess(p):
        """How far a footprint centred at p protrudes past the map edge (mm)."""
        if map_size is None:
            return 0.0
        mw, md = map_size
        return max(0.0,
                   half_w - p[0], (p[0] + half_w) - mw,
                   half_l - p[1], (p[1] + half_l) - md)

    slack = max(_excess(start), _excess(goal))
    # The arena itself is never negotiable: an endpoint whose footprint is
    # (partly) OUTSIDE the map is refused -- the caller must bring the rover
    # back inside first. Within the wall_clearance band an endpoint may sit
    # closer to the wall, and the route may then use as much of the band.
    if map_size is not None and slack > wall_clearance + 1e-6:
        return None, ("start_blocked" if _excess(start) >= _excess(goal) else "goal_blocked")

    def in_map(p):
        if map_size is None:
            return True
        return _excess(p) <= slack + 1e-6

    # Operator override: the rover's OWN current cell is allowed to sit inside
    # an obstacle/clearance zone (e.g. it drifted in, or the zone was added
    # after the rover was placed there) without refusing to plan a way out.
    # Drop any box that contains `start` entirely for this plan -- the rover's
    # position is ignored, but every other obstacle (including that same one,
    # anywhere else on the map) still fully blocks the path.
    if ignore_start_obstacle and not free(start):
        boxes = [b for b in boxes if not _inside(start[0], start[1], b)]

    if not free(start):
        return None, "start_blocked"
    if not free(goal):
        return None, "goal_blocked"

    xs = sorted({start[0], goal[0]} | {v for b in boxes for v in (b[0], b[2])}
                | set(extra_xs))
    zs = sorted({start[1], goal[1]} | {v for b in boxes for v in (b[1], b[3])}
                | set(extra_zs))
    if midlines:
        xs = sorted(set(xs) | {0.5 * (a + b) for a, b in zip(xs, xs[1:])})
        zs = sorted(set(zs) | {0.5 * (a + b) for a, b in zip(zs, zs[1:])})

    idx, nodes = {}, []
    for x in xs:
        for z in zs:
            if free((x, z)) and in_map((x, z)):
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
                if edge_cost is not None:
                    w *= edge_cost(a, b)
                adj[idx[a]].append((idx[b], w, 1))
                adj[idx[b]].append((idx[a], w, 1))
    for x in xs:
        for j in range(len(zs) - 1):
            a, b = (x, zs[j]), (x, zs[j + 1])
            if a in idx and b in idx and not hits(a[0], a[1], b[0], b[1]):
                w = zs[j + 1] - zs[j]
                if edge_cost is not None:
                    w *= edge_cost(a, b)
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


# ----------------- obstacle schema -----------------
# Two schemas are accepted, detected by CONTENT rather than by a version field.
# Misreading one as the other would shift every panel by half its depth --
# silently, and in the direction that causes collisions -- so a forgotten
# version bump must not be able to cause it.
#
#   MIDLINE (current): config.json carries the shared size, and each obstacle is
#       located by the point on its -x edge that lies on its z midline:
#           config.json : "obstacle_size": {"w": 1350, "h": 40}
#                         "obstacle_pad":  {"x": 0, "z": 155}
#           map.json    : {"x": 182, "z": 4398, "tags": [...]}
#       Detected by config.json carrying "obstacle_size".
#
#   LEGACY (min-corner): each obstacle carries its own w/h and (x, z) is its
#       minimum corner. Detected by the obstacle carrying "w"/"h".
#
# `obstacle_pad` separates PHYSICAL EXTENT from SAFETY POLICY. The panels are
# thin walls (~40 mm) standing in stabiliser feet that reach further in z, so
# the rover collides with the feet long before the wall. The pad is that foot
# overhang: real, fixed geometry. `clearance` remains the tunable margin held on
# top. Folding the feet into `clearance` instead would work in z but would also
# inflate every panel END in x, where the feet do not reach, needlessly
# narrowing every corridor -- and it would make the reported clearance figure
# meaningless, since it would silently include the foot depth.


def obstacle_boxes(map_dict, config_dict=None, kind="collision"):
    """Obstacles as canonical MIN-CORNER dicts: [{"x", "z", "w", "h"}, ...].

    kind="collision" -> the wall plus the stabiliser-foot pad: what the rover
                        must not enter. Used for planning, inflation, e-stop.
    kind="panel"     -> the bare wall: where the surface (and therefore the
                        tags) physically is. Used for tag geometry, and for any
                        perception that must agree with what the camera sees.

    Order matches map_dict["obstacles"], so callers can zip against it.
    """
    cfg = config_dict or {}
    size = cfg.get("obstacle_size")
    pad = cfg.get("obstacle_pad") or {}
    px = float(pad.get("x", 0.0)) if kind == "collision" else 0.0
    pz = float(pad.get("z", 0.0)) if kind == "collision" else 0.0
    out = []
    for ob in map_dict.get("obstacles", []):
        if size is not None and "w" not in ob and "h" not in ob:
            # MIDLINE schema: (x, z) is the -x edge on the z midline.
            w, h = float(size["w"]), float(size["h"])
            x1 = float(ob["x"]) - px
            z1 = float(ob["z"]) - h / 2.0 - pz
            w_out, h_out = w + 2 * px, h + 2 * pz
        else:
            # LEGACY schema: (x, z) is the minimum corner and w/h are per-obstacle.
            # No pad is applied: a legacy box already described the full
            # footprint, so padding it again would double-count the feet.
            x1, z1 = float(ob["x"]), float(ob["z"])
            w_out, h_out = float(ob["w"]), float(ob["h"])
        out.append({"x": x1, "z": z1, "w": w_out, "h": h_out})
    return out


def obstacle_rects(map_dict, config_dict=None, kind="collision"):
    """Same as obstacle_boxes but as (x1, z1, x2, z2) tuples."""
    return [(o["x"], o["z"], o["x"] + o["w"], o["z"] + o["h"])
            for o in obstacle_boxes(map_dict, config_dict, kind)]


def describe_obstacle_schema(map_dict, config_dict=None):
    """One-line summary of how the obstacles were interpreted, for startup logs.
    Printing this makes a schema misread visible instead of silent."""
    obs = map_dict.get("obstacles", [])
    if not obs:
        return "no obstacles"
    cfg = config_dict or {}
    midline = cfg.get("obstacle_size") is not None and "w" not in obs[0]
    r = obstacle_rects(map_dict, config_dict, "collision")[0]
    return ("%s schema, %d obstacles; #0 keep-out x %.0f..%.0f z %.0f..%.0f"
            % ("MIDLINE" if midline else "LEGACY min-corner", len(obs),
               r[0], r[2], r[1], r[3]))


# ----------------- clearance maximisation -----------------
def _drop_boxes_containing(obstacles, car_w, car_l, point, clearance):
    """Obstacles whose inflated keep-out does NOT contain `point`.

    Used for the ignore_start_obstacle override. It is evaluated ONCE at the
    BASE clearance and the result reused for every clearance the search tries,
    so raising the search clearance can never cause an extra obstacle to be
    discarded (which would let a route pass straight through a real obstacle).
    """
    mx = car_w / 2.0 + clearance
    mz = car_l / 2.0 + clearance
    return [o for o in obstacles
            if not _inside(point[0], point[1], _inflate(o, mx, mz))]


def _endpoint_caps(obstacles, car_w, car_l, points):
    """Per obstacle: how much it can be inflated before it swallows one of
    `points` (the start/goal) -- its footprint_gap to the nearest endpoint,
    minus 1 mm so the endpoint stays strictly outside."""
    return [min(footprint_gap(px, pz, px, pz, [o], car_w, car_l) for px, pz in points) - 1.0
            for o in obstacles]


def _route_gap(path, obstacles, car_w, car_l):
    """Smallest footprint_gap along a planned path (its honest clearance)."""
    return min(footprint_gap(a[0], a[1], b[0], b[1], obstacles, car_w, car_l)
               for a, b in zip(path, path[1:])) if len(path) > 1 else \
        footprint_gap(path[0][0], path[0][1], path[0][0], path[0][1], obstacles, car_w, car_l)


def _plan_max_clearance(obstacles, car_w, car_l, start, goal, clearance,
                        turn_penalty, map_size, max_clearance, tol,
                        wall_clearance=0.0, endpoint_clearance=None,
                        preferred_clearance=0.0, clearance_weight=0.0):
    """Clearance-first planning. Returns (path | None, achieved_clearance | None, status).

    1. Endpoints. An obstacle is never inflated past the start or goal pose
       (_endpoint_caps): a pose that sits close to ONE panel only lowers that
       panel's margin, instead of capping the whole route (previously a goal
       near a panel limited every obstacle, and a start near one kept that
       panel at the minimum for the entire route). Endpoints closer than
       `endpoint_clearance` (default: `clearance`) are refused.
    2. Bottleneck. Feasibility is monotonic in clearance, so a binary search
       finds the largest clearance c the tightest point of any route admits
       (floor `clearance`, cap `max_clearance`).
    3. Everywhere else. One final pass at c prefers routes whose swept car
       rectangle stays >= `preferred_clearance` from obstacles: each leg costs
       length x (1 + clearance_weight x shortfall / preferred_clearance), with
       extra grid lines (corridor midlines, obstacles grown to the preferred
       clearance) so the route can centre itself and back away from panels.
    The returned clearance is the smallest footprint_gap along the route."""
    ep_min = clearance if endpoint_clearance is None else float(endpoint_clearance)
    for p, status in ((start, "start_blocked"), (goal, "goal_blocked")):
        if footprint_gap(p[0], p[1], p[0], p[1], obstacles, car_w, car_l) < ep_min:
            return None, None, status
    caps = _endpoint_caps(obstacles, car_w, car_l, (start, goal))

    def attempt(c, **kw):
        return _plan(obstacles, car_w, car_l, start, goal, c, turn_penalty,
                     map_size=map_size, wall_clearance=wall_clearance,
                     inflation=[max(0.0, min(c, cap)) for cap in caps], **kw)

    best, status = attempt(clearance)
    if best is None:
        return None, None, status
    best_c = float(clearance)
    lo, hi = float(clearance), float(max_clearance)
    tol = max(float(tol), 1.0)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        path, _st = attempt(mid)
        if path is None:
            hi = mid
        else:
            best, best_c, lo = path, mid, mid

    pref, weight = float(preferred_clearance), float(clearance_weight)
    if obstacles and pref > 0.0 and weight > 0.0:
        def cost(a, b):
            g = footprint_gap(a[0], a[1], b[0], b[1], obstacles, car_w, car_l)
            return 1.0 + weight * max(0.0, pref - max(g, 0.0)) / pref
        px, pz = car_w / 2.0 + pref, car_l / 2.0 + pref
        ex = [v for o in obstacles for v in (o[0] - px, o[2] + px)]
        ez = [v for o in obstacles for v in (o[1] - pz, o[3] + pz)]
        wide, _st = attempt(best_c, edge_cost=cost, extra_xs=ex, extra_zs=ez,
                            midlines=True)
        if wide is not None:
            best = wide
    achieved = _route_gap(best, obstacles, car_w, car_l) if obstacles else best_c
    return best, achieved, "ok"


# ----------------- public API -----------------
def plan_rectilinear_path(map_file, config_file, start, goal,
                          ignore_start_obstacle=False):
    """Backwards-compatible entry point: returns list[(axis, mm)] or None.

    See plan_rectilinear_path_ex for the achieved-clearance value.
    """
    segs, _clearance = plan_rectilinear_path_ex(
        map_file, config_file, start, goal,
        ignore_start_obstacle=ignore_start_obstacle)
    return segs


def plan_rectilinear_path_ex(map_file, config_file, start, goal,
                             ignore_start_obstacle=False):
    """As plan_rectilinear_path, but returns (segments, achieved_clearance_mm).

    The route maximises its minimum distance to any obstacle (see
    _plan_max_clearance) instead of merely clearing them by the configured
    minimum, trading a little extra path length for a wider safety margin.
    Both are None when no path exists.

    config.json keys (all optional except `clearance`):
      clearance            minimum body-to-obstacle distance, mm -- the floor
      max_clearance        cap for the search, mm (default 400). Set this at or
                           below `clearance` to disable the search entirely and
                           plan exactly as before.
      clearance_tolerance  search resolution, mm (default 10)
      wall_clearance       gap held between the car body and the ARENA WALLS,
                           mm (default 0 = flush). The footprint is always
                           kept inside the map; an endpoint outside it is
                           refused.
      endpoint_clearance   how close the start/goal pose may be to an
                           obstacle, mm (default = clearance)
      preferred_clearance  distance the whole route tries to keep, mm
                           (default 0 = off)
      clearance_weight     how strongly (default 0 = off; ~6 = clearance first)

    ignore_start_obstacle: when True, the rover's own current cell is never the
    reason a plan is refused -- any obstacle/clearance zone containing `start`
    is treated as absent for this call. Everything else (the goal, every other
    obstacle) is still fully enforced.
    """
    m = _load(map_file)
    c = _load(config_file)

    obstacles = obstacle_rects(m, c, "collision")
    car_w = c["car"]["width"]
    car_l = c["car"]["length"]
    clearance = c.get("clearance", 0)
    turn_penalty = c.get("turn_penalty", 0)
    max_clearance = c.get("max_clearance", 400)
    tol = c.get("clearance_tolerance", 10)
    wall_clearance = c.get("wall_clearance", 0)
    endpoint_clearance = c.get("endpoint_clearance", clearance)
    preferred_clearance = c.get("preferred_clearance", 0)
    clearance_weight = c.get("clearance_weight", 0)
    size = m.get("size") or {}
    map_size = ((float(size["width"]), float(size["depth"]))
               if "width" in size and "depth" in size else None)

    start, goal = tuple(start), tuple(goal)
    # Resolve the start-cell override up front, at the BASE clearance, so the
    # obstacle set is identical for every clearance the search below tries.
    if ignore_start_obstacle:
        obstacles = _drop_boxes_containing(obstacles, car_w, car_l,
                                           start, clearance)

    path, achieved, status = _plan_max_clearance(
        obstacles, car_w, car_l, start, goal, clearance, turn_penalty,
        map_size, max_clearance, tol, wall_clearance=wall_clearance,
        endpoint_clearance=endpoint_clearance,
        preferred_clearance=preferred_clearance,
        clearance_weight=clearance_weight)
    if path is None:
        return None, None

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
    return [(a, d) for a, d in merged if d != 0], achieved


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
