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
def _plan(obstacles, car_w, car_l, start, goal, clearance, turn_penalty,
          ignore_start_obstacle=False, map_size=None, wall_clearance=0.0):
    mx = car_w / 2.0 + clearance
    mz = car_l / 2.0 + clearance
    boxes = [_inflate(o, mx, mz) for o in obstacles]

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

    xs = sorted({start[0], goal[0]} | {v for b in boxes for v in (b[0], b[2])})
    zs = sorted({start[1], goal[1]} | {v for b in boxes for v in (b[1], b[3])})

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
def _body_gap(point, obstacles, car_w, car_l):
    """Smallest distance from the car body centred at `point` to any RAW
    obstacle (0 when overlapping). Used to keep the reported clearance honest:
    a route can be planned wider than the pose the rover starts from, but the
    figure shown to the operator must not exceed what the rover actually gets."""
    hw, hl = car_w / 2.0, car_l / 2.0
    best = None
    for (x1, z1, x2, z2) in obstacles:
        dx = max(0.0, x1 - (point[0] + hw), (point[0] - hw) - x2)
        dz = max(0.0, z1 - (point[1] + hl), (point[1] - hl) - z2)
        d = math.hypot(dx, dz)
        if best is None or d < best:
            best = d
    return best


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


def _plan_max_clearance(obstacles, car_w, car_l, start, goal, clearance,
                        turn_penalty, map_size, max_clearance, tol,
                        wall_clearance=0.0):
    """Widest-path variant: return the path that maximises the MINIMUM distance
    between the car body and any obstacle, plus the clearance it achieved.

    Feasibility is monotonic in `clearance` -- inflating obstacles only ever
    shrinks free space -- so if a path exists at some clearance, one exists at
    every smaller clearance too. The largest feasible clearance is therefore
    exactly the max-min (bottleneck) clearance, and a binary search finds it.

    The base clearance is always tried first and kept as the fallback, so this
    never fails where a plain single-shot plan would have succeeded, and never
    returns a path with LESS clearance than the configured minimum.

    Returns (path | None, achieved_clearance | None, status).
    """
    # Is the rover's body ALREADY overlapping a raw obstacle? If so the start is
    # a genuine collision, not merely a tight margin, and the relaxation below
    # must NOT fire: refusing to plan is the correct, visible answer, and the
    # caller can still pass ignore_start_obstacle=True deliberately to drive out.
    # Inflating by the half-extents only (no clearance term) is what "body
    # overlaps the obstacle" means, as distinct from "body is inside the buffer".
    start_in_raw = any(_inside(start[0], start[1],
                               _inflate(o, car_w / 2.0, car_l / 2.0))
                       for o in obstacles)
    # Obstacles whose BASE keep-out already contains the start. These keep their
    # base inflation throughout the search (see attempt): they are the ones the
    # rover is parked too close to, and letting them scale with the trial
    # clearance is what previously capped the whole plan.
    _mx0, _mz0 = car_w / 2.0 + clearance, car_l / 2.0 + clearance
    tight_at_base = {o for o in obstacles
                     if _inside(start[0], start[1], _inflate(o, _mx0, _mz0))}

    def attempt(c):
        """Try clearance c, with a PER-OBSTACLE floor so the rover's own pose
        cannot veto the whole search.

        Without this the search stops the moment a raised clearance swallows the
        START pose: a rover parked close to a panel could only ever plan at the
        clearance it happens to be sitting at, so it slid along the obstacle
        instead of first backing away from it. But simply ignoring that obstacle
        (the ignore_start_obstacle flag) removes it from the graph entirely, and
        the route would then drive straight through the panel.

        Instead, obstacles that already contain the start keep their BASE
        inflation while every other obstacle is inflated to c. The panel is
        still solid — the route can never cross it — but standing near it no
        longer caps how wide the rest of the route is planned. The rover's first
        move is simply to leave, which is the reverse-then-traverse behaviour
        wanted.
        """
        if start_in_raw or not tight_at_base:
            return _plan(obstacles, car_w, car_l, start, goal, c, turn_penalty,
                         ignore_start_obstacle=False, map_size=map_size,
                         wall_clearance=wall_clearance)
        # Shrink only the obstacles the start is already too close to, by
        # pre-inflating everything else to c and passing a base clearance of 0.
        mx_hi, mz_hi = c - clearance, c - clearance
        mixed = [o if o in tight_at_base
                 else (o[0] - mx_hi, o[1] - mz_hi, o[2] + mx_hi, o[3] + mz_hi)
                 for o in obstacles]
        return _plan(mixed, car_w, car_l, start, goal, clearance, turn_penalty,
                     ignore_start_obstacle=False, map_size=map_size,
                     wall_clearance=wall_clearance)

    best, status = attempt(clearance)
    if best is None:
        return None, None, status              # infeasible at base -> as before
    best_c = float(clearance)
    lo, hi = float(clearance), float(max_clearance)
    tol = max(float(tol), 1.0)
    # Only search when there is a meaningful band above the base clearance.
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        path, _st = attempt(mid)
        if path is None:
            hi = mid                            # too tight -> shrink the bracket
        else:
            best, best_c, lo = path, mid, mid   # feasible -> keep and push up
    # The relaxation above lets the search exceed the clearance the rover is
    # ALREADY sitting at (that is the point: back away, then travel wide). But
    # the reported figure must describe what the rover really experiences along
    # the whole route, start pose included, or the UI would claim more margin
    # than exists and the e-stop comparison would be misleading. Cap the claim
    # at the start pose's own gap.
    start_gap = _body_gap(start, obstacles, car_w, car_l)
    if start_gap is not None and start_gap < best_c:
        best_c = start_gap
    return best, best_c, "ok"


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
      wall_clearance       extra gap held between the car body and the ARENA
                           WALLS, mm (default 0 = body may sit flush with the
                           edge). The footprint is always kept inside the map
                           regardless; this only adds margin on top.

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
        map_size, max_clearance, tol, wall_clearance=wall_clearance)
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
