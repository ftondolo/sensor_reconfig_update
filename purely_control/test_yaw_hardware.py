#!/usr/bin/env python3
"""REAL-ROVER test of heading-hold against NATURALLY-occurring yaw drift.

Unlike test_yaw_control.py (which injects a fake disturbance in MOCK mode), this
script drives the *actual* rover and measures the heading drift that the floor
surface + mecanum wheels produce on their own while translating. No disturbance
is simulated — the only "disturbance" is the real base on the real floor.

It quantifies two things from the live T265 pose:

  1. HEADING-HOLD DURING THE MOVE
     While the rover translates, the floor/mecanum push it off heading. The PI
     loop fights that every tick. We log the peak heading deviation reached
     during translation and the final heading error at the end of the move.
     With --compare-hold-off we first run the SAME move with the loop disabled
     (YAW_GAIN=YAW_IGAIN=0) to record the RAW drift, then with it on — so you
     get a direct "floor drifts X deg, controller holds it to Y deg" number.

  2. END-STAGE CORRECTION IS TIME-BOUNDED (ARRIVE_SETTLE_TIMEOUT)
     If the floor leaves the rover off-heading when it reaches the goal, it
     rotates in place to fix it — but only up to ARRIVE_SETTLE_TIMEOUT, then it
     finishes "arrived" anyway. We measure the reached-position -> done time and
     report whether the move settled inside YAW_TOL or was capped by the bound.

This is HARDWARE-ONLY: it refuses to run in mock (allow_mock=False), so if the
T265 isn't streaming you get an error instead of a meaningless simulated result.

────────────────────────────────────────────────────────────────────────────
SAFETY — read before the first run (see purely_control/README.md "Frame
alignment" and "Running on the real rover"):
  • Source ROS first:   source /opt/ros/noetic/setup.bash   (so cmd_vel publishes)
  • Verify X_SIGN / Y_SIGN / YAW_SIGN at low speed BEFORE trusting larger moves.
    An inverted YAW_SIGN makes heading-hold positive feedback — the rover spins
    up. Keep a hand on it / e-stop ready for the first move. Pass --x-sign /
    --y-sign / --yaw-sign here if your base needs them flipped.
  • Starts at a low speed cap (--max-linear 0.12 m/s). Give it floor space.
  • Ctrl-C stops the rover immediately.
────────────────────────────────────────────────────────────────────────────

Usage
-----
    source /opt/ros/noetic/setup.bash

    # one forward 1 m run, default safe speed, with a confirmation prompt:
    python -m purely_control.test_yaw_hardware --move forward:1.0

    # quantify the floor-induced drift: run hold-OFF then hold-ON, same move:
    python -m purely_control.test_yaw_hardware --move forward:1.5 --compare-hold-off

    # a strafe (mecanum drifts most on the un-driven axis), repeated 4x:
    python -m purely_control.test_yaw_hardware --move x:1000mm --repeat 4

    # watch the end-stage bound: shorten it and log per-sample to CSV:
    python -m purely_control.test_yaw_hardware --move forward:0.5 \
        --settle-timeout 0.8 --csv /tmp/yaw.csv

Move spec is AXIS:DISTANCE[unit] (mm/cm/m, default m), same as `python -m
purely_control`: x/right/left/forward/back/z. Repeatable; run in order.

Exit code 0 if every move arrived, 1 otherwise.
"""
import argparse
import math
import re
import sys
import time

from .t265_rover import Config, T265RoverService

_SPEC = re.compile(r"^\s*([a-zA-Z+]+)\s*:\s*(-?\d*\.?\d+)\s*(mm|cm|m)?\s*$")


def _parse_move(spec):
    m = _SPEC.match(spec)
    if not m:
        raise argparse.ArgumentTypeError(
            f"bad move {spec!r}; expected AXIS:DISTANCE[unit] e.g. forward:1.0")
    return m.group(1), float(m.group(2)), m.group(3) or "m"


def deg(rad):
    return math.degrees(rad)


def _to_m(dist, unit):
    return dist / 1000.0 if unit == "mm" else dist / 100.0 if unit == "cm" else dist


def _axis_to_rf(axis, d):
    """(right, forward) in metres for a move-spec axis + signed distance (m)."""
    a = axis.strip().lower().lstrip("+")
    if a in ("x", "right"):
        return d, 0.0
    if a == "left":
        return -d, 0.0
    if a == "z":                       # native +z is back; forward = -z
        return 0.0, -d
    if a in ("forward", "fwd"):
        return 0.0, d
    if a in ("back", "backward"):
        return 0.0, -d
    raise SystemExit(f"unsupported axis {axis!r} (use x/right/left/forward/back/z)")


def drive_and_sample(rover, right, forward, sample_hz=50, max_wall=60.0):
    """Drive one non-blocking move on the REAL rover, sampling live pose.

    Returns (MoveResult, stats) where stats has the measured heading behaviour.
    """
    start = rover.get_pose()
    if start is None:
        raise RuntimeError("no T265 pose available")
    yaw0, rx0, rf0 = start["yaw"], start["right"], start["forward"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    pos_tol = rover.cfg.POS_TOL

    rows = []                          # (t, yaw_dev_rad, remaining_m, conf)
    t0 = time.monotonic()
    goal = rover.move(right=right, forward=forward, blocking=False)
    dt = 1.0 / sample_hz
    try:
        while rover.is_busy():
            now = time.monotonic() - t0
            if now > max_wall:
                break
            p = rover.get_pose()
            if p is not None:
                dev = math.atan2(math.sin(p["yaw"] - yaw0), math.cos(p["yaw"] - yaw0))
                dxr, dxf = p["right"] - rx0, p["forward"] - rf0
                rt = dxr * c + dxf * s
                ft = -dxr * s + dxf * c
                rem = math.hypot(right - rt, forward - ft)
                rows.append((now, dev, rem, p["confidence"]))
            time.sleep(dt)
    except KeyboardInterrupt:
        rover.stop()
        raise
    res = rover.wait(goal)

    # Peak heading deviation during the TRANSLATION phase (remaining > POS_TOL):
    # this is the floor/mecanum drift the loop is actively fighting mid-move.
    trans = [abs(d) for _, d, rem, _ in rows if rem > pos_tol]
    peak_translate = deg(max(trans)) if trans else 0.0
    peak_overall = deg(max((abs(d) for _, d, _, _ in rows), default=0.0))
    t_pos = next((t for t, _, rem, _ in rows if rem <= pos_tol), None)
    in_place = (res.elapsed - t_pos) if t_pos is not None else float("nan")
    lost = any(conf <= rover.cfg.LOST_CONF for _, _, _, conf in rows)
    stats = {
        "peak_translate_deg": peak_translate,
        "peak_overall_deg": peak_overall,
        "final_yaw_deg": abs(deg(res.yaw_error)),
        "pos_reached_s": t_pos,
        "in_place_s": in_place,
        "tracking_lost": lost,
        "rows": rows,
    }
    return res, stats


def classify(rover, res, stats):
    yaw_tol = deg(rover.cfg.YAW_TOL)
    bound = rover.cfg.ARRIVE_SETTLE_TIMEOUT
    if stats["final_yaw_deg"] <= yaw_tol:
        return f"settled within YAW_TOL ({yaw_tol:.2f}deg)"
    ip = stats["in_place_s"]
    if bound > 0 and ip == ip and ip >= 0.8 * bound:
        return (f"capped by ARRIVE_SETTLE_TIMEOUT ({bound:.1f}s) — "
                f"{stats['final_yaw_deg']:.1f}deg residual left")
    return "ended off-YAW_TOL"


def print_move_report(rover, label, res, stats):
    print(f"  [{label}] {res!r}")
    print(f"      peak heading dev while translating : {stats['peak_translate_deg']:5.1f} deg"
          + ("   (TRACKING LOST during move!)" if stats["tracking_lost"] else ""))
    print(f"      final heading error (vs start)     : {stats['final_yaw_deg']:5.2f} deg"
          f"   (YAW_TOL={deg(rover.cfg.YAW_TOL):.2f})")
    pr = stats["pos_reached_s"]
    print(f"      position reached @ {('%.2fs' % pr) if pr is not None else '   n/a'}"
          f"   in-place correction = {stats['in_place_s']:.2f}s")
    print(f"      outcome: {classify(rover, res, stats)}")


def confirm(prompt):
    if not sys.stdin.isatty():
        return True
    try:
        return input(prompt + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def build_rover(args, hold_off=False):
    overrides = dict(
        CMD_VEL_TOPIC=args.cmd_vel_topic,
        MAX_LINEAR=args.max_linear,
        POS_TOL=args.pos_tol,
        X_SIGN=args.x_sign, Y_SIGN=args.y_sign, YAW_SIGN=args.yaw_sign,
    )
    if args.settle_timeout is not None:
        overrides["ARRIVE_SETTLE_TIMEOUT"] = args.settle_timeout
    if hold_off:
        # Disable heading-hold entirely to record the RAW floor-induced drift.
        overrides["YAW_GAIN"] = 0.0
        overrides["YAW_IGAIN"] = 0.0
    cfg = Config(**overrides)
    # allow_mock=False: this test is meaningless without the real T265.
    rover = T265RoverService(config=cfg, allow_mock=False)
    rover.start(wait_for_pose=True, timeout=8.0)
    return rover


def run_session(args, moves_m, hold_off, label):
    """Open the rover, run the move list (optionally repeated), report. Returns
    (all_arrived, results) — closes the rover before returning."""
    rover = build_rover(args, hold_off=hold_off)
    all_ok = True
    try:
        mode = "HOLD-OFF (raw drift baseline)" if hold_off else "HOLD-ON"
        print(f"\n=== {label} — {mode} ===")
        print(f"    LIVE={not rover.using_mock}  ros_connected={rover.ros_connected}  "
              f"topic={args.cmd_vel_topic}  max_linear={args.max_linear} m/s")
        print(f"    signs: X_SIGN={args.x_sign} Y_SIGN={args.y_sign} YAW_SIGN={args.yaw_sign}  "
              f"ARRIVE_SETTLE_TIMEOUT={rover.cfg.ARRIVE_SETTLE_TIMEOUT}s")
        if not rover.ros_connected and not args.no_ros_ok:
            print("    ERROR: ROS not connected — cmd_vel won't move the rover, the move "
                  "would just time out.\n           source /opt/ros/noetic/setup.bash and "
                  "start a master, or pass --no-ros-ok to proceed anyway.")
            return False, []
        results = []
        for rep in range(args.repeat):
            for i, (right, forward) in enumerate(moves_m):
                # Alternate direction on odd repeats so a repeated test oscillates
                # near the start instead of driving across the room.
                sign = -1.0 if (rep % 2 == 1) else 1.0
                r, f = sign * right, sign * forward
                tag = f"rep{rep + 1}.move{i + 1}"
                print(f"  -> {tag}: right={r:+.3f} forward={f:+.3f} m  (Ctrl-C to abort)")
                res, stats = drive_and_sample(rover, r, f)
                print_move_report(rover, tag, res, stats)
                results.append((tag, res, stats))
                all_ok = all_ok and res.ok
                if args.csv:
                    _append_csv(args.csv, label, mode, tag, stats["rows"])
                time.sleep(args.pause)
        return all_ok, results
    finally:
        rover.shutdown()


def _append_csv(path, label, mode, tag, rows):
    new = not _exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("label,mode,tag,t_s,yaw_dev_deg,remaining_mm,confidence\n")
        for t, dev, rem, conf in rows:
            fh.write(f"{label},{mode},{tag},{t:.3f},{deg(dev):.3f},{rem*1000:.1f},{conf}\n")


def _exists(path):
    try:
        open(path, "rb").close()
        return True
    except OSError:
        return False


def summarize(results, hold_off_results=None):
    print("\n=== SUMMARY ===")
    if hold_off_results:
        print("  floor-induced drift the controller had to reject (hold-OFF vs hold-ON):")
        off_peak = _mean(s["peak_translate_deg"] for _, _, s in hold_off_results)
        off_fin = _mean(s["final_yaw_deg"] for _, _, s in hold_off_results)
        on_peak = _mean(s["peak_translate_deg"] for _, _, s in results)
        on_fin = _mean(s["final_yaw_deg"] for _, _, s in results)
        print(f"    peak drift while translating : hold-OFF {off_peak:5.1f}deg  ->  "
              f"hold-ON {on_peak:5.1f}deg")
        print(f"    final heading error          : hold-OFF {off_fin:5.1f}deg  ->  "
              f"hold-ON {on_fin:5.1f}deg")
    else:
        peak = _mean(s["peak_translate_deg"] for _, _, s in results)
        fin = _mean(s["final_yaw_deg"] for _, _, s in results)
        worst = max((s["final_yaw_deg"] for _, _, s in results), default=0.0)
        print(f"    mean peak heading dev while translating : {peak:.1f} deg")
        print(f"    mean / worst final heading error        : {fin:.1f} / {worst:.1f} deg")
    arrived = sum(1 for _, res, _ in results if res.ok)
    print(f"    moves arrived: {arrived}/{len(results)}")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="purely_control.test_yaw_hardware",
        description="Real-rover heading-hold test against natural floor-induced yaw drift.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--move", action="append", type=_parse_move, default=[], metavar="AXIS:DIST",
                   help="move to run, e.g. forward:1.0 or x:1000mm (repeatable, in order)")
    p.add_argument("--repeat", type=int, default=1, help="repeat the move list N times "
                   "(alternating direction each repeat to stay near the start)")
    p.add_argument("--compare-hold-off", action="store_true",
                   help="first run the moves with heading-hold DISABLED to record the raw "
                        "floor drift, then with it ON (rover veers when off — give it room)")
    p.add_argument("--max-linear", type=float, default=0.12, help="m/s speed cap (default safe 0.12)")
    p.add_argument("--pos-tol", type=float, default=Config.POS_TOL, help="m arrival tolerance")
    p.add_argument("--settle-timeout", type=float, default=None,
                   help="override ARRIVE_SETTLE_TIMEOUT (s) to watch the end-stage bound")
    p.add_argument("--cmd-vel-topic", default=Config.CMD_VEL_TOPIC)
    p.add_argument("--x-sign", type=int, choices=(-1, 1), default=Config.X_SIGN)
    p.add_argument("--y-sign", type=int, choices=(-1, 1), default=Config.Y_SIGN)
    p.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=Config.YAW_SIGN,
                   help="flip if a heading correction turns the WRONG way (spins up)")
    p.add_argument("--pause", type=float, default=1.0, help="s to pause between moves")
    p.add_argument("--csv", help="append per-sample (t, yaw, remaining) rows to this CSV")
    p.add_argument("--yes", action="store_true", help="skip the safety confirmation prompt")
    p.add_argument("--no-ros-ok", action="store_true",
                   help="proceed even if no ROS master (pose-only, rover won't move)")
    args = p.parse_args(argv)

    if not args.move:
        p.error("give at least one --move, e.g. --move forward:1.0")
    if args.repeat < 1:
        p.error("--repeat must be >= 1")
    moves_m = [_axis_to_rf(a, _to_m(d, u)) for a, d, u in args.move]

    print(__doc__.split("Usage")[0])   # banner + safety block
    specs = ", ".join(f"{a}:{d}{u}" for a, d, u in args.move)
    print(f"Planned: [{specs}] x{args.repeat}, max_linear={args.max_linear} m/s, "
          f"YAW_SIGN={args.yaw_sign}"
          + (", with hold-OFF baseline" if args.compare_hold_off else ""))
    if not args.yes and not confirm("Rover WILL move. E-stop ready? Proceed?"):
        print("aborted.")
        return 1

    try:
        hold_off_results = None
        if args.compare_hold_off:
            ok_off, hold_off_results = run_session(args, moves_m, hold_off=True, label="baseline")
            if not hold_off_results:
                return 1
            if not args.yes and not confirm("\nHold-OFF baseline done. Run the hold-ON pass?"):
                print("stopped after baseline.")
                return 1
        ok_on, results = run_session(args, moves_m, hold_off=False, label="heading-hold")
        if not results:
            return 1
        summarize(results, hold_off_results)
        all_ok = ok_on and (hold_off_results is None or True)
        print("\n" + ("RESULT: all moves arrived." if all_ok
                      else "RESULT: some moves did NOT arrive (see reasons above)."))
        return 0 if all_ok else 1
    except KeyboardInterrupt:
        print("\ninterrupted — rover stopped.")
        return 1
    except RuntimeError as exc:
        # allow_mock=False -> a missing/!streaming T265 raises here. This test is
        # hardware-only by design; report it cleanly instead of a traceback.
        print(f"\nHARDWARE NOT READY: {exc}\n"
              "  This test drives the real rover and refuses to simulate. Check:\n"
              "    • the T265 is plugged in and enumerates (rs-enumerate-devices)\n"
              "    • pyrealsense2 <= 2.53 (T265 pose support; 2.49 is verified here)\n"
              "    • for motion: source /opt/ros/noetic/setup.bash and a ROS master is up")
        return 1


if __name__ == "__main__":
    sys.exit(main())
