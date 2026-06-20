"""CLI for quick manual testing of the T265 rover service.

Examples
--------
    # right 500 mm, then forward 200 mm (-z is forward):
    python -m purely_control --move x:500mm --move z:-200mm

    # a single forward 0.3 m move, dry-run (mock, no hardware/ROS needed):
    python -m purely_control --move forward:0.3 --mock

A move spec is ``AXIS:DISTANCE`` where DISTANCE may carry a unit suffix
(``mm``/``cm``/``m``, default ``m``): ``x:500mm``, ``z:-0.2``, ``forward:30cm``.
Moves run in the order given; the program waits for each to finish.
"""
import argparse
import re
import sys

from .t265_rover import Config, T265RoverService

_SPEC = re.compile(r"^\s*([a-zA-Z+]+)\s*:\s*(-?\d*\.?\d+)\s*(mm|cm|m)?\s*$")


def _parse_move(spec):
    m = _SPEC.match(spec)
    if not m:
        raise argparse.ArgumentTypeError(
            f"bad move {spec!r}; expected AXIS:DISTANCE[unit] e.g. x:500mm")
    axis, dist, unit = m.group(1), float(m.group(2)), m.group(3) or "m"
    return axis, dist, unit


def main(argv=None):
    p = argparse.ArgumentParser(prog="purely_control", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--move", action="append", type=_parse_move, default=[], metavar="AXIS:DIST",
                   help="a move to run, e.g. x:500mm or z:-0.2 (repeatable, run in order)")
    p.add_argument("--mock", action="store_true",
                   help="force mock mode (no T265/ROS; integrates commanded velocity)")
    p.add_argument("--cmd-vel-topic", default=Config.CMD_VEL_TOPIC)
    p.add_argument("--max-linear", type=float, default=Config.MAX_LINEAR,
                   help="m/s translation speed cap")
    p.add_argument("--pos-tol", type=float, default=Config.POS_TOL,
                   help="m arrival tolerance")
    args = p.parse_args(argv)

    if not args.move:
        p.error("give at least one --move")

    cfg = Config(CMD_VEL_TOPIC=args.cmd_vel_topic, MAX_LINEAR=args.max_linear,
                 POS_TOL=args.pos_tol)
    # --mock forces mock by hiding hardware; otherwise real with mock fallback.
    if args.mock:
        import purely_control.t265_rover as mod
        mod.rs = None
        mod.rospy = None

    rc = 0
    with T265RoverService(config=cfg) as rover:
        print(f"[cli] {'MOCK' if rover.using_mock else 'LIVE'}  "
              f"ros_connected={rover.ros_connected}  topic={cfg.CMD_VEL_TOPIC}")
        for axis, dist, unit in args.move:
            print(f"[cli] move {axis}:{dist}{unit} ...", flush=True)
            res = rover.move_axis(axis, dist, units=unit)
            print(f"[cli]   -> {res}")
            if not res.ok:
                rc = 1
                break
    return rc


if __name__ == "__main__":
    sys.exit(main())
