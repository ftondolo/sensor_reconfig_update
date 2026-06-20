"""purely_control — drive the rover exact distances using only the T265.

See :mod:`purely_control.t265_rover`. Typical use::

    from purely_control import T265RoverService

    with T265RoverService() as rover:
        rover.move_axis("x", 500, units="mm")   # +x (right) 500 mm
        rover.move_axis("z", -200, units="mm")  # -z (forward) 200 mm
"""
from .t265_rover import Config, MoveResult, T265RoverService

__all__ = ["T265RoverService", "Config", "MoveResult"]
