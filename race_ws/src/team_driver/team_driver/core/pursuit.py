"""Pure pursuit: steer along the arc that reaches a point ahead on the path.

Chosen over Stanley because the lookahead distance is one intuitive knob -
too short and the car saws at the wheel, too long and it cuts every apex -
and because it is forgiving of the small discontinuities an optimiser-
generated line can have.

The geometry: with the target at straight-line distance `d` from the car and
lateral offset `y` in the car's frame, the arc through both has curvature
2y/d^2, and the steering angle producing it on an Ackermann car of wheelbase
L is atan(2*L*y/d^2). For a car sitting anywhere on a circle of radius R that
evaluates to exactly atan(L/R) whatever the lookahead - which is what
`test_on_a_circle_the_steering_matches_the_closed_form` pins, and why this
module needs no tuning to be correct, only to be fast.
"""

import math
from dataclasses import dataclass

import numpy as np

from .track import Centreline
from .vehicle import Vehicle

_EPS = 1e-9


@dataclass
class PursuitResult:
    steer_rad: float
    target_xy: np.ndarray
    lookahead: float


class PurePursuit:
    """Args:
        k_lookahead: extra lookahead per m/s. Scaling lookahead with speed is
            the standard refinement: a fixed one is either twitchy when fast
            or lazy when slow.
        min_lookahead / max_lookahead: bounds, in metres.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        k_lookahead: float = 0.35,
        min_lookahead: float = 0.6,
        max_lookahead: float = 3.0,
    ):
        self.vehicle = vehicle
        self.k_lookahead = float(k_lookahead)
        self.min_lookahead = float(min_lookahead)
        self.max_lookahead = float(max_lookahead)

    def lookahead_for(self, speed: float) -> float:
        speed = float(speed) if math.isfinite(float(speed)) else 0.0
        distance = self.min_lookahead + self.k_lookahead * max(speed, 0.0)
        return min(max(distance, self.min_lookahead), self.max_lookahead)

    def step(self, pose_xy, yaw: float, speed: float, path: Centreline) -> PursuitResult:
        lookahead = self.lookahead_for(speed)

        # Walk forward along the path by one lookahead from whatever sample we
        # are closest to. point_at_s wraps, so a lap has no seam.
        index, _ = path.nearest(pose_xy)
        target = path.point_at_s(path.s[index] + lookahead)

        dx = float(target[0]) - float(pose_xy[0])
        dy = float(target[1]) - float(pose_xy[1])

        yaw = float(yaw)
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # The chord to the target, not the arc length we walked to find it:
        # the closed form above is in terms of the straight-line distance.
        chord_sq = dx * dx + dy * dy
        curvature = 0.0 if chord_sq < _EPS else 2.0 * lateral / chord_sq

        lock = self.vehicle.max_steer_rad
        steer = math.atan(self.vehicle.wheelbase * curvature)
        return PursuitResult(
            steer_rad=min(max(steer, -lock), lock),
            target_xy=target,
            lookahead=lookahead,
        )
