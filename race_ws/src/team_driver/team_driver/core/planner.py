"""What to do, given a scan and a pose.

The ROS adapter calls exactly one thing: `Planner.step`. Everything the car
decides happens below this line and none of it imports ROS, which is what
makes it testable in milliseconds rather than in six-minute simulator runs.

Modes
-----
`gap`   reactive, needs nothing but the scan. The permanent fallback.
`line`  follows the optimised racing line against ground-truth pose, with a
        speed profile that brakes for a corner before it can see it. Added in
        a later task; `gap` is what ships until then.

The contract the adapter relies on, and the tests enforce: `step` always
returns finite commands inside [-1, 1], whatever the scan, the pose or the
speed look like. A planner that raises leaves the bridge repeating the last
command it got, which is the car driving on at its last throttle.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .gap import GapConfig, GapFollower
from .speed_controller import SpeedController
from .vehicle import Vehicle


@dataclass
class PlannerConfig:
    gap: GapConfig = field(default_factory=GapConfig)
    # Speed controller gains, exposed here so a sweep can reach them through
    # one ROS parameter file without touching code.
    kp: float = 0.05
    ki: float = 0.15
    integral_limit: float = 0.30
    lift_band: float = 0.5
    brake_gain: float = 0.30


@dataclass
class PlanResult:
    steer_norm: float
    throttle: float
    mode: str
    v_target: float


class Planner:
    def __init__(self, vehicle: Vehicle, config: Optional[PlannerConfig] = None):
        self.vehicle = vehicle
        self.config = config or PlannerConfig()
        self.gap_follower = GapFollower(vehicle, self.config.gap)
        self.speed_controller = SpeedController(
            vehicle,
            kp=self.config.kp,
            ki=self.config.ki,
            integral_limit=self.config.integral_limit,
            lift_band=self.config.lift_band,
            brake_gain=self.config.brake_gain,
        )

    def reset(self) -> None:
        self.speed_controller.reset()

    def step(self, ranges, angles, pose, yaw, speed, dt) -> PlanResult:
        speed = _finite(speed, 0.0)
        dt = _finite(dt, 0.0)

        # A pose we cannot trust is worse than no pose: it puts the car on a
        # line it is not actually on. Gap mode needs neither.
        _ = _valid_pose(pose, yaw)

        result = self.gap_follower.step(ranges, angles)
        mode = "gap"

        steer_norm = self.vehicle.steer_to_norm(result.steer_rad)
        throttle = self.speed_controller.step(result.v_target, speed, dt)

        return PlanResult(
            steer_norm=_clip_unit(steer_norm),
            throttle=_clip_unit(throttle),
            mode=mode,
            v_target=max(0.0, _finite(result.v_target, 0.0)),
        )


def _valid_pose(pose, yaw) -> Optional[Tuple[float, float, float]]:
    """(x, y, yaw) when all three are usable, otherwise None."""
    if pose is None or yaw is None:
        return None
    try:
        x, y = float(pose[0]), float(pose[1])
        heading = float(yaw)
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
        return None
    return x, y, heading


def _finite(value, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _clip_unit(value: float) -> float:
    value = _finite(value, 0.0)
    return -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
