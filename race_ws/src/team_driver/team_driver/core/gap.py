"""Follow-the-gap with disparity extension.

The strongest purely reactive approach, and the one that works on a circuit
with no shipped racing line. Three ideas stacked up (docs/04-algorithms.md
4.2):

  1. a safety bubble, so the car can never aim at a point it would clip on the
     way to;
  2. disparity extension, so an opening the LiDAR can see through but the car
     cannot fit through stops looking like an opening;
  3. aim at the middle of the widest gap, not at the deepest beam - the
     deepest beam makes the car twitch between two nearly identical
     directions.

Its ceiling is set by memory it does not have: it brakes for a corner only
once it can see it, and it can see 10 m of a 54 m lap. That is why this is the
fallback rather than the plan. It is also why it has to be reliable: every
failure path in `planner.py` ends here.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .lidar import clean, disparity_extend, safety_bubble
from .vehicle import Vehicle


@dataclass
class GapConfig:
    max_range: float = 10.0
    #: Half the car's width (0.135 m) plus margin. The single most sensitive
    #: number here: too generous and disparity extension closes every gap the
    #: car might have driven through; too tight and it clips ducts.
    car_half_width: float = 0.20
    disparity_threshold: float = 0.3
    bubble_radius: float = 0.35
    #: A beam counts as "open" at this fraction of the deepest reading left
    #: after preprocessing. Relative rather than absolute so it still works in
    #: a tight section where nothing reads near 10 m.
    gap_threshold_frac: float = 0.55
    forward_cone_rad: float = 0.26
    #: Distance we insist on still having in hand when stopped, in metres.
    stop_margin: float = 0.35
    v_min: float = 0.5
    v_max: float = 6.0
    #: How much of v_max a full-lock steering angle gives up.
    steer_speed_gain: float = 0.7


@dataclass
class GapResult:
    steer_rad: float
    v_target: float
    gap_centre_index: Optional[int]


def widest_gap(mask) -> Optional[Tuple[int, int]]:
    """The longest run of True in `mask`, as a half-open [start, end).

    Returns None when nothing is open. Vectorised: a Python loop over 1080
    beams would not fit the 25 ms budget with room to spare.
    """
    flags = np.asarray(mask, dtype=bool)
    if not flags.any():
        return None
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    widest = int(np.argmax(ends - starts))
    return int(starts[widest]), int(ends[widest])


class GapFollower:
    """Reactive driver. Needs nothing but the current scan.

    Takes a `Vehicle` because the speed it asks for is derived from braking
    distance rather than tuned: the fastest speed from which the car can still
    stop inside the clearance it can see is `sqrt(2 * a_brake * (d - margin))`.
    That number is defensible; a fudge factor is not.
    """

    def __init__(self, vehicle: Vehicle, config: Optional[GapConfig] = None):
        self.vehicle = vehicle
        self.config = config or GapConfig()

    def step(self, ranges, angles) -> GapResult:
        cfg = self.config
        angles = np.asarray(angles, dtype=np.float64)

        scan = clean(ranges, cfg.max_range)
        extended = disparity_extend(
            scan, angles, cfg.car_half_width, cfg.disparity_threshold
        )
        bubbled = safety_bubble(extended, angles, cfg.bubble_radius)

        steer, centre = self._aim(bubbled, angles)
        speed = self._speed(extended, angles, steer)
        return GapResult(steer_rad=steer, v_target=speed, gap_centre_index=centre)

    # -- where to point --------------------------------------------------

    def _aim(self, bubbled, angles):
        peak = float(bubbled.max()) if bubbled.size else 0.0
        if peak <= 0.0:
            # Nothing readable at all: a dropout, or the car is inside
            # geometry. Hold straight; `_speed` decides how fast.
            return 0.0, None

        gap = widest_gap(bubbled >= self.config.gap_threshold_frac * peak)
        if gap is None:
            return 0.0, None

        start, end = gap
        centre = (start + end - 1) // 2
        lock = self.vehicle.max_steer_rad
        return float(np.clip(angles[centre], -lock, lock)), centre

    # -- how fast --------------------------------------------------------

    def _speed(self, extended, angles, steer):
        cfg = self.config

        # Clearance is read from the extended scan, not the bubbled one: the
        # bubble writes zeros, and a zero here would read as "no room" every
        # time anything came close.
        forward = np.abs(angles) <= cfg.forward_cone_rad
        clearance = float(extended[forward].min()) if forward.any() else 0.0

        # The fastest speed we could still stop from inside what we can see.
        v_clear = np.sqrt(max(0.0, 2.0 * self.vehicle.a_brake_max * (clearance - cfg.stop_margin)))

        # Give up speed for steering angle: a scaled car that turns hard at
        # speed spins, and a spin is a collision on a 0.9 m corridor.
        lock_fraction = min(abs(steer) / self.vehicle.max_steer_rad, 1.0)
        v_steer = cfg.v_max * max(0.0, 1.0 - cfg.steer_speed_gain * lock_fraction)

        return float(np.clip(min(v_clear, v_steer), cfg.v_min, cfg.v_max))
