"""An offline simulator: the thing that makes tuning affordable.

A real 12-lap run on AutoDRIVE costs about six minutes of wall clock, is
locked to real time, and is not perfectly repeatable. On one laptop that caps
honest experimentation at roughly 25 configurations a day. This runs a lap in
well under a second, deterministically, so the search happens here and only
the shortlist ever touches the real simulator.

The model
---------
A kinematic bicycle with three additions that matter on a scaled car:

  * the steering actuator is rate-limited (3.2 rad/s), so a step command does
    not arrive instantly however hard the planner asks;
  * longitudinal dynamics approach the actuator map's steady-state speed at a
    bounded acceleration, clipped by the measured traction limits;
  * cornering is grip-limited: asking for more lateral acceleration than
    a_lat_max produces understeer, not a magically tighter turn.

The LiDAR is sphere-traced against a Euclidean distance transform of the
grid, which is what makes this fast. Marching a ray in fixed steps costs
hundreds of lookups per beam; stepping by the distance to the nearest
obstacle converges in a handful, and the same distance field answers "is the
car touching anything?" for free.

Honesty
-------
This is a model and it can lie. A model that disagrees with AutoDRIVE
optimises the car onto a line it cannot hold. The correlation between
surrogate and simulator lap times is measured by lab/report.py, it is
reported, and the surrogate is not trusted outside the range where it has
been checked. Its collision count in particular is an approximation of the
simulator's, not a reproduction of it.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.track import OccupancyGrid
from ..core.vehicle import Vehicle

DEFAULT_FOV = 4.712389          # 270 degrees, as the devkit's scan
DEFAULT_BEAMS = 181


class SurrogateSensor:
    """A synthetic 2D LiDAR, sphere-traced against a distance field.

    `beam_count` is deliberately lower than the devkit's 1080: the planner
    reads angular increments from the scan rather than assuming a count, so a
    coarser scan costs fidelity rather than correctness, and it is the single
    biggest lever on surrogate speed.
    """

    def __init__(self, grid: OccupancyGrid, max_range: float = 10.0,
                 beam_count: int = DEFAULT_BEAMS, fov: float = DEFAULT_FOV,
                 max_steps: int = 96):
        from scipy import ndimage

        self.grid = grid
        self.max_range = float(max_range)
        self.max_steps = int(max_steps)

        free = np.zeros(grid.values.shape, dtype=bool)
        free.flat = [grid.is_free_value(v) for v in grid.values.flat] if False else None
        # Vectorised form of the trinary rule; the loop above is left as a
        # comment on what it means.
        free = (255.0 - grid.values.astype(np.float64)) / 255.0 < grid.free_thresh

        #: Metres from each cell to the nearest cell that is not free.
        self.distance = ndimage.distance_transform_edt(free) * grid.resolution
        self.hit_threshold = grid.resolution
        self.min_step = grid.resolution

        self.angles = np.linspace(-fov / 2.0, fov / 2.0, int(beam_count))

    def clearance(self, xs, ys) -> np.ndarray:
        """Distance to the nearest obstacle at each world point.

        Off-grid is zero: outside the traced area is not free space, and a
        planner told otherwise drives off the map.
        """
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        grid = self.grid

        cols = np.floor((xs - grid.origin[0]) / grid.resolution).astype(np.int64)
        rows = np.floor((ys - grid.origin[1]) / grid.resolution).astype(np.int64)
        inside = (
            (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        )

        out = np.zeros(xs.shape, dtype=np.float64)
        if np.any(inside):
            out[inside] = self.distance[rows[inside], cols[inside]]
        return out

    def scan(self, x: float, y: float, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        """Ranges and their beam angles, as a LaserScan would report them."""
        bearings = self.angles + float(yaw)
        cos_b, sin_b = np.cos(bearings), np.sin(bearings)

        travelled = np.zeros(self.angles.shape, dtype=np.float64)
        active = np.ones(self.angles.shape, dtype=bool)

        for _ in range(self.max_steps):
            if not active.any():
                break
            clearance = self.clearance(x + travelled * cos_b, y + travelled * sin_b)
            active &= clearance >= self.hit_threshold
            step = np.maximum(clearance, self.min_step)
            travelled = np.where(active, np.minimum(travelled + step, self.max_range), travelled)
            active &= travelled < self.max_range

        return np.clip(travelled, 0.0, self.max_range), self.angles


def longitudinal_accel(vehicle: Vehicle, throttle: float, speed: float,
                       lag: float = 2.0) -> float:
    """Acceleration, in m/s^2, for a throttle command at a given speed.

    Three regimes, and getting their ordering right matters more than getting
    any one of them exactly right:

      * throttle < 0 brakes, up to `a_brake_max` at full lock;
      * throttle >= 0 below its steady-state speed accelerates, traction-limited;
      * throttle >= 0 above its steady-state speed coasts down at no more than
        `a_coast`.

    The first version made coasting decelerate at the full braking limit,
    which made lifting off stronger than braking. That is backwards, and it
    hid a missing third of the controller's brake authority.
    """
    if throttle < 0.0:
        return -vehicle.a_brake_max * min(-throttle, 1.0)

    target = vehicle.speed_for_throttle(throttle)
    if speed <= target:
        return min((target - speed) * lag, vehicle.a_accel_max)
    return -min((speed - target) * lag, vehicle.a_coast)


@dataclass
class SimResult:
    status: str
    elapsed_s: float
    collisions: int
    #: Progress along the planner's path, or trajectory length when it has
    #: none. This, not `travelled_m`, is what completes a lap.
    distance_m: float
    travelled_m: float = 0.0
    trace: Dict[str, np.ndarray] = field(default_factory=dict)


def simulate(
    planner,
    vehicle: Vehicle,
    grid: OccupancyGrid,
    start: Tuple[float, float, float],
    distance_m: float,
    max_time_s: float = 120.0,
    dt: float = 0.025,
    sensor: Optional[SurrogateSensor] = None,
    beam_count: int = DEFAULT_BEAMS,
    steer_override: Optional[float] = None,
    speed_lag: float = 2.0,
    collision_debounce_s: float = 0.25,
    stuck_timeout_s: float = 3.0,
    max_off_track_m: float = 3.0,
) -> SimResult:
    """Drive `planner` until it has covered `distance_m` of path.

    Progress is measured along the planner's path when it has one, and as
    trajectory length when it does not. The difference matters: measuring
    distance travelled lets a car that leaves the circuit and drives in a
    straight line accumulate a lap length of travel and be recorded
    COMPLETE, which is a silently wrong number rather than a failure.
    """
    sensor = sensor or SurrogateSensor(grid, max_range=10.0, beam_count=beam_count)
    planner.reset()

    x, y, yaw = (float(v) for v in start)
    speed = 0.0
    steer = 0.0

    path = getattr(planner, "path", None)
    previous_s = None
    progress = 0.0

    travelled = 0.0
    elapsed = 0.0
    collisions = 0
    since_contact = collision_debounce_s
    stuck_for = 0.0
    status = "TIMEOUT"

    # Points along the car's body, measured forward from the rear axle.
    body = np.array([0.0, vehicle.wheelbase / 2.0, vehicle.wheelbase,
                     vehicle.length - 0.08])
    half_width = vehicle.width / 2.0

    steps = int(max_time_s / dt)
    trace = {k: np.zeros(steps) for k in
             ("t", "x", "y", "yaw", "v", "steer", "v_target", "throttle", "offset")}
    modes = []
    recorded = 0

    for _ in range(steps):
        ranges, angles = sensor.scan(x, y, yaw)
        plan = planner.step(ranges, angles, (x, y), yaw, speed, dt)

        # -- actuators -----------------------------------------------------
        wanted = vehicle.norm_to_steer(
            plan.steer_norm if steer_override is None else steer_override
        )
        limit = vehicle.max_steer_rate * dt
        steer += min(max(wanted - steer, -limit), limit)

        accel = longitudinal_accel(vehicle, plan.throttle, speed, speed_lag)
        speed = max(0.0, speed + accel * dt)

        # -- motion, grip-limited -----------------------------------------
        yaw_rate = speed * math.tan(steer) / vehicle.wheelbase
        if speed > 1e-6:
            grip_limit = vehicle.a_lat_max / speed
            yaw_rate = min(max(yaw_rate, -grip_limit), grip_limit)

        yaw += yaw_rate * dt
        step_distance = speed * dt
        x += step_distance * math.cos(yaw)
        y += step_distance * math.sin(yaw)

        travelled += step_distance
        elapsed += dt
        since_contact += dt

        # Progress along the path, unwrapped across the start/finish seam.
        if path is not None:
            index, offset = path.nearest((x, y))
            current_s = float(path.s[index])
            if previous_s is not None:
                step_s = current_s - previous_s
                if step_s < -path.length / 2.0:
                    step_s += path.length
                elif step_s > path.length / 2.0:
                    step_s -= path.length
                progress += step_s
            previous_s = current_s
        else:
            offset = 0.0
            progress = travelled

        # -- contact -------------------------------------------------------
        touching = bool(
            np.any(
                sensor.clearance(x + body * math.cos(yaw), y + body * math.sin(yaw))
                < half_width
            )
        )
        # Continuous contact keeps counting. Resetting the timer on every
        # frame of contact would record a car wedged against a wall for
        # thirty seconds as a single collision, and a sweep scored on that
        # learns to prefer configurations that scrape.
        if touching and since_contact >= collision_debounce_s:
            collisions += 1
            since_contact = 0.0

        for key, value in (("t", elapsed), ("x", x), ("y", y), ("yaw", yaw),
                           ("v", speed), ("steer", steer),
                           ("v_target", plan.v_target), ("throttle", plan.throttle),
                           ("offset", offset)):
            trace[key][recorded] = value
        modes.append(plan.mode)
        recorded += 1

        stuck_for = stuck_for + dt if speed < 0.05 else 0.0
        if stuck_for >= stuck_timeout_s:
            status = "STUCK"
            break
        if path is not None and offset > max_off_track_m:
            status = "OFF_TRACK"
            break
        if progress >= distance_m:
            status = "COMPLETE"
            break

    trace = {k: v[:recorded] for k, v in trace.items()}
    trace["mode"] = np.array(modes)
    return SimResult(
        status=status,
        elapsed_s=elapsed,
        collisions=collisions,
        distance_m=progress,
        travelled_m=travelled,
        trace=trace,
    )
