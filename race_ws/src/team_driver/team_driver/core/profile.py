"""A speed for every point on the path.

This is the single biggest win available from a planner, and the reason the
speed controller had to come first: the output here is a *target speed*, and
something has to turn it into a throttle.

Three steps, in order:

  1. cap cornering speed by the grip we can hold, v = sqrt(a_lat / |k|);
  2. propagate the acceleration limit forwards, so the car never asks for more
     than it has;
  3. propagate the braking limit backwards, so it slows down *in time* rather
     than once it is already in the corner.

Steps 2 and 3 wrap around the lap. A corner sitting near the start/finish line
has to be braked for from the far end of the array, and a profile that treats
the array as a line rather than a loop arrives at the seam far too fast.

Computed offline, once, so the Python loops below cost nothing at run time.
"""

import numpy as np

from .vehicle import Vehicle

_EPS = 1e-9


def speed_profile(
    curvature,
    s,
    vehicle: Vehicle,
    grip_fraction: float = 1.0,
    v_max=None,
    v_min: float = 0.3,
    total_length=None,
    rounds: int = 2,
):
    """Target speed at every sample of a closed path.

    Args:
        curvature: signed or unsigned curvature at each sample, 1/m.
        s: cumulative arc length at each sample, m.
        vehicle: supplies a_lat_max, a_accel_max, a_brake_max and v_max.
        grip_fraction: how much of the measured lateral grip to use. THE
            aggression dial: 1.0 asks for everything the tyres have, which is
            also where a small model error becomes a collision.
        v_max: ceiling, defaulting to the vehicle's top speed.
        v_min: floor, so an absurd curvature cannot ask for a standstill.
        total_length: lap length, used for the wrap segment. Defaults to
            assuming the closing segment matches the mean spacing.
        rounds: forward/backward sweeps. Two converges the wrap; more is
            harmless and costs microseconds.
    """
    kappa = np.abs(np.asarray(curvature, dtype=np.float64))
    arc = np.asarray(s, dtype=np.float64)
    count = kappa.shape[0]
    if count == 0:
        return np.zeros(0)

    ceiling = float(vehicle.v_max if v_max is None else v_max)
    lateral = max(float(vehicle.a_lat_max) * float(grip_fraction), _EPS)

    # 1. the grip limit. Straight sections have zero curvature and would
    # divide by zero, so they take the ceiling directly.
    speeds = np.where(kappa > _EPS, np.sqrt(lateral / np.maximum(kappa, _EPS)), ceiling)
    speeds = np.clip(speeds, float(v_min), ceiling)

    segments = _segment_lengths(arc, count, total_length)

    accel = float(vehicle.a_accel_max)
    brake = float(vehicle.a_brake_max)

    for _ in range(max(int(rounds), 1)):
        # 2. forwards: never ask for more acceleration than we have.
        for i in range(count):
            j = (i + 1) % count
            reachable = np.sqrt(speeds[i] ** 2 + 2.0 * accel * segments[i])
            if speeds[j] > reachable:
                speeds[j] = reachable

        # 3. backwards: be slow enough now to make the corner later.
        for i in range(count - 1, -1, -1):
            j = (i + 1) % count
            survivable = np.sqrt(speeds[j] ** 2 + 2.0 * brake * segments[i])
            if speeds[i] > survivable:
                speeds[i] = survivable

    return np.clip(speeds, float(v_min), ceiling)


def _segment_lengths(arc, count, total_length):
    """Length of the segment leaving each sample, including the closing one."""
    if count < 2:
        return np.full(count, _EPS)

    steps = np.diff(arc)
    if total_length is not None:
        closing = float(total_length) - float(arc[-1])
    else:
        closing = float(np.mean(steps))

    lengths = np.concatenate((steps, [closing]))
    return np.maximum(lengths, _EPS)
