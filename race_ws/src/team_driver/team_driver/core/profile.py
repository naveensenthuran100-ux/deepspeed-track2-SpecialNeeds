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
    friction_circle: bool = False,
    grip_fraction_fast: float = 0.0,
    fast_radius: float = 2.0,
    fast_buffer_m: float = 1.0,
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
        grip_fraction_fast: grip fraction for corners of radius >= fast_radius
            (0 = same as grip_fraction). Measured on the real car, the tight
            hairpins are limited by STEERING LOCK, not grip, so the single
            grip setting is set by them - and the open corners, which the
            tyres (~9 m/s^2) could take far faster, inherit their caution.
        fast_radius: the radius, in metres, from which grip_fraction_fast applies.
        fast_buffer_m: no fast grip within this distance of a tighter corner.
            Measured on the real car: a hairpin EXIT (R 2.1 m) fell inside the
            fast zone, the car accelerated toward the fast target while still
            at full lock at the apex, where the saturated front tyres hold only
            ~5 m/s^2, ran wide and hit the outer wall.
        friction_circle: share grip between cornering and accelerating or
            braking, a_x <= a_x_max * sqrt(1 - (v^2 k / a_lat)^2). Without it
            the profile brakes at a_brake_max right up to a corner it also
            takes at the full lateral limit. Measured on the real car, that
            combined demand saturated the tyres on entry: the car ran wide at
            full lock, and every collision was on the outside of a corner.
    """
    kappa = np.abs(np.asarray(curvature, dtype=np.float64))
    arc = np.asarray(s, dtype=np.float64)
    count = kappa.shape[0]
    if count == 0:
        return np.zeros(0)

    ceiling = float(vehicle.v_max if v_max is None else v_max)
    lateral = np.full(count, max(float(vehicle.a_lat_max) * float(grip_fraction), _EPS))
    if grip_fraction_fast > 0.0:
        tight = kappa * float(fast_radius) > 1.0            # radius < fast_radius
        steps = np.diff(arc)
        spacing = float(np.mean(steps)) if len(steps) else 1.0
        reach = int(np.ceil(max(float(fast_buffer_m), 0.0) / max(spacing, _EPS)))
        if reach > 0 and tight.any():
            # Grow every tight zone by `reach` samples each way, around the loop.
            grown = tight.copy()
            for shift in range(1, reach + 1):
                grown |= np.roll(tight, shift) | np.roll(tight, -shift)
            tight = grown
        lateral[~tight] = max(float(vehicle.a_lat_max) * float(grip_fraction_fast), _EPS)

    # 1. the grip limit. Straight sections have zero curvature and would
    # divide by zero, so they take the ceiling directly.
    speeds = np.where(kappa > _EPS, np.sqrt(lateral / np.maximum(kappa, _EPS)), ceiling)
    speeds = np.clip(speeds, float(v_min), ceiling)

    segments = _segment_lengths(arc, count, total_length)

    accel = float(vehicle.a_accel_max)
    brake = float(vehicle.a_brake_max)

    def share(limit, v, k, lat):
        """Longitudinal acceleration left over at speed v on curvature k."""
        if not friction_circle:
            return limit
        used = min(v * v * k / lat, 1.0)
        return limit * np.sqrt(max(1.0 - used * used, 0.0))

    for _ in range(max(int(rounds), 1)):
        # 2. forwards: never ask for more acceleration than we have.
        for i in range(count):
            j = (i + 1) % count
            reachable = np.sqrt(speeds[i] ** 2
                                + 2.0 * share(accel, speeds[i], kappa[i], lateral[i]) * segments[i])
            if speeds[j] > reachable:
                speeds[j] = reachable

        # 3. backwards: be slow enough now to make the corner later.
        for i in range(count - 1, -1, -1):
            j = (i + 1) % count
            survivable = np.sqrt(speeds[j] ** 2
                                 + 2.0 * share(brake, speeds[j], kappa[j], lateral[j]) * segments[i])
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
