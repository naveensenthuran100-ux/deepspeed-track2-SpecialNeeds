"""Speed profile: a speed for every point on the path.

The single biggest win available from a planner. A centreline driven at one
speed is not a lap; what makes it one is capping cornering speed by the grip
we can hold, then propagating braking and acceleration limits around the loop
so the car slows down *in time* rather than once it is already in the corner.
"""
import numpy as np
import pytest

from team_driver.core.profile import speed_profile
from team_driver.core.vehicle import Vehicle


def _limits_respected(speeds, s, a_accel, a_brake, tol=1e-6):
    ds = np.diff(np.concatenate((s, [s[-1] + (s[1] - s[0])])))
    nxt = np.roll(speeds, -1)
    accel = (nxt ** 2 - speeds ** 2) / (2.0 * ds)
    return np.all(accel <= a_accel + tol) and np.all(-accel <= a_brake + tol)


def test_constant_radius_gives_constant_speed():
    curvature = np.full(400, 0.5)
    s = np.linspace(0.0, 2 * np.pi * 2.0, 400)
    speeds = speed_profile(curvature, s, Vehicle.default(), 1.0)
    assert np.allclose(speeds, speeds[0], rtol=0.02)


def test_cornering_speed_matches_the_grip_limit():
    vehicle = Vehicle.default()
    curvature = np.full(400, 0.5)
    s = np.linspace(0.0, 2 * np.pi * 2.0, 400)
    speeds = speed_profile(curvature, s, vehicle, 1.0)
    assert abs(speeds[0] - np.sqrt(vehicle.a_lat_max / 0.5)) < 0.05


def test_grip_fraction_scales_cornering_speed():
    curvature = np.full(400, 0.5)
    s = np.linspace(0.0, 2 * np.pi * 2.0, 400)
    full = speed_profile(curvature, s, Vehicle.default(), 1.0)
    half = speed_profile(curvature, s, Vehicle.default(), 0.5)
    assert np.all(half < full)


def test_brakes_before_the_corner_not_in_it():
    curvature = np.zeros(400)
    curvature[200:] = 1.0
    s = np.linspace(0.0, 80.0, 400)
    speeds = speed_profile(curvature, s, Vehicle.default(), 1.0)
    assert speeds[199] < speeds[100]


def test_respects_acceleration_and_braking_limits():
    vehicle = Vehicle.default()
    curvature = np.zeros(400)
    curvature[200:260] = 1.2
    s = np.linspace(0.0, 80.0, 400)
    speeds = speed_profile(curvature, s, vehicle, 1.0)
    assert _limits_respected(speeds, s, vehicle.a_accel_max, vehicle.a_brake_max, tol=1e-3)


def test_straight_line_is_capped_by_v_max():
    speeds = speed_profile(
        np.zeros(200), np.linspace(0.0, 400.0, 200), Vehicle.default(), 1.0, v_max=5.0
    )
    assert np.all(speeds <= 5.0 + 1e-9)
    assert np.max(speeds) > 4.9


def test_zero_curvature_does_not_divide_by_zero():
    speeds = speed_profile(
        np.zeros(100), np.linspace(0.0, 50.0, 100), Vehicle.default(), 1.0
    )
    assert np.all(np.isfinite(speeds))


def test_speeds_are_positive_and_finite_for_extreme_curvature():
    curvature = np.concatenate((np.zeros(50), np.full(50, 1e6)))
    speeds = speed_profile(
        curvature, np.linspace(0.0, 50.0, 100), Vehicle.default(), 1.0
    )
    assert np.all(np.isfinite(speeds))
    assert np.all(speeds > 0.0)


def test_profile_is_periodic_around_the_lap():
    # A corner placed at the seam must be braked for from the far end of the
    # array, or the car arrives at the start/finish line far too fast.
    curvature = np.zeros(400)
    curvature[:20] = 1.5
    s = np.linspace(0.0, 80.0, 400)
    speeds = speed_profile(curvature, s, Vehicle.default(), 1.0)
    assert speeds[-1] < speeds[200]
