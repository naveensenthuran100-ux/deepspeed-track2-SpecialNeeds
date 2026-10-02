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


def test_friction_circle_brakes_earlier_and_never_faster():
    """Sharing grip with the corner leaves less to brake with, so the braking
    zone starts earlier: every sample is at most as fast, and some slower."""
    s = np.linspace(0.0, 40.0, 401)
    kappa = np.where((s > 20.0) & (s < 24.0), 1.0, 0.0)
    v = Vehicle.default()
    plain = speed_profile(kappa, s, v, v_max=12.0, total_length=40.1)
    shared = speed_profile(kappa, s, v, v_max=12.0, total_length=40.1, friction_circle=True)
    assert np.all(shared <= plain + 1e-9)
    assert np.any(shared < plain - 0.1)


def test_friction_circle_keeps_the_cornering_speed():
    """At the apex all the grip goes to turning; the speed there is unchanged."""
    s = np.linspace(0.0, 40.0, 401)
    kappa = np.where((s > 20.0) & (s < 24.0), 1.0, 0.0)
    v = Vehicle.default()
    shared = speed_profile(kappa, s, v, v_max=12.0, total_length=40.1, friction_circle=True)
    apex = (s > 21.0) & (s < 23.0)
    assert np.allclose(shared[apex], np.sqrt(v.a_lat_max / 1.0), rtol=0.02)


def test_fast_corner_grip_lifts_only_the_open_corners():
    """A wide bend (R 4 m) gets the fast grip; a hairpin (R 0.8 m) keeps the base."""
    s = np.linspace(0.0, 60.0, 601)
    kappa = np.where((s > 10) & (s < 16), 0.25, np.where((s > 35) & (s < 38), 1.25, 0.0))
    v = Vehicle.default()
    base = speed_profile(kappa, s, v, grip_fraction=1.0, v_max=15.0, total_length=60.1)
    fast = speed_profile(kappa, s, v, grip_fraction=1.0, v_max=15.0, total_length=60.1,
                         grip_fraction_fast=1.5, fast_radius=2.0)
    wide = (s > 12) & (s < 14)
    tight = (s > 36) & (s < 37)
    assert np.all(fast[wide] > base[wide] + 0.1)
    assert np.allclose(fast[tight], base[tight])


def test_fast_corner_grip_off_by_default_changes_nothing():
    s = np.linspace(0.0, 40.0, 401)
    kappa = np.where((s > 20.0) & (s < 24.0), 0.3, 0.0)
    v = Vehicle.default()
    assert np.allclose(speed_profile(kappa, s, v, v_max=12.0, total_length=40.1),
                       speed_profile(kappa, s, v, v_max=12.0, total_length=40.1,
                                     grip_fraction_fast=0.0, fast_radius=2.0))


def test_fast_grip_keeps_a_buffer_around_tight_corners():
    """Just past a hairpin the speed must match the no-fast-grip profile: the
    buffer keeps the exit at the hairpin's caution."""
    s = np.linspace(0.0, 60.0, 601)
    kappa = np.where((s > 30) & (s < 33), 1.25, 0.05)
    v = Vehicle.default()
    base = speed_profile(kappa, s, v, grip_fraction=1.0, v_max=15.0, total_length=60.1)
    fast = speed_profile(kappa, s, v, grip_fraction=1.0, v_max=15.0, total_length=60.1,
                         grip_fraction_fast=1.5, fast_radius=2.0, fast_buffer_m=1.0)
    exit_zone = (s > 33.0) & (s < 33.9)
    far = (s > 45) & (s < 50)
    assert np.allclose(fast[exit_zone], base[exit_zone])
    assert np.all(fast[far] >= base[far] - 1e-9)
