"""Pure pursuit.

Two properties make it the standard choice: it is a few lines of geometry,
and the lookahead distance is a single intuitive knob. The geometry is exact
enough to test against a closed form, which is what most of these do.
"""
import numpy as np
import pytest

from team_driver.core.pursuit import PurePursuit
from team_driver.core.track import Centreline
from team_driver.core.vehicle import Vehicle


def _straight(length=40.0, count=400):
    return Centreline(
        np.stack([np.linspace(0.0, length, count), np.zeros(count)], axis=1)
    )


def _circle(radius=2.0, count=800):
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    return Centreline(np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1))


def test_straight_ahead_gives_zero_steering():
    result = PurePursuit(Vehicle.default()).step((0.0, 0.0), 0.0, 3.0, _straight())
    assert abs(result.steer_rad) < 1e-6
    # ... and it is aiming at a real point one lookahead down the path, not
    # merely returning zero.
    assert abs(result.target_xy[0] - result.lookahead) < 0.2
    assert abs(result.target_xy[1]) < 1e-6


def test_on_a_circle_the_steering_matches_the_closed_form():
    # For a car on a circle of radius R, pure pursuit's curvature is exactly
    # 1/R, so delta = atan(wheelbase / R) whatever the lookahead.
    vehicle = Vehicle.default()
    radius = 2.0
    result = PurePursuit(vehicle).step((radius, 0.0), np.pi / 2, 2.0, _circle(radius))
    assert abs(result.steer_rad - np.arctan(vehicle.wheelbase / radius)) < 0.01


def test_turning_left_on_a_counter_clockwise_circle():
    result = PurePursuit(Vehicle.default()).step((2.0, 0.0), np.pi / 2, 2.0, _circle())
    assert result.steer_rad > 0.0


def test_lookahead_scales_with_speed():
    pursuit = PurePursuit(Vehicle.default())
    path = _straight()
    slow = pursuit.step((0.0, 0.0), 0.0, 1.0, path).lookahead
    fast = pursuit.step((0.0, 0.0), 0.0, 6.0, path).lookahead
    assert fast > slow


def test_lookahead_is_bounded():
    pursuit = PurePursuit(Vehicle.default(), min_lookahead=0.6, max_lookahead=3.0)
    path = _straight()
    assert pursuit.step((0.0, 0.0), 0.0, 0.0, path).lookahead >= 0.6
    assert pursuit.step((0.0, 0.0), 0.0, 100.0, path).lookahead <= 3.0


def test_steers_back_towards_a_line_it_has_drifted_off():
    # Sitting 0.4 m to the left of a straight path, pointing along it: the
    # correction must be to the right.
    result = PurePursuit(Vehicle.default()).step((5.0, 0.4), 0.0, 3.0, _straight())
    assert result.steer_rad < 0.0


def test_steering_is_clamped_to_the_lock():
    vehicle = Vehicle.default()
    # Facing away from a tight circle: the geometric answer exceeds full lock.
    result = PurePursuit(vehicle).step((0.0, 0.0), np.pi, 1.0, _circle(0.5))
    assert abs(result.steer_rad) <= vehicle.max_steer_rad + 1e-9


def test_target_wraps_around_the_end_of_a_closed_lap():
    path = _circle()
    index = len(path.xy) - 2
    result = PurePursuit(Vehicle.default()).step(
        tuple(path.xy[index]), 0.0, 3.0, path
    )
    assert np.all(np.isfinite(result.target_xy))


def test_outputs_are_finite_from_anywhere_on_the_map():
    rng = np.random.RandomState(11)
    pursuit = PurePursuit(Vehicle.default())
    path = _circle()
    for _ in range(300):
        result = pursuit.step(
            (float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20))),
            float(rng.uniform(-np.pi, np.pi)),
            float(rng.uniform(0.0, 10.0)),
            path,
        )
        assert np.isfinite(result.steer_rad)
        assert np.all(np.isfinite(result.target_xy))
