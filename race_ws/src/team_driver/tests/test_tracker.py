"""Curvature feed-forward path tracking.

Pure pursuit is purely reactive: it only turns once cross-track error has
already appeared. On this circuit that is fatal. The tightest corner is
0.83 m radius, the capped lookahead there is about 0.67 m, and at 5 m/s the
car covers that in 0.13 s - while the 3.2 rad/s steering rate limit needs
0.16 s to reach full lock. The car cannot physically be at the angle by the
time it needs it, so it runs wide, so grip_fraction had to come down to 0.30
to keep it on the island. That is a 3x lap-time penalty imposed by the
controller, not by the tyres.

This tracker commands the corner's steering angle BEFORE the error appears:

    delta = atan(L * kappa_preview)      feed-forward, from the path itself
          + k_heading * heading_error    align with the path
          + atan(-k_cross * e / v)       pull back onto it

with the preview distance scaled by speed, so the actuator is given the lag
it actually has.
"""
import numpy as np
import pytest

from team_driver.core.track import Centreline
from team_driver.core.tracker import FeedForwardTracker
from team_driver.core.vehicle import Vehicle


def _straight(length=40.0, count=400):
    return Centreline(np.stack([np.linspace(0.0, length, count),
                                np.zeros(count)], axis=1))


def _circle(radius=2.0, count=800):
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    return Centreline(np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1))


def _tracker(**kwargs):
    return FeedForwardTracker(Vehicle.default(), **kwargs)


def test_on_a_straight_perfectly_aligned_it_does_nothing():
    result = _tracker().step((5.0, 0.0), 0.0, 4.0, _straight())
    assert abs(result.steer_rad) < 1e-6


def test_feed_forward_alone_holds_a_circle():
    """Sitting exactly on a circle, aligned: the answer is pure geometry,
    atan(wheelbase / R), with no error term contributing anything."""
    vehicle = Vehicle.default()
    radius = 2.0
    path = _circle(radius)
    result = _tracker().step((radius, 0.0), np.pi / 2, 4.0, path)
    assert abs(result.steer_rad - np.arctan(vehicle.wheelbase / radius)) < 0.02


def test_steers_right_when_left_of_the_path():
    result = _tracker().step((5.0, 0.3), 0.0, 4.0, _straight())
    assert result.steer_rad < 0.0


def test_steers_left_when_right_of_the_path():
    result = _tracker().step((5.0, -0.3), 0.0, 4.0, _straight())
    assert result.steer_rad > 0.0


def test_corrects_a_heading_error():
    """Pointing left of a straight path, sitting on it: steer right."""
    result = _tracker().step((5.0, 0.0), 0.25, 4.0, _straight())
    assert result.steer_rad < 0.0


def test_preview_looks_further_ahead_at_speed():
    path = _circle(2.0)
    slow = _tracker().step((2.0, 0.0), np.pi / 2, 1.0, path)
    fast = _tracker().step((2.0, 0.0), np.pi / 2, 8.0, path)
    assert fast.preview_m > slow.preview_m


def test_feed_forward_leads_the_corner():
    """Approaching a corner on a straight, the command must already be
    turning before the car reaches it. This is the whole point."""
    # Straight for 10 m, then a left-hand quarter circle.
    straight = np.stack([np.linspace(0.0, 10.0, 100), np.zeros(100)], axis=1)
    angles = np.linspace(-np.pi / 2, 0.0, 100)
    arc = np.stack([10.0 + 2.0 * np.cos(angles), 2.0 + 2.0 * np.sin(angles)], axis=1)
    path = Centreline(np.concatenate([straight, arc]))

    tracker = _tracker()
    approaching = tracker.step((9.4, 0.0), 0.0, 6.0, path)
    assert approaching.steer_rad > 0.02, "not turning in before the corner arrives"


def test_cross_track_gain_softens_with_speed():
    """A fixed lateral gain that is stable at 2 m/s oscillates at 8 m/s."""
    gentle = _tracker().step((5.0, 0.3), 0.0, 8.0, _straight())
    sharp = _tracker().step((5.0, 0.3), 0.0, 2.0, _straight())
    assert abs(sharp.steer_rad) > abs(gentle.steer_rad)


def test_steering_is_clamped_to_the_lock():
    vehicle = Vehicle.default()
    result = _tracker().step((0.0, 0.0), np.pi, 1.0, _circle(0.4))
    assert abs(result.steer_rad) <= vehicle.max_steer_rad + 1e-9


def test_outputs_are_finite_from_anywhere():
    rng = np.random.RandomState(4)
    tracker = _tracker()
    path = _circle()
    for _ in range(300):
        result = tracker.step(
            (float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20))),
            float(rng.uniform(-np.pi, np.pi)),
            float(rng.uniform(0.0, 12.0)),
            path,
        )
        assert np.isfinite(result.steer_rad)
