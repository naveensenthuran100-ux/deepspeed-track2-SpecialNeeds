"""Model-predictive steering: sign, the corner it is in, bounds, and speed."""
import math
import time

import numpy as np
import pytest

from team_driver.core.mpc import MPCTracker
from team_driver.core.track import Centreline
from team_driver.core.vehicle import Vehicle

STRAIGHT = Centreline(np.stack([np.linspace(0.0, 60.0, 300), np.zeros(300)], axis=1))
_T = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
CIRCLE = Centreline(np.stack([2.0 * np.cos(_T), 2.0 * np.sin(_T)], axis=1))   # R 2 m, left turn


def _mpc(path, speed=4.0, **kw):
    m = MPCTracker(Vehicle.default(), **kw)
    m.set_profile(np.full(len(path.xy), speed))
    return m


def test_steers_back_towards_the_line():
    left = _mpc(STRAIGHT).step((10.0, 0.4), 0.0, 4.0, STRAIGHT).steer_rad
    right = _mpc(STRAIGHT).step((10.0, -0.4), 0.0, 4.0, STRAIGHT).steer_rad
    assert left < 0.0 < right


def test_on_the_line_in_a_corner_it_steers_the_corner():
    """Exactly on a 2 m circle, heading along it: the answer is atan(L/R)."""
    m = _mpc(CIRCLE)
    m.last_command = math.atan(Vehicle.default().wheelbase / 2.0)   # already turning in
    steer = m.step((2.0, 0.0), math.pi / 2, 4.0, CIRCLE).steer_rad
    assert steer == pytest.approx(math.atan(Vehicle.default().wheelbase / 2.0), abs=0.03)


def test_command_stays_within_the_steering_lock():
    lock = Vehicle.default().max_steer_rad
    for e in (-3.0, 3.0):
        assert abs(_mpc(STRAIGHT).step((10.0, e), 0.0, 6.0, STRAIGHT).steer_rad) <= lock + 1e-12


def test_fits_the_control_budget():
    m = _mpc(CIRCLE, horizon=20)
    started = time.perf_counter()
    for k in range(50):
        m.step((2.0, 0.05), math.pi / 2 + 0.02, 5.0, CIRCLE)
    per_step = (time.perf_counter() - started) / 50
    assert per_step < 0.005, f"{per_step*1000:.1f} ms per step"
