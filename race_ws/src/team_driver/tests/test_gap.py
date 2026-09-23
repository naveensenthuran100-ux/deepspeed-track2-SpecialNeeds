"""Follow-the-gap: the driver that has to work before anything else does.

This is the permanent fallback. When the racing line is blocked, when the pose
is implausible, or when anything upstream fails, the car ends up here - so its
degenerate cases matter more than its fast ones.
"""
import numpy as np
import pytest

from team_driver.core.gap import GapConfig, GapFollower, widest_gap
from team_driver.core.vehicle import Vehicle

ANGLES_201 = np.linspace(-1.5, 1.5, 201)


def _follower(**kwargs):
    return GapFollower(Vehicle.default(), GapConfig(**kwargs))


# -- the selection rule, in isolation ---------------------------------------

def test_widest_gap_picks_the_longest_run():
    mask = np.zeros(20, dtype=bool)
    mask[2:5] = True            # 3 wide
    mask[10:18] = True          # 8 wide, the answer
    assert widest_gap(mask) == (10, 18)


def test_widest_gap_prefers_width_over_position():
    mask = np.zeros(20, dtype=bool)
    mask[0:9] = True
    mask[12:16] = True
    assert widest_gap(mask) == (0, 9)


def test_widest_gap_handles_a_run_touching_the_end():
    mask = np.zeros(10, dtype=bool)
    mask[6:] = True
    assert widest_gap(mask) == (6, 10)


def test_widest_gap_with_no_gap_returns_none():
    assert widest_gap(np.zeros(10, dtype=bool)) is None


def test_widest_gap_with_everything_open():
    assert widest_gap(np.ones(10, dtype=bool)) == (0, 10)


# -- the follower -----------------------------------------------------------

def test_steers_towards_the_open_side():
    ranges = np.full(201, 1.0)
    ranges[130:] = 8.0                       # a genuinely passable opening
    result = _follower().step(ranges, ANGLES_201)
    assert result.steer_rad > 0.0            # positive angles are to the left


def test_picks_widest_not_deepest():
    ranges = np.full(201, 3.0)
    ranges[60:80] = 10.0                     # deeper, but narrow
    ranges[140:] = 6.0                       # shallower, but much wider
    result = _follower().step(ranges, ANGLES_201)
    # Assert on the gap actually selected, not on the steering angle: the
    # wide gap sits beyond full lock, so the command is clamped there. That
    # clamp is correct and has its own test.
    assert result.gap_centre_index is not None
    assert 140 <= result.gap_centre_index <= 200
    assert result.steer_rad > 0.0


def test_slows_for_a_close_obstacle_ahead():
    open_scan = np.full(201, 10.0)
    blocked = np.full(201, 10.0)
    blocked[90:111] = 1.2                    # something just ahead
    follower = _follower()
    assert follower.step(blocked, ANGLES_201).v_target < follower.step(
        open_scan, ANGLES_201
    ).v_target


def test_slows_when_steering_hard():
    straight = np.full(201, 10.0)
    turning = np.full(201, 1.5)
    turning[180:] = 10.0                     # only a hard left is open
    follower = _follower()
    assert follower.step(turning, ANGLES_201).v_target < follower.step(
        straight, ANGLES_201
    ).v_target


def test_degenerate_scans_return_finite_commands():
    for ranges in [
        np.zeros(201),
        np.full(201, np.inf),
        np.full(201, np.nan),
        np.full(201, -1.0),
    ]:
        result = _follower().step(ranges, ANGLES_201)
        assert np.isfinite(result.steer_rad)
        assert np.isfinite(result.v_target)
        assert result.v_target >= 0.0


def test_steering_never_exceeds_the_steering_lock():
    rng = np.random.RandomState(7)
    follower = _follower()
    vehicle = Vehicle.default()
    for _ in range(300):
        ranges = rng.uniform(0.0, 10.0, 201)
        result = follower.step(ranges, ANGLES_201)
        assert abs(result.steer_rad) <= vehicle.max_steer_rad + 1e-9


def test_target_speed_never_exceeds_the_configured_maximum():
    follower = _follower(v_max=4.0)
    assert follower.step(np.full(201, 10.0), ANGLES_201).v_target <= 4.0
