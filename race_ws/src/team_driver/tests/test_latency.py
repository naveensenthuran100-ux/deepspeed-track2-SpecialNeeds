"""Latency compensation: the projected pose must match the motion it models."""
import math

import pytest

from team_driver.core.latency import predict_pose


def test_straight_line_moves_along_the_heading():
    x, y, yaw = predict_pose(1.0, 2.0, math.pi / 2, 4.0, 0.0, 0.25)
    assert (x, y, yaw) == pytest.approx((1.0, 3.0, math.pi / 2))


def test_constant_turn_stays_on_the_circle():
    """Speed 2, yaw rate 1 -> radius 2. A quarter turn ends a radius away on both axes."""
    x, y, yaw = predict_pose(0.0, 0.0, 0.0, 2.0, 1.0, math.pi / 2)
    assert (x, y, yaw) == pytest.approx((2.0, 2.0, math.pi / 2), abs=1e-9)


def test_zero_or_negative_time_and_bad_input_change_nothing():
    assert predict_pose(1.0, 1.0, 0.3, 5.0, 0.4, 0.0) == (1.0, 1.0, 0.3)
    assert predict_pose(1.0, 1.0, 0.3, 5.0, 0.4, -0.1) == (1.0, 1.0, 0.3)
    assert predict_pose(1.0, 1.0, 0.3, float("nan"), 0.4, 0.05) == (1.0, 1.0, 0.3)
