"""The planner: the only thing the ROS adapter calls.

Everything the node needs to decide happens here, so these tests are the
guarantee the adapter relies on: whatever the scan and the pose look like,
`step` returns two finite, bounded commands and says which mode produced
them.
"""
import numpy as np
import pytest

from team_driver.core.planner import Planner, PlannerConfig, PlanResult
from team_driver.core.vehicle import Vehicle

ANGLES = np.linspace(-2.356, 2.356, 201)
OPEN_SCAN = np.full(201, 8.0)


def _planner(**kwargs):
    return Planner(Vehicle.default(), PlannerConfig(**kwargs))


def test_falls_back_to_gap_without_pose():
    result = _planner().step(OPEN_SCAN, ANGLES, pose=None, yaw=None, speed=0.0, dt=0.025)
    assert result.mode == "gap"
    assert -1.0 <= result.steer_norm <= 1.0
    assert -1.0 <= result.throttle <= 1.0


def test_gap_mode_when_no_path_is_loaded():
    result = _planner().step(
        OPEN_SCAN, ANGLES, pose=(0.8, 3.16), yaw=-1.57, speed=2.0, dt=0.025
    )
    assert result.mode == "gap"


def test_accelerates_from_rest_on_an_open_track():
    result = _planner().step(OPEN_SCAN, ANGLES, pose=None, yaw=None, speed=0.0, dt=0.025)
    assert result.throttle > 0.0


def test_outputs_are_bounded_and_finite_for_any_scan():
    rng = np.random.RandomState(3)
    planner = _planner()
    for _ in range(500):
        ranges = rng.uniform(-1.0, 12.0, 201)
        ranges[rng.rand(201) < 0.05] = np.nan
        ranges[rng.rand(201) < 0.05] = np.inf
        result = planner.step(
            ranges, ANGLES,
            pose=None, yaw=None,
            speed=float(rng.uniform(0.0, 8.0)),
            dt=float(rng.uniform(0.0, 0.05)),
        )
        assert np.isfinite(result.steer_norm) and np.isfinite(result.throttle)
        assert -1.0 <= result.steer_norm <= 1.0
        assert -1.0 <= result.throttle <= 1.0


def test_non_finite_pose_is_treated_as_no_pose():
    result = _planner().step(
        OPEN_SCAN, ANGLES, pose=(np.nan, 3.0), yaw=0.0, speed=1.0, dt=0.025
    )
    assert result.mode == "gap"
    assert np.isfinite(result.steer_norm)


def test_non_finite_speed_does_not_poison_the_throttle():
    result = _planner().step(
        OPEN_SCAN, ANGLES, pose=None, yaw=None, speed=np.nan, dt=0.025
    )
    assert np.isfinite(result.throttle)


def test_reset_clears_controller_state():
    planner = _planner()
    for _ in range(50):
        planner.step(OPEN_SCAN, ANGLES, pose=None, yaw=None, speed=0.0, dt=0.025)
    planner.reset()
    assert planner.speed_controller.integral == 0.0


def test_result_carries_the_target_speed_for_debugging():
    result = _planner().step(OPEN_SCAN, ANGLES, pose=None, yaw=None, speed=1.0, dt=0.025)
    assert isinstance(result, PlanResult)
    assert result.v_target >= 0.0
