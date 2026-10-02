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


# ===========================================================================
# Line mode, and the switch back to gap. Task 9.
# ===========================================================================

from team_driver.core.profile import speed_profile
from team_driver.core.track import Centreline

STRAIGHT = Centreline(np.stack([np.linspace(0.0, 60.0, 300), np.zeros(300)], axis=1))
CLEAR = np.full(201, 10.0)


def _line_planner(**kwargs):
    planner = Planner(Vehicle.default(), PlannerConfig(**kwargs))
    profile = speed_profile(
        STRAIGHT.curvature, STRAIGHT.s, Vehicle.default(), 1.0, v_max=5.0,
        total_length=STRAIGHT.length,
    )
    planner.load_path(STRAIGHT, profile)
    return planner


def _blocked_scan():
    """Something solid 1 m ahead, right where the path goes."""
    scan = np.full(201, 10.0)
    scan[np.abs(ANGLES) < 0.35] = 1.0
    return scan


def test_uses_line_mode_when_path_and_pose_present():
    result = _line_planner().step(
        CLEAR, ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025
    )
    assert result.mode == "line"


def test_line_mode_target_speed_comes_from_the_profile():
    planner = _line_planner()
    result = planner.step(CLEAR, ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025)
    assert abs(result.v_target - 5.0) < 0.5      # straight path, v_max 5.0


def test_far_from_the_path_falls_back_to_gap():
    result = _line_planner(max_path_offset=1.0).step(
        CLEAR, ANGLES, pose=(10.0, 8.0), yaw=0.0, speed=3.0, dt=0.025
    )
    assert result.mode == "gap"


def test_blocked_path_switches_to_gap():
    planner = _line_planner(block_enter_frames=2)
    for _ in range(5):
        result = planner.step(
            _blocked_scan(), ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025
        )
    assert result.mode == "gap"


def test_a_single_blocked_frame_does_not_switch():
    planner = _line_planner(block_enter_frames=3)
    planner.step(CLEAR, ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025)
    result = planner.step(
        _blocked_scan(), ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025
    )
    assert result.mode == "line"


def test_returns_to_the_line_once_clear_again():
    planner = _line_planner(block_enter_frames=1, block_exit_frames=2)
    for _ in range(3):
        planner.step(_blocked_scan(), ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025)
    for _ in range(5):
        result = planner.step(CLEAR, ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025)
    assert result.mode == "line"


def test_mode_switch_does_not_chatter():
    """Alternating blocked/clear frames must not flip mode every frame."""
    planner = _line_planner(block_enter_frames=3, block_exit_frames=3)
    modes = []
    for i in range(20):
        scan = _blocked_scan() if i % 2 else CLEAR
        modes.append(
            planner.step(scan, ANGLES, pose=(10.0, 0.0), yaw=0.0, speed=3.0, dt=0.025).mode
        )
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    assert switches <= 1


def test_line_mode_still_clips_its_outputs():
    rng = np.random.RandomState(5)
    planner = _line_planner()
    for _ in range(300):
        result = planner.step(
            rng.uniform(0.0, 10.0, 201), ANGLES,
            pose=(float(rng.uniform(0.0, 60.0)), float(rng.uniform(-0.5, 0.5))),
            yaw=float(rng.uniform(-0.3, 0.3)),
            speed=float(rng.uniform(0.0, 6.0)),
            dt=0.025,
        )
        assert -1.0 <= result.steer_norm <= 1.0
        assert -1.0 <= result.throttle <= 1.0
        assert np.isfinite(result.v_target)


def test_load_path_rejects_a_mismatched_profile():
    planner = Planner(Vehicle.default(), PlannerConfig())
    with pytest.raises(ValueError):
        planner.load_path(STRAIGHT, np.zeros(7))


# ===========================================================================
# Regressions from the first surrogate lap.
# ===========================================================================

def _arc_path(radius=1.2, count=400):
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    return Centreline(np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1))


def _planner_on(path, **kwargs):
    planner = Planner(Vehicle.default(), PlannerConfig(**kwargs))
    planner.load_path(path, np.full(len(path.xy), 4.0))
    return planner


def test_an_obstacle_sitting_on_the_path_is_still_blocked():
    """The check has to keep working once it stops using line-of-sight."""
    path = _arc_path()
    planner = _planner_on(path, block_enter_frames=1)
    pose = tuple(path.xy[0])
    yaw = float(np.arctan2(path.tangent[0][1], path.tangent[0][0]))

    # A wall of returns landing right on the path a metre or so ahead.
    ahead = path.point_at_s(path.s[0] + 1.5)
    bearing = np.arctan2(ahead[1] - pose[1], ahead[0] - pose[0]) - yaw
    scan = np.full(201, 10.0)
    scan[np.abs(np.arctan2(np.sin(ANGLES - bearing), np.cos(ANGLES - bearing))) < 0.25] = (
        float(np.hypot(ahead[0] - pose[0], ahead[1] - pose[1]))
    )
    for _ in range(3):
        result = planner.step(scan, ANGLES, pose=pose, yaw=yaw, speed=3.0, dt=0.025)
    assert result.mode == "gap"


def _step_profile_planner(**kwargs):
    """Profile fast for the first half of the straight, slow after it."""
    planner = Planner(Vehicle.default(), PlannerConfig(**kwargs))
    profile = np.where(STRAIGHT.s < 30.0, 8.0, 2.0)
    planner.load_path(STRAIGHT, profile)
    return planner


def test_speed_preview_reads_the_profile_ahead_of_the_car():
    """At 8 m/s with 0.5 s of preview, a slow zone 2 m ahead is already the target."""
    here = _step_profile_planner().step(CLEAR, ANGLES, pose=(28.0, 0.0), yaw=0.0,
                                        speed=8.0, dt=0.025)
    ahead = _step_profile_planner(speed_preview_s=0.5).step(
        CLEAR, ANGLES, pose=(28.0, 0.0), yaw=0.0, speed=8.0, dt=0.025)
    assert here.v_target == pytest.approx(8.0)
    assert ahead.v_target == pytest.approx(2.0)


def test_speed_preview_wraps_across_the_start_line():
    planner = _step_profile_planner(speed_preview_s=1.0)
    result = planner.step(CLEAR, ANGLES, pose=(59.5, 0.0), yaw=0.0, speed=8.0, dt=0.025)
    assert np.isfinite(result.v_target)
