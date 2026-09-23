"""Vehicle model: the actuator map and the measured limits everything else uses."""
import json

import numpy as np
import pytest

from team_driver.core.vehicle import Vehicle


def test_throttle_speed_roundtrip():
    v = Vehicle.default()
    for speed in [0.5, 1.0, 3.0, 5.0]:
        u = v.throttle_for_speed(speed)
        assert 0.0 <= u <= 1.0
        assert abs(v.speed_for_throttle(u) - speed) < 1e-6


def test_zero_target_gives_zero_throttle():
    assert Vehicle.default().throttle_for_speed(0.0) == 0.0


def test_negative_target_gives_zero_throttle():
    assert Vehicle.default().throttle_for_speed(-2.0) == 0.0


def test_throttle_for_unreachable_speed_clamps_to_one():
    v = Vehicle.default()
    assert v.throttle_for_speed(1e6) == 1.0


def test_default_is_marked_unmeasured():
    assert Vehicle.default().measured is False


def test_steering_normalisation_roundtrip():
    v = Vehicle.default()
    for rad in [-0.5236, -0.2, 0.0, 0.3, 0.5236]:
        assert abs(v.norm_to_steer(v.steer_to_norm(rad)) - rad) < 1e-9


def test_steering_normalisation_clamps_beyond_lock():
    v = Vehicle.default()
    assert v.steer_to_norm(10.0) == 1.0
    assert v.steer_to_norm(-10.0) == -1.0


def test_load_roundtrips_through_json(tmp_path):
    path = tmp_path / "vehicle.json"
    path.write_text(json.dumps(Vehicle.default().to_dict()))
    loaded = Vehicle.load(str(path))
    assert loaded.wheelbase == Vehicle.default().wheelbase
    assert loaded.measured is False


def test_speed_for_throttle_is_monotonic_over_full_range():
    v = Vehicle.default()
    speeds = [v.speed_for_throttle(u) for u in np.linspace(0.0, 1.0, 50)]
    assert all(b > a for a, b in zip(speeds, speeds[1:]))
