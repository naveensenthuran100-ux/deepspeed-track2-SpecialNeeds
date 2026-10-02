"""Speed controller: the thing AutoDRIVE does not give us.

Throttle is normalised torque, so every target speed the planner produces has
to be turned into one here. These tests pin the four behaviours that cost lap
time or laps when they are wrong: it reaches the setpoint, the integral cannot
wind up, a small overspeed lifts instead of locking the wheels, and a large
one actually brakes.
"""
import numpy as np
import pytest

from team_driver.core.speed_controller import SpeedController
from team_driver.core.vehicle import Vehicle


def _plant_step(vehicle, speed, throttle, lag=0.12):
    """First-order lag towards the steady-state speed for that throttle.

    Crude on purpose: these tests are about the controller, not the car. The
    real dynamics arrive with sysid, and the surrogate is where they are used.
    """
    return speed + (vehicle.speed_for_throttle(throttle) - speed) * lag


def test_holds_setpoint_from_rest():
    vehicle = Vehicle.default()
    controller = SpeedController(vehicle)
    speed = 0.0
    for _ in range(400):
        speed = _plant_step(vehicle, speed, controller.step(3.0, speed, 0.025))
    assert abs(speed - 3.0) < 0.15


def test_integral_is_clamped():
    controller = SpeedController(Vehicle.default(), integral_limit=0.2)
    for _ in range(1000):
        controller.step(5.0, 0.0, 0.025)
    assert abs(controller.integral) <= 0.2 + 1e-9


def test_slightly_fast_lifts_rather_than_brakes():
    controller = SpeedController(Vehicle.default())
    assert controller.step(3.0, 3.2, 0.025) >= 0.0


def test_badly_fast_brakes():
    controller = SpeedController(Vehicle.default())
    assert controller.step(1.0, 6.0, 0.025) < 0.0


def test_zero_and_negative_target_are_safe():
    controller = SpeedController(Vehicle.default())
    for target in [0.0, -1.0]:
        controller.reset()
        for _ in range(200):
            throttle = controller.step(target, 0.0, 0.025)
            assert -1.0 <= throttle <= 1.0


def test_reset_clears_the_integral():
    controller = SpeedController(Vehicle.default())
    for _ in range(50):
        controller.step(5.0, 0.0, 0.025)
    assert controller.integral != 0.0
    controller.reset()
    assert controller.integral == 0.0


def test_non_positive_dt_does_not_move_the_integral():
    controller = SpeedController(Vehicle.default())
    controller.step(5.0, 0.0, 0.025)
    before = controller.integral
    controller.step(5.0, 0.0, 0.0)
    controller.step(5.0, 0.0, -0.01)
    assert controller.integral == before


def test_output_is_always_bounded_and_finite():
    controller = SpeedController(Vehicle.default())
    rng = np.random.RandomState(0)
    for _ in range(2000):
        throttle = controller.step(
            float(rng.uniform(-5.0, 25.0)),
            float(rng.uniform(-5.0, 25.0)),
            float(rng.uniform(0.0, 0.05)),
        )
        assert np.isfinite(throttle)
        assert -1.0 <= throttle <= 1.0


def test_brake_command_can_reach_full_authority():
    """The profile plans its braking at a_brake_max, so the controller has to
    be able to ask for a_brake_max. It could not: a proportional law at
    0.3 throttle per m/s of excess capped real deceleration at about a third
    of what the profile assumed, so the car arrived at every corner too fast
    and ran wide. Caught by the first surrogate lap.
    """
    controller = SpeedController(Vehicle.default())
    assert controller.step(2.0, 4.0, 0.025) <= -0.9


def test_braking_is_proportional_before_it_saturates():
    controller = SpeedController(Vehicle.default())
    mild = controller.step(3.0, 3.8, 0.025)
    controller.reset()
    hard = controller.step(3.0, 5.5, 0.025)
    assert hard < mild <= 0.0


def test_full_throttle_band_drives_at_the_target_instead_of_creeping():
    """Far under target is time lost: with the band on, the answer is 1.0."""
    on = SpeedController(Vehicle.default(), full_throttle_band=0.5)
    off = SpeedController(Vehicle.default())
    assert on.step(8.0, 4.0, 0.025) == 1.0
    assert off.step(8.0, 4.0, 0.025) < 1.0


def test_full_throttle_band_hands_over_to_the_trim_near_target():
    controller = SpeedController(Vehicle.default(), full_throttle_band=0.5)
    near = controller.step(8.0, 7.8, 0.025)
    assert 0.0 < near < 1.0


def test_full_throttle_band_never_overrides_braking():
    controller = SpeedController(Vehicle.default(), full_throttle_band=0.5)
    assert controller.step(3.0, 6.0, 0.025) < 0.0
