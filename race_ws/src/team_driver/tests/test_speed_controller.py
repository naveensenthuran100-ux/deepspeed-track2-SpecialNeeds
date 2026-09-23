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
