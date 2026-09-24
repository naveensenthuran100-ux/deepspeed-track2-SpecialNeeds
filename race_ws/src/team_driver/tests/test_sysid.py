"""System identification: replacing assumed numbers with measured ones.

Everything downstream is relative to these. The speed profile's cornering cap
is sqrt(a_lat / |k|), the braking it plans for is a_brake_max, and the
controller's feed-forward is the throttle/speed map. Ship them wrong and the
car is tuned to a vehicle that does not exist.

These tests fit against data generated from a KNOWN vehicle and check the fit
recovers it. That is the only honest way to test a fitter before the real car
has ever been driven.
"""
import numpy as np
import pytest

from team_driver.core.vehicle import Vehicle
from team_driver.tools.sysid import (fit_actuator_map, fit_grip, fit_longitudinal,
                                     fit_vehicle)


def _truth():
    truth = Vehicle.default()
    truth.throttle_v_coeffs = [18.0, -4.0]
    truth.a_accel_max = 7.5
    truth.a_brake_max = 6.2
    truth.a_coast = 1.9
    truth.a_lat_max = 8.1
    return truth


def test_actuator_map_recovers_known_coefficients():
    truth = _truth()
    throttles = np.linspace(0.05, 1.0, 20)
    speeds = np.array([truth.speed_for_throttle(u) for u in throttles])
    coeffs = fit_actuator_map(throttles, speeds)
    assert np.allclose(coeffs, truth.throttle_v_coeffs, atol=0.05)


def test_actuator_map_tolerates_noise():
    truth = _truth()
    rng = np.random.RandomState(0)
    throttles = np.linspace(0.05, 1.0, 60)
    speeds = np.array([truth.speed_for_throttle(u) for u in throttles])
    speeds = speeds + rng.normal(0.0, 0.05, speeds.shape)
    coeffs = fit_actuator_map(throttles, speeds)
    assert np.allclose(coeffs, truth.throttle_v_coeffs, atol=0.4)


def test_longitudinal_limits_are_the_observed_extremes():
    # dv/dt samples: strong acceleration, gentle coasting, hard braking.
    times = np.arange(0, 3.0, 0.025)
    accel = fit_longitudinal(
        times,
        speed=np.concatenate([
            np.linspace(0, 6.0, 40),        # accelerating
            np.linspace(6.0, 5.0, 40),      # coasting
            np.linspace(5.0, 0.0, 40),      # braking
        ]),
        throttle=np.concatenate([
            np.full(40, 0.8), np.zeros(40), np.full(40, -1.0),
        ]),
        dt=0.025,
    )
    assert accel["a_accel_max"] > 0.0
    assert accel["a_brake_max"] > accel["a_coast"] > 0.0


def test_grip_is_lateral_acceleration_at_the_limit():
    truth = _truth()
    # Circles at increasing speed; the car holds until it exceeds a_lat_max.
    radius = 1.5
    speeds = np.linspace(1.0, 5.0, 50)
    lateral = np.minimum(speeds ** 2 / radius, truth.a_lat_max)
    assert abs(fit_grip(speeds, np.full(50, radius), lateral) - truth.a_lat_max) < 0.3


def test_fit_vehicle_marks_the_result_measured():
    truth = _truth()
    throttles = np.linspace(0.05, 1.0, 20)
    log = {
        "throttle_sweep": (throttles,
                           np.array([truth.speed_for_throttle(u) for u in throttles])),
        "grip": (np.linspace(1.0, 5.0, 50), np.full(50, 1.5),
                 np.minimum(np.linspace(1.0, 5.0, 50) ** 2 / 1.5, truth.a_lat_max)),
    }
    fitted = fit_vehicle(log)
    assert fitted.measured is True
    assert abs(fitted.a_lat_max - truth.a_lat_max) < 0.3


def test_fit_vehicle_keeps_published_geometry():
    """Wheelbase and width are specification, not something to measure."""
    throttles = np.linspace(0.05, 1.0, 20)
    fitted = fit_vehicle({
        "throttle_sweep": (throttles,
                           np.array([_truth().speed_for_throttle(u) for u in throttles])),
    })
    assert fitted.wheelbase == Vehicle.default().wheelbase
    assert fitted.width == Vehicle.default().width


def test_fit_vehicle_without_data_refuses_to_claim_measurement():
    fitted = fit_vehicle({})
    assert fitted.measured is False


# ===========================================================================
# Defects the first real identification run exposed.
# ===========================================================================

def test_v_max_is_never_extrapolated_outside_the_sampled_throttles():
    """The real run swept throttle 0.08-0.35 and the fitted quadratic curves
    back down outside that range. Evaluating it at u=1.0 gave v_max = -60.45,
    and a negative ceiling makes `throttle_for_speed` return 1.0 for every
    speed - permanent full throttle. The fit must not be trusted past the
    data that produced it.
    """
    throttles = np.linspace(0.08, 0.35, 12)
    speeds = 39.803 * throttles - 100.254 * throttles ** 2   # the real fit
    fitted = fit_vehicle({"throttle_sweep": (throttles, speeds)})
    assert fitted.v_max > 0.0
    assert fitted.speed_for_throttle(1.0) > 0.0


def test_actuator_map_stays_monotonic_over_the_whole_range():
    """`throttle_for_speed` inverts this map in closed form, which is only
    valid while it is monotonic on [0, 1]."""
    throttles = np.linspace(0.08, 0.35, 12)
    speeds = 39.803 * throttles - 100.254 * throttles ** 2
    fitted = fit_vehicle({"throttle_sweep": (throttles, speeds)})
    sampled = [fitted.speed_for_throttle(u) for u in np.linspace(0.0, 1.0, 60)]
    assert all(b >= a - 1e-9 for a, b in zip(sampled, sampled[1:]))


def test_coasting_never_fits_harder_than_braking():
    """The run measured a_coast 8.37 against a_brake 7.59, which is
    physically backwards and silently inverts the controller's whole
    lift-rather-than-brake policy."""
    times = np.arange(0, 3.0, 0.025)
    speed = np.concatenate([np.linspace(0, 6.0, 40),
                            np.linspace(6.0, 2.0, 40),    # violent "coast"
                            np.linspace(2.0, 1.5, 40)])   # gentle "brake"
    throttle = np.concatenate([np.full(40, 0.8), np.zeros(40), np.full(40, -1.0)])
    limits = fit_longitudinal(times, speed, throttle, dt=0.025)
    assert limits["a_coast"] <= limits["a_brake_max"] + 1e-9


def test_fitted_vehicle_is_always_usable_by_the_controller():
    """Whatever the data, the result has to be a car the stack can drive."""
    throttles = np.linspace(0.08, 0.35, 12)
    speeds = 39.803 * throttles - 100.254 * throttles ** 2
    fitted = fit_vehicle({"throttle_sweep": (throttles, speeds)})
    for target in (0.0, 1.0, 3.0, 8.0, 50.0):
        u = fitted.throttle_for_speed(target)
        assert 0.0 <= u <= 1.0
    assert fitted.throttle_for_speed(0.0) == 0.0


def test_grip_is_reported_as_a_lower_bound_when_never_saturated():
    """If the lateral acceleration never plateaus, the car never reached its
    limit and the number is a floor, not a measurement."""
    from team_driver.tools.sysid import grip_saturated

    speeds = np.linspace(1.0, 5.0, 60)
    rising = speeds ** 2 / 1.5                      # never levels off
    plateaued = np.minimum(speeds ** 2 / 1.5, 8.1)  # clearly levels off
    assert grip_saturated(rising) is False
    assert grip_saturated(plateaued) is True
