"""Fit vehicle parameters from recorded driving.

Everything downstream is relative to these numbers. The speed profile's
cornering cap is sqrt(a_lat / |k|), the braking it plans for is a_brake_max,
and the controller's feed-forward is the throttle/speed map. Assume them and
the car is tuned to a vehicle that does not exist; measure them and the
tuning is about the driver rather than about the guess.

Run `sysid_drive` first to produce the log:

    ros2 run team_driver sysid_drive                       # ~2 minutes
    python3 -m team_driver.tools.sysid --log results/sysid.csv \
        --out race_ws/src/team_driver/config/vehicle.json

Measuring the car, rather than tuning against it, is also the thing that
makes the interview easy: every constant in the stack has a recorded
provenance.
"""

import argparse
import csv
from typing import Dict, Optional

import numpy as np

from ..core.vehicle import Vehicle

#: Read limits at a high percentile rather than the outright maximum. One
#: sample of a wheel skipping a kerb is not the grip limit.
LIMIT_PERCENTILE = 95.0

#: Floor on the actuator map's slope, so a degenerate sweep cannot produce a
#: car that refuses to move.
_MIN_SLOPE = 1.0

#: Lateral acceleration counts as saturated if its top decile is this flat.
_SATURATION_TOLERANCE = 0.06


def fit_actuator_map(throttles, speeds):
    """Least-squares [c1, c2] for the steady-state map v = c1*u + c2*u^2.

    No intercept on purpose: zero throttle holds zero speed, and letting the
    fit invent an offset there makes `throttle_for_speed(0)` nonzero, which
    creeps the car off the grid before the flag.

    The quadratic is forced to stay monotonic over the whole [0, 1] range,
    not just over the throttles we sampled. On a real run the car can only be
    swept over a narrow band - a 54 m circuit has nowhere to hold full
    throttle - and an unconstrained fit curves back down outside it. The
    first real identification run produced v = 39.8u - 100.3u^2, which peaks
    at u = 0.2 and is NEGATIVE by u = 1.0. `throttle_for_speed` inverts this
    map in closed form and returns 1.0 for anything above its top speed, so a
    negative top speed means permanent full throttle.
    """
    u = np.asarray(throttles, dtype=np.float64).ravel()
    v = np.asarray(speeds, dtype=np.float64).ravel()
    design = np.stack([u, u * u], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, v, rcond=None)
    c1, c2 = float(coeffs[0]), float(coeffs[1])

    # Monotonic on [0, 1] requires dv/du = c1 + 2*c2*u >= 0 at u = 1, i.e.
    # c2 >= -c1/2. Clamp rather than refit: the data constrains the low-
    # throttle behaviour well and says nothing about the top, so the flattest
    # curve consistent with both is the honest choice.
    if c1 <= 0.0:
        return [max(c1, _MIN_SLOPE), 0.0]
    return [c1, max(c2, -c1 / 2.0)]


def fit_longitudinal(times, speed, throttle, dt=None) -> Dict[str, float]:
    """Acceleration, coast-down and braking limits from a speed trace.

    Split by what the throttle was doing at the time, because the three
    regimes are physically different and the surrogate models them
    separately: the ordering coast < brake is what makes "lift, do not
    stamp" a meaningful policy.
    """
    speed = np.asarray(speed, dtype=np.float64).ravel()
    throttle = np.asarray(throttle, dtype=np.float64).ravel()

    if dt is None:
        times = np.asarray(times, dtype=np.float64).ravel()
        steps = np.diff(times)
    else:
        steps = np.full(speed.shape[0] - 1, float(dt))

    accel = np.diff(speed) / np.maximum(steps, 1e-9)
    commanded = throttle[: accel.shape[0]]

    def limit(mask, values):
        chosen = values[mask]
        return float(np.percentile(chosen, LIMIT_PERCENTILE)) if chosen.size else 0.0

    coast = limit(np.abs(commanded) <= 0.05, -accel)
    brake = limit(commanded < -0.05, -accel)

    # Coasting cannot out-brake braking. When the fit says otherwise the
    # coast samples are contaminated - on a circuit the car is usually
    # cornering while it coasts, and scrubbing off speed in a corner is not
    # coast-down. Believe the braking number and treat coast as bounded by it.
    if coast > brake > 0.0:
        coast = min(coast, brake)

    return {
        "a_accel_max": limit(commanded > 0.05, accel),
        "a_coast": coast,
        "a_brake_max": brake,
    }


def grip_saturated(lateral, tolerance: float = _SATURATION_TOLERANCE) -> bool:
    """Did the lateral acceleration actually plateau?

    If it is still rising when the run ends, the car never reached the limit
    and the figure is a FLOOR, not a measurement. Optimising a speed profile
    against a floor is optimising for a car with more grip than we have
    evidence for, which is how a sweep talks itself into a disqualification.
    """
    values = np.sort(np.abs(np.asarray(lateral, dtype=np.float64).ravel()))
    if values.size < 20:
        return False
    top = values[int(0.90 * values.size):]
    peak = float(top[-1])
    if peak <= 1e-9:
        return False
    return bool((peak - float(top[0])) / peak < tolerance)


def fit_grip(speeds, radii, lateral) -> float:
    """Peak sustained lateral acceleration, in m/s^2.

    Driving circles at increasing speed, the lateral acceleration the car
    actually achieves stops rising once the tyres let go. That plateau is the
    grip limit, and it is the single number the whole speed profile hangs on.
    """
    lateral = np.abs(np.asarray(lateral, dtype=np.float64).ravel())
    if lateral.size == 0:
        return 0.0
    return float(np.percentile(lateral, LIMIT_PERCENTILE))


def fit_vehicle(log: Dict, base: Optional[Vehicle] = None) -> Vehicle:
    """A Vehicle from whatever phases the log contains.

    Geometry - wheelbase, width, steering lock - is published specification
    and is never fitted. Only the limits and the actuator map are measured.

    `measured` is set only when something actually was. A file claiming
    measurement it does not have is worse than an honest default, because
    nothing downstream would warn about it.
    """
    vehicle = base or Vehicle.default()
    fitted = False

    if "throttle_sweep" in log:
        throttles, speeds = log["throttle_sweep"]
        if len(throttles):
            vehicle.throttle_v_coeffs = fit_actuator_map(throttles, speeds)
            vehicle.v_max = vehicle.speed_for_throttle(1.0)
            fitted = True

    if "longitudinal" in log:
        times, speed, throttle, dt = log["longitudinal"]
        if len(speed) > 1:
            limits = fit_longitudinal(times, speed, throttle, dt)
            for key, value in limits.items():
                if value > 0.0:
                    setattr(vehicle, key, value)
            fitted = True

    if "grip" in log:
        speeds, radii, lateral = log["grip"]
        if len(lateral):
            vehicle.a_lat_max = fit_grip(speeds, radii, lateral)
            vehicle.grip_saturated = grip_saturated(lateral)
            fitted = True

    vehicle.measured = bool(fitted)
    return vehicle


def load_log(path: str) -> Dict:
    """Read the CSV `sysid_drive` writes into the phases `fit_vehicle` wants.

    Columns: phase, t, throttle, steering, speed, yaw_rate.
    """
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    def column(subset, name):
        return np.array([float(r[name]) for r in subset], dtype=np.float64)

    log = {}

    sweep = [r for r in rows if r["phase"] == "throttle_sweep"]
    if sweep:
        log["throttle_sweep"] = (column(sweep, "throttle"), column(sweep, "speed"))

    longitudinal = [r for r in rows if r["phase"] == "longitudinal"]
    if len(longitudinal) > 1:
        times = column(longitudinal, "t")
        log["longitudinal"] = (times, column(longitudinal, "speed"),
                               column(longitudinal, "throttle"), None)

    grip = [r for r in rows if r["phase"] == "grip"]
    if grip:
        speeds = column(grip, "speed")
        yaw_rate = column(grip, "yaw_rate")
        # Lateral acceleration is v * yaw_rate; the radius follows from both.
        lateral = speeds * yaw_rate
        radii = np.where(np.abs(yaw_rate) > 1e-6, speeds / np.maximum(np.abs(yaw_rate), 1e-6), np.inf)
        log["grip"] = (speeds, radii, lateral)

    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="CSV written by sysid_drive")
    parser.add_argument("--out", required=True, help="where to write vehicle.json")
    args = parser.parse_args(argv)

    log = load_log(args.log)
    if not log:
        print(f"{args.log} contains no recognised phases; nothing fitted.")
        return 1

    before = Vehicle.default()
    vehicle = fit_vehicle(log)
    vehicle.save(args.out)

    print(f"fitted from {args.log}  (phases: {', '.join(sorted(log))})\n")
    print(f"{'parameter':<20} {'assumed':>10} {'measured':>10}")
    for name in ("a_lat_max", "a_accel_max", "a_brake_max", "a_coast", "v_max"):
        print(f"{name:<20} {getattr(before, name):>10.2f} {getattr(vehicle, name):>10.2f}")
    print(f"{'throttle_v_coeffs':<20} {str(before.throttle_v_coeffs):>10} "
          f"{str([round(c, 3) for c in vehicle.throttle_v_coeffs]):>10}")
    print(f"\nwrote {args.out} (measured={vehicle.measured})")
    if not getattr(vehicle, "grip_saturated", False):
        print(
            "\nWARNING: lateral acceleration never plateaued, so a_lat_max is a\n"
            "         LOWER BOUND, not the limit - the car never reached the edge\n"
            "         of grip during identification. Treat it as 'at least this\n"
            "         much' and find the real edge by raising line.grip_fraction\n"
            "         on the simulator until contact starts."
        )
    print("Re-run lab/sweep.py: every previous result was relative to the assumed values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
