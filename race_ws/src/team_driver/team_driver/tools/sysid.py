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


def fit_actuator_map(throttles, speeds):
    """Least-squares [c1, c2] for the steady-state map v = c1*u + c2*u^2.

    No intercept on purpose: zero throttle holds zero speed, and letting the
    fit invent an offset there makes `throttle_for_speed(0)` nonzero, which
    creeps the car off the grid before the flag.
    """
    u = np.asarray(throttles, dtype=np.float64).ravel()
    v = np.asarray(speeds, dtype=np.float64).ravel()
    design = np.stack([u, u * u], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, v, rcond=None)
    return [float(coeffs[0]), float(coeffs[1])]


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

    return {
        "a_accel_max": limit(commanded > 0.05, accel),
        "a_coast": limit(np.abs(commanded) <= 0.05, -accel),
        "a_brake_max": limit(commanded < -0.05, -accel),
    }


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
    print("Re-run lab/sweep.py: every previous result was relative to the assumed values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
