"""Latency compensation: steer from where the car WILL be, not where it was.

The control loop runs on the LiDAR (40 Hz) but steers from the most recent
odometry, which is already up to one odometry period old when the scan
arrives, and the command then takes about one more frame to reach the
wheels. Measured on the simulator: odometry every 16-40 ms, actual steering
following the command one ~20 ms sample later. At 8 m/s that is 13-30 cm of
travel the tracker never sees - in a 1.3 m hairpin channel - and it shows up
as steering that chatters between corrections.

The fix is the standard one: integrate the pose forward over the known age
plus the actuation delay with a constant-speed, constant-turn-rate model.
"""

import math
from typing import Tuple


def predict_pose(x: float, y: float, yaw: float, speed: float, yaw_rate: float,
                 dt: float) -> Tuple[float, float, float]:
    """Pose after `dt` seconds of constant speed and turn rate (CTRV)."""
    if dt <= 0.0 or not all(map(math.isfinite, (x, y, yaw, speed, yaw_rate, dt))):
        return x, y, yaw
    if abs(yaw_rate) < 1e-4:
        return (x + speed * dt * math.cos(yaw), y + speed * dt * math.sin(yaw), yaw)
    turn = yaw + yaw_rate * dt
    radius = speed / yaw_rate
    return (x + radius * (math.sin(turn) - math.sin(yaw)),
            y - radius * (math.cos(turn) - math.cos(yaw)),
            turn)
