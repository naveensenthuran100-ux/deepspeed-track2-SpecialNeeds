"""The car, as numbers.

Everything downstream asks this module two kinds of question:

  * "what throttle holds this speed?"       -> the actuator map
  * "how hard can I corner / accelerate?"   -> the measured limits

The actuator map matters because AutoDRIVE takes normalised motor torque, not
a speed (docs/02-simulator.md 2.4). A feed-forward term derived from a measured
map does most of the work of holding a speed; the PI trim in
`speed_controller.py` only cleans up what the map gets wrong.

Until `tools/sysid.py` has run against the real car, `Vehicle.default()`
carries values derived from the published vehicle specification with
`measured = False`. Those are good enough to develop and test against and are
NOT good enough to race: a run scored on assumed limits is a run tuned to a
car that does not exist.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import List

# Published specification, docs/02-simulator.md 2.10.
WHEELBASE_M = 0.324
LENGTH_M = 0.500
WIDTH_M = 0.270
MASS_KG = 3.906
WHEEL_RADIUS_M = 0.059
MAX_STEER_RAD = 0.5236          # +-30 degrees; steering_command 1.0 means this
MAX_STEER_RATE = 3.2            # rad/s; a step command does not arrive at once
TOP_SPEED_MPS = 22.88           # you will not get near this

_EPS = 1e-12


@dataclass
class Vehicle:
    """Vehicle parameters plus the actuator map.

    `throttle_v_coeffs` are [c1, c2] in the steady-state map

        v = c1 * u + c2 * u**2        for u in [0, 1]

    which is monotonic over [0, 1] for the values we fit, and so invertible in
    closed form. A quadratic is the simplest shape that captures the drop in
    marginal speed per unit throttle at the top of the range; if a fit ever
    needs more than this, the map becomes a lookup table and this comment is
    the thing to delete.
    """

    wheelbase: float = WHEELBASE_M
    length: float = LENGTH_M
    width: float = WIDTH_M
    mass: float = MASS_KG
    wheel_radius: float = WHEEL_RADIUS_M
    max_steer_rad: float = MAX_STEER_RAD
    max_steer_rate: float = MAX_STEER_RATE

    throttle_v_coeffs: List[float] = field(default_factory=lambda: [26.0, -3.12])

    # Limits. Placeholders until sysid measures them - see `measured`.
    a_lat_max: float = 6.0      # m/s^2, cornering grip
    a_accel_max: float = 6.0    # m/s^2, traction-limited acceleration
    a_brake_max: float = 5.0    # m/s^2, deceleration we are willing to ask for
    a_coast: float = 1.5        # m/s^2, deceleration with the throttle simply lifted
    v_max: float = TOP_SPEED_MPS

    measured: bool = False

    # -- construction ----------------------------------------------------

    @classmethod
    def default(cls) -> "Vehicle":
        """Published specification plus assumed limits. `measured` is False."""
        return cls()

    @classmethod
    def load(cls, path: str) -> "Vehicle":
        with open(path, "r") as handle:
            data = json.load(handle)
        known = {f for f in cls().to_dict()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    # -- the actuator map ------------------------------------------------

    def speed_for_throttle(self, u: float) -> float:
        """Steady-state speed held by throttle `u`. Clamped to [0, 1]."""
        c1, c2 = self.throttle_v_coeffs
        u = min(max(float(u), 0.0), 1.0)
        return c1 * u + c2 * u * u

    def throttle_for_speed(self, v: float) -> float:
        """Throttle that holds speed `v` in steady state, clamped to [0, 1].

        This is the feed-forward term. Solving c2*u^2 + c1*u - v = 0 rather
        than searching keeps it exact and keeps the control loop cheap.
        """
        v = float(v)
        if v <= 0.0:
            return 0.0
        if v >= self.speed_for_throttle(1.0):
            return 1.0

        c1, c2 = self.throttle_v_coeffs
        if abs(c2) < _EPS:
            return min(max(v / c1, 0.0), 1.0)

        disc = c1 * c1 + 4.0 * c2 * v
        u = (-c1 + math.sqrt(max(disc, 0.0))) / (2.0 * c2)
        return min(max(u, 0.0), 1.0)

    # -- steering --------------------------------------------------------

    def steer_to_norm(self, rad: float) -> float:
        """Steering angle in radians -> the normalised command in [-1, 1]."""
        return min(max(float(rad) / self.max_steer_rad, -1.0), 1.0)

    def norm_to_steer(self, norm: float) -> float:
        """Normalised steering command -> the angle it asks for, in radians."""
        return min(max(float(norm), -1.0), 1.0) * self.max_steer_rad
