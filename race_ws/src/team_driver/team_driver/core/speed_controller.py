"""Turning a target speed into a throttle.

AutoDRIVE takes normalised motor torque in [-1, 1], not a speed request, so
nothing between this node and the wheels closes the speed loop. Every planner
in `core/` produces a target speed; this is the only place that becomes a
command.

The shape is feed-forward plus a PI trim:

  * feed-forward carries most of the load, because `Vehicle.throttle_for_speed`
    already knows roughly what throttle holds a speed;
  * PI cleans up what the map gets wrong - a gradient, a tyre scrubbing in a
    corner, a map fitted on a day the car behaved slightly differently.

Two details are load-bearing rather than decorative:

  * The integral is clamped. One slow corner is otherwise enough to wind it up
    to the point where it holds the throttle open halfway down the next
    straight.
  * A small overspeed lifts; only a large one brakes. The car has no ABS, so a
    negative throttle locks the wheels, and a locked wheel steers nowhere at
    all - which on a 0.9 m corridor is a collision rather than a slow corner.
"""

from .vehicle import Vehicle


class SpeedController:
    """Feed-forward + PI on speed error, with anti-windup and a lift band.

    Args:
        vehicle: supplies the feed-forward actuator map.
        kp, ki: proportional and integral gains on speed error, in throttle
            units per (m/s) and per (m/s * s) respectively.
        integral_limit: hard clamp on the integral term, in throttle units.
        lift_band: how far over the target we coast before we brake, in m/s.
        brake_gain: throttle per m/s of excess once past the lift band. The
            speed profile plans its braking at `a_brake_max`, so this has to
            be high enough that the controller can actually ask for
            `a_brake_max` - at 0.3 it never asked for more than a third of
            it, and the car arrived at every corner too fast.
        full_throttle_band: below the target by more than this, in m/s, the
            answer is full throttle. 0 disables it. The speed profile already
            plans every acceleration at the traction limit, so being under
            it is time lost; feed-forward plus a trim instead creeps up on
            the target asymptotically. On the real car, whose feed-forward
            was ~38% short, a 10 m/s target settled near 9.2 m/s.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        kp: float = 0.05,
        ki: float = 0.15,
        integral_limit: float = 0.30,
        lift_band: float = 0.5,
        brake_gain: float = 1.0,
        full_throttle_band: float = 0.0,
    ):
        self.vehicle = vehicle
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_limit = abs(float(integral_limit))
        self.lift_band = abs(float(lift_band))
        self.brake_gain = abs(float(brake_gain))
        self.full_throttle_band = max(float(full_throttle_band), 0.0)
        self.integral = 0.0

    def reset(self) -> None:
        """Clear the integral. Call between runs, and on a mode change."""
        self.integral = 0.0

    def step(self, v_target: float, v_measured: float, dt: float) -> float:
        """One control step. Returns a throttle in [-1, 1]."""
        v_target = float(v_target)
        v_measured = float(v_measured)
        error = v_target - v_measured

        # Badly fast: brake, and do not let the integral remember it. The
        # integral exists to correct a steady-state offset, and a braking
        # event is not one.
        if error < -self.lift_band:
            self.integral = 0.0
            excess = -error - self.lift_band
            return _clip(-self.brake_gain * excess, -1.0, 0.0)

        if self.full_throttle_band > 0.0 and error > self.full_throttle_band:
            return 1.0

        feed_forward = self.vehicle.throttle_for_speed(v_target)
        unsaturated = feed_forward + self.kp * error + self.ki * self.integral

        # Anti-windup: only integrate while the output can still respond. dt is
        # guarded because a dropped scan or a first call can hand us zero.
        if dt > 0.0 and 0.0 < unsaturated < 1.0:
            self.integral = _clip(
                self.integral + error * dt, -self.integral_limit, self.integral_limit
            )

        throttle = feed_forward + self.kp * error + self.ki * self.integral

        # Lift, never brake, inside the band: floor at coasting.
        return _clip(throttle, 0.0, 1.0)


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)
