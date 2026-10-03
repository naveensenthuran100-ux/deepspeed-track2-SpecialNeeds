"""Path tracking with curvature feed-forward.

Pure pursuit only turns once cross-track error exists. On this circuit that
is a 3x lap-time penalty, and the arithmetic is brutal: the tightest corner
is 0.83 m radius, the lookahead there is about 0.67 m, and at 5 m/s the car
covers that in 0.13 s - while the 3.2 rad/s steering rate limit needs 0.16 s
to reach full lock. The command arrives after the corner does, the car runs
wide, and the only way to keep it on the island is to slow down. That is why
grip_fraction had to drop to 0.30: the tyres were never the limit, the
controller was.

This tracker reads the steering angle out of the path instead of out of the
error:

    delta = atan(L * kappa_preview)        what the corner requires
          + k_heading * heading_error      align with the path
          + atan(-k_cross * e / v)         pull back onto it

The first term is the whole idea. The path's curvature a preview distance
ahead says what the steering angle is about to need to be, so the actuator
is given its lag up front rather than spending it catching up. The preview
scales with speed because the lag is a time, not a distance.

The two error terms are Stanley's, which is the standard pairing: it tracks
a line more exactly than pure pursuit, and its weakness - intolerance of
discontinuities in the path - does not apply to a minimum-curvature line,
which is smooth by construction.
"""

import math
from dataclasses import dataclass

import numpy as np

from .track import Centreline
from .vehicle import Vehicle

_EPS = 1e-9


@dataclass
class TrackerResult:
    steer_rad: float
    preview_m: float
    cross_track_m: float
    heading_error_rad: float
    kappa_preview: float


class FeedForwardTracker:
    """Args:
        tau_preview: how far ahead to read curvature, in seconds of travel.
            Roughly the steering actuator's time constant - the point is to
            hand it the lag it actually has.
        min_preview / max_preview: bounds on that distance, in metres.
        k_heading: weight on heading error.
        k_cross: weight on cross-track error, divided by speed so a gain that
            is stable at 2 m/s does not oscillate at 8 m/s.
        soft_speed: floor under that division, so a stationary car does not
            ask for infinite correction.
        heading_speed_ref: above this speed (m/s) the heading gain falls as
            ref/v. 0 = off. The heading loop's bandwidth is v*k_heading/L: at
            7 m/s and k_heading 0.66 it corrects in ~70 ms, about the loop's
            own latency (stale odometry + actuation, 40-60 ms) - so it
            overshoots, and the real car weaved at ~1 Hz on the straights.
            Scaling the gain by 1/v holds that bandwidth constant.
        k_yaw: yaw-rate damping, 0 = off. Steering against the gap between the
            car's measured yaw rate and the one the line asks for (v * kappa)
            is the damper the heading loop lacks.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        tau_preview: float = 0.16,
        min_preview: float = 0.25,
        max_preview: float = 2.5,
        k_heading: float = 1.0,
        k_cross: float = 2.2,
        soft_speed: float = 1.0,
        heading_speed_ref: float = 0.0,
        k_yaw: float = 0.0,
    ):
        self.vehicle = vehicle
        self.tau_preview = float(tau_preview)
        self.min_preview = float(min_preview)
        self.max_preview = float(max_preview)
        self.k_heading = float(k_heading)
        self.k_cross = float(k_cross)
        self.soft_speed = float(soft_speed)
        self.heading_speed_ref = float(heading_speed_ref)
        self.k_yaw = float(k_yaw)

    def preview_for(self, speed: float) -> float:
        speed = float(speed) if math.isfinite(float(speed)) else 0.0
        distance = self.tau_preview * max(speed, 0.0)
        return min(max(distance, self.min_preview), self.max_preview)

    def step(self, pose_xy, yaw: float, speed: float, path: Centreline,
             yaw_rate=None) -> TrackerResult:
        yaw = float(yaw)
        speed = float(speed) if math.isfinite(float(speed)) else 0.0

        index, _ = path.nearest(pose_xy)
        preview = self.preview_for(speed)

        # -- feed-forward: what the corner ahead requires ------------------
        kappa = self._curvature_at(path, path.s[index] + preview)
        feed_forward = math.atan(self.vehicle.wheelbase * kappa)

        # -- heading error --------------------------------------------------
        tangent = path.tangent[index]
        path_heading = math.atan2(tangent[1], tangent[0])
        heading_error = _wrap(path_heading - yaw)

        # -- cross-track error, signed, positive when left of the path -----
        offset = np.asarray(pose_xy, dtype=np.float64) - path.xy[index]
        cross_track = float(np.dot(offset, path.normal[index]))

        correction = math.atan(
            -self.k_cross * cross_track / (abs(speed) + self.soft_speed)
        )

        k_heading = self.k_heading
        if self.heading_speed_ref > 0.0 and speed > self.heading_speed_ref:
            k_heading *= self.heading_speed_ref / speed

        # -- yaw-rate damping: oppose turning faster or slower than the line --
        damping = 0.0
        if self.k_yaw > 0.0 and yaw_rate is not None and math.isfinite(float(yaw_rate)):
            expected = speed * float(path.curvature[index])
            damping = (-self.k_yaw * self.vehicle.wheelbase * (float(yaw_rate) - expected)
                       / max(abs(speed), self.soft_speed))

        steer = feed_forward + k_heading * heading_error + correction + damping
        lock = self.vehicle.max_steer_rad
        return TrackerResult(
            steer_rad=min(max(steer, -lock), lock),
            preview_m=preview,
            cross_track_m=cross_track,
            heading_error_rad=heading_error,
            kappa_preview=kappa,
        )

    @staticmethod
    def _curvature_at(path: Centreline, s: float) -> float:
        """Curvature at an arc length, interpolated and wrapping the lap."""
        s = float(s) % path.length
        knots = np.concatenate((path.s, [path.length]))
        closed = np.concatenate((path.curvature, path.curvature[:1]))
        return float(np.interp(s, knots, closed))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
