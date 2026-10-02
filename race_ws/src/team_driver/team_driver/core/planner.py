"""What to do, given a scan and a pose.

The ROS adapter calls exactly one thing: `Planner.step`. Everything the car
decides happens below this line and none of it imports ROS, which is what
makes it testable in milliseconds rather than in six-minute simulator runs.

Modes
-----
`line`  Follows the optimised racing line against ground-truth pose, at the
        speed the offline profile computed for this point. This is where the
        lap time is: the profile knows about a corner the 10 m LiDAR cannot
        see yet, so the car brakes before it rather than in it.
`gap`   Reactive, needs nothing but the scan. Used whenever the line cannot
        be trusted: no pose, an implausible pose, a pose too far off the
        path, or something sitting on the path ahead.

The switch is hysteretic. A path follower follows its line and nothing else,
so it must give way when the line is blocked - but a mode that flips every
frame steers worse than either mode alone, which on a 0.9 m corridor is a
collision.

The contract the adapter relies on, and the tests enforce: `step` always
returns finite commands inside [-1, 1], whatever the scan, the pose or the
speed look like. A planner that raises leaves the bridge repeating the last
command it got, which is the car driving on at its last throttle.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .gap import GapConfig, GapFollower
from .lidar import clean
from .pursuit import PurePursuit
from .tracker import FeedForwardTracker
from .speed_controller import SpeedController
from .track import Centreline
from .vehicle import Vehicle

LINE = "line"
GAP = "gap"


@dataclass
class PlannerConfig:
    gap: GapConfig = field(default_factory=GapConfig)

    # Speed controller.
    kp: float = 0.05
    ki: float = 0.15
    integral_limit: float = 0.30
    #: Small on purpose: coasting through a 0.5 m/s overspeed before braking
    #: starts the braking late, and cost ~0.2 s a lap in tuning.
    lift_band: float = 0.15
    #: Must match params.py. The speed profile plans its braking at
    #: a_brake_max, so this has to be high enough to actually ask for it.
    brake_gain: float = 1.0
    #: Below the target by more than this (m/s), full throttle. 0 = off.
    full_throttle_band: float = 0.0
    #: Read the target speed this many seconds of travel ahead of the car,
    #: to cover actuation and pose latency. 0 = at the car's own position.
    speed_preview_s: float = 0.0

    # Pure pursuit.
    k_lookahead: float = 0.35
    min_lookahead: float = 0.45
    max_lookahead: float = 2.0
    #: Cap the lookahead at this fraction of the local turn radius. Aiming
    #: further ahead than the corner radius tracks a chord across the corner.
    curvature_cap: float = 0.8

    #: "feedforward" (curvature feed-forward + Stanley errors) or "pursuit".
    #: Pure pursuit only turns once error exists, which on 0.83 m corners
    #: arrives after the corner does; see core/tracker.py.
    tracker: str = "feedforward"
    tau_preview: float = 0.16
    min_preview: float = 0.25
    max_preview: float = 2.5
    k_heading: float = 0.7
    k_cross: float = 2.2

    #: Give up on the line if we are further than this from it, in metres.
    #: Either the pose is wrong or we have already left the track; in both
    #: cases the scan is the only thing worth believing.
    max_path_offset: float = 1.2

    #: How far along the path to check for obstructions, as a floor on the
    #: braking distance at the current speed.
    block_min_horizon: float = 1.5
    block_samples: int = 8
    #: A LiDAR return within this distance of the path ahead counts as
    #: something in the way. Must stay well under the corridor half-width
    #: (about 0.67 m at the tightest point) or the boundary itself reads as
    #: an obstruction.
    block_radius: float = 0.30

    #: Hysteresis, in frames at 40 Hz. Roughly 75 ms in, 250 ms out - quick to
    #: duck, slow to commit back.
    block_enter_frames: int = 3
    block_exit_frames: int = 10


@dataclass
class PlanResult:
    steer_norm: float
    throttle: float
    mode: str
    v_target: float


class Planner:
    def __init__(self, vehicle: Vehicle, config: Optional[PlannerConfig] = None):
        self.vehicle = vehicle
        self.config = config or PlannerConfig()

        self.gap_follower = GapFollower(vehicle, self.config.gap)
        self.pursuit = PurePursuit(
            vehicle,
            k_lookahead=self.config.k_lookahead,
            min_lookahead=self.config.min_lookahead,
            max_lookahead=self.config.max_lookahead,
            curvature_cap=self.config.curvature_cap,
        )
        self.tracker = FeedForwardTracker(
            vehicle,
            tau_preview=self.config.tau_preview,
            min_preview=self.config.min_preview,
            max_preview=self.config.max_preview,
            k_heading=self.config.k_heading,
            k_cross=self.config.k_cross,
        )
        self.speed_controller = SpeedController(
            vehicle,
            kp=self.config.kp,
            ki=self.config.ki,
            integral_limit=self.config.integral_limit,
            lift_band=self.config.lift_band,
            brake_gain=self.config.brake_gain,
            full_throttle_band=self.config.full_throttle_band,
        )

        self.path: Optional[Centreline] = None
        self.profile: Optional[np.ndarray] = None

        self.mode = GAP
        self._blocked_frames = 0
        self._clear_frames = 0

    # -- setup -----------------------------------------------------------

    def load_path(self, path: Centreline, profile) -> None:
        """Install the racing line and its speed profile.

        Validated rather than trusted: a profile of the wrong length would
        silently index the wrong corner, which is a collision at the first
        place the two disagree.
        """
        profile = np.asarray(profile, dtype=np.float64)
        if profile.shape[0] != len(path.xy):
            raise ValueError(
                f"profile has {profile.shape[0]} samples, path has {len(path.xy)}"
            )
        self.path = path
        self.profile = profile
        # Acquire the line immediately. The hysteresis below governs recovery
        # after a block, not first acquisition - the blocked check runs on
        # frame one, so waiting out block_exit_frames here would just give
        # away a quarter of a second of every run for nothing.
        self.mode = LINE
        self._blocked_frames = self._clear_frames = 0

    def reset(self) -> None:
        self.speed_controller.reset()
        self.mode = LINE if self.path is not None else GAP
        self._blocked_frames = 0
        self._clear_frames = 0

    # -- the control step ------------------------------------------------

    def step(self, ranges, angles, pose, yaw, speed, dt) -> PlanResult:
        speed = _finite(speed, 0.0)
        dt = _finite(dt, 0.0)
        angles = np.asarray(angles, dtype=np.float64)
        scan = clean(ranges, self.config.gap.max_range)

        pose_yaw = _valid_pose(pose, yaw)
        steer_rad, v_target, mode = self._decide(scan, angles, pose_yaw, speed)

        throttle = self.speed_controller.step(v_target, speed, dt)
        return PlanResult(
            steer_norm=_clip_unit(self.vehicle.steer_to_norm(steer_rad)),
            throttle=_clip_unit(throttle),
            mode=mode,
            v_target=max(0.0, _finite(v_target, 0.0)),
        )

    def _decide(self, scan, angles, pose_yaw, speed):
        if self.path is None or self.profile is None or pose_yaw is None:
            return self._gap(scan, angles)

        x, y, heading = pose_yaw
        index, offset = self.path.nearest((x, y))
        if offset > self.config.max_path_offset:
            # Not where we think we are, or no longer on the track. Either
            # way the line is not information any more.
            self._blocked_frames = self._clear_frames = 0
            self.mode = GAP
            return self._gap(scan, angles)

        self._update_mode(self._path_blocked(scan, angles, (x, y), heading, index, speed))

        if self.mode == GAP:
            return self._gap(scan, angles)

        if self.config.tracker == "pursuit":
            result = self.pursuit.step((x, y), heading, speed, self.path)
        else:
            result = self.tracker.step((x, y), heading, speed, self.path)
        # The profile's backward pass already guarantees this speed is slow
        # enough to make whatever is coming, so it is read at our own
        # position rather than at the lookahead point - braking for the
        # lookahead too would be braking twice for the same corner.
        return float(result.steer_rad), self._target_speed(index, speed), LINE

    def _target_speed(self, index, speed) -> float:
        """The profile's speed, read `speed_preview_s` of travel ahead.

        Latency is the reason: a brake command arrives at the wheels a few
        frames after it is computed, from a pose that was already a frame or
        two old, so reading the profile at our own position starts every
        braking zone late by that much.
        """
        preview = self.config.speed_preview_s
        if preview <= 0.0:
            return float(self.profile[index])
        s = (self.path.s[index] + max(speed, 0.0) * preview) % self.path.length
        ahead = int(np.searchsorted(self.path.s, s, side="right") - 1)
        return float(self.profile[max(ahead, 0)])

    def _gap(self, scan, angles):
        result = self.gap_follower.step(scan, angles)
        return float(result.steer_rad), float(result.v_target), GAP

    # -- is the line usable right now? -----------------------------------

    def _update_mode(self, blocked: bool) -> None:
        if blocked:
            self._blocked_frames += 1
            self._clear_frames = 0
        else:
            self._clear_frames += 1
            self._blocked_frames = 0

        if self.mode == LINE and self._blocked_frames >= self.config.block_enter_frames:
            self.mode = GAP
        elif self.mode == GAP and self._clear_frames >= self.config.block_exit_frames:
            self.mode = LINE

    def _path_blocked(self, scan, angles, pose_xy, heading, index, speed) -> bool:
        """Is anything sitting ON the path we intend to drive?

        Asks whether any LiDAR return lands within `block_radius` of a path
        sample ahead.

        Deliberately NOT a line-of-sight test. An occlusion test - "can I see
        the path point?" - looks equivalent and is not: on corners of about a
        metre radius, the straight chord from the car to a point one braking
        distance ahead leaves the corridor by geometry, so the wall on the
        outside of the corner occludes it. That reports a blockage at every
        corner entry on a completely clear track, which is where the speed
        profile is worth the most. It cost 26 seconds a lap before the
        surrogate caught it.
        """
        cfg = self.config
        braking = (speed * speed) / (2.0 * max(self.vehicle.a_brake_max, 1e-6))
        horizon = min(
            max(braking + cfg.block_min_horizon, cfg.block_min_horizon),
            cfg.gap.max_range,
        )

        offsets = np.linspace(0.5, horizon, max(int(cfg.block_samples), 1))
        points = np.array([self.path.point_at_s(self.path.s[index] + d) for d in offsets])

        # A beam sitting at max range saw nothing, so it is not an obstacle.
        hits = scan < cfg.gap.max_range - 1e-3
        if not np.any(hits):
            return False

        bearings = angles[hits] + heading
        obstacle_x = pose_xy[0] + scan[hits] * np.cos(bearings)
        obstacle_y = pose_xy[1] + scan[hits] * np.sin(bearings)

        dx = points[:, 0][:, None] - obstacle_x[None, :]
        dy = points[:, 1][:, None] - obstacle_y[None, :]
        return bool(np.any(dx * dx + dy * dy < cfg.block_radius ** 2))


# -- helpers ----------------------------------------------------------------

def _valid_pose(pose, yaw) -> Optional[Tuple[float, float, float]]:
    """(x, y, yaw) when all three are usable, otherwise None."""
    if pose is None or yaw is None:
        return None
    try:
        x, y = float(pose[0]), float(pose[1])
        heading = float(yaw)
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
        return None
    return x, y, heading


def _finite(value, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _clip_unit(value: float) -> float:
    value = _finite(value, 0.0)
    return -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
