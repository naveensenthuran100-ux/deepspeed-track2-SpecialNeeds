"""Model-predictive steering: plan the next second, apply the first step.

Why: the feed-forward tracker (core/tracker.py) corrects the error it sees
now. On the real car that error is already 40-60 ms old when the command
lands (odometry age + actuation), and at 6-7.5 m/s that is 25-45 cm of
travel. The result, measured: a ~1 Hz weave on the straights and ~0.45 m
drifts in the fast sections - the drifts are what still causes collisions,
and what stops more cornering grip being usable.

The model, in path coordinates, linearised about the line:

    e[k+1]   = e[k]   + v_k dt psi[k]                  lateral error, + left
    psi[k+1] = psi[k] + v_k dt (tan(delta_k) - L kappa_k) / L
                                                       heading error vs path

with tan(delta) linearised about the feed-forward angle delta0 = atan(L kappa)
- the steering the corner itself requires. Writing delta = delta0 + w, the
curvature term cancels exactly and

    psi[k+1] = psi[k] + b_k w_k,      b_k = v_k dt / (L cos^2 delta0_k)

so the controller decides only the DEVIATION w from the feed-forward. Speeds
v_k come from the speed profile along the horizon, so the prediction knows
the car will brake for the corner ahead.

Delay is modelled rather than fought: the first `delay_steps` of the horizon
use the command already sent (it is in flight and cannot be changed), and the
command returned is the first one the car will actually receive.

Cost: q_e e^2 + q_psi psi^2 + r_du (delta_k - delta_{k-1})^2 + r_w w^2 over
the horizon - track the line, smoothly. With no constraints inside the
horizon this is a small linear least-squares problem (horizon x horizon), a
fraction of a millisecond in numpy; the steering lock is applied to the
returned command.
"""

import math
from typing import Optional

import numpy as np

from .track import Centreline
from .tracker import TrackerResult
from .vehicle import Vehicle


class MPCTracker:
    def __init__(self, vehicle: Vehicle, horizon: int = 20, dt: float = 0.05,
                 q_e: float = 4.0, q_psi: float = 1.0, r_du: float = 5.0, r_w: float = 0.05,
                 delay_steps: int = 1, min_speed: float = 0.5):
        self.vehicle = vehicle
        self.horizon = max(int(horizon), 2)
        self.dt = float(dt)
        self.q_e, self.q_psi = float(q_e), float(q_psi)
        self.r_du, self.r_w = float(r_du), float(r_w)
        self.delay_steps = max(int(delay_steps), 0)
        self.min_speed = float(min_speed)
        self.profile: Optional[np.ndarray] = None
        self.last_command = 0.0

    def set_profile(self, profile) -> None:
        self.profile = None if profile is None else np.asarray(profile, dtype=np.float64)

    def reset(self) -> None:
        self.last_command = 0.0

    def step(self, pose_xy, yaw: float, speed: float, path: Centreline,
             yaw_rate=None) -> TrackerResult:
        L = self.vehicle.wheelbase
        lock = self.vehicle.max_steer_rad
        speed = float(speed) if math.isfinite(float(speed)) else 0.0
        index, _ = path.nearest(pose_xy)

        tangent = path.tangent[index]
        path_heading = math.atan2(tangent[1], tangent[0])
        psi0 = _wrap(float(yaw) - path_heading)                  # + = pointing left of the path
        offset = np.asarray(pose_xy, dtype=np.float64) - path.xy[index]
        e0 = float(np.dot(offset, path.normal[index]))           # + = left of the path

        # -- the horizon along the path: where the car will be, how fast ----
        n, dt = self.horizon, self.dt
        s = float(path.s[index])
        v = np.empty(n)
        kappa = np.empty(n)
        current = max(speed, self.min_speed)
        for k in range(n):
            planned = self._profile_at(path, s)
            # blend from the measured speed into the profile over ~0.3 s
            alpha = min(k * dt / 0.3, 1.0)
            v[k] = max((1 - alpha) * current + alpha * (planned if planned else current), self.min_speed)
            kappa[k] = _curvature_at(path, s)
            s += v[k] * dt
        delta0 = np.arctan(L * kappa)
        b = v * dt / (L * np.cos(delta0) ** 2)

        # -- prediction: x_k = free response + G w (w = deviations) --------
        # e and psi at steps 1..n as linear functions of w_0..w_{n-1}.
        d = min(self.delay_steps, n - 1)
        G_psi = np.zeros((n, n))
        G_e = np.zeros((n, n))
        f_psi = np.zeros(n)
        f_e = np.zeros(n)
        psi, e = psi0, e0
        Gp = np.zeros(n)                                          # d psi_k / d w
        Ge = np.zeros(n)                                          # d e_k / d w
        for k in range(n):
            e_next = e + v[k] * dt * psi
            Ge_next = Ge + v[k] * dt * Gp
            if k < d:
                psi_next = psi + b[k] * (self.last_command - delta0[k])   # in flight
                Gp_next = Gp.copy()
            else:
                psi_next = psi
                Gp_next = Gp.copy()
                Gp_next[k] += b[k]
            psi, e, Gp, Ge = psi_next, e_next, Gp_next, Ge_next
            f_psi[k], f_e[k] = psi, e
            G_psi[k], G_e[k] = Gp, Ge

        # -- least squares: tracking + smoothness + small effort ------------
        free = np.arange(d, n)                                    # decision variables
        rows, rhs = [], []
        se, sp = math.sqrt(self.q_e), math.sqrt(self.q_psi)
        rows.append(se * G_e[:, free]);   rhs.append(-se * f_e)
        rows.append(sp * G_psi[:, free]); rhs.append(-sp * f_psi)
        # steering rate: (delta0_k + w_k) - (delta0_{k-1} + w_{k-1})
        m = len(free)
        D = np.zeros((m, m))
        c = np.zeros(m)
        for j, k in enumerate(free):
            D[j, j] = 1.0
            if j == 0:
                c[j] = delta0[k] - self.last_command                 # vs the command in flight
            else:
                D[j, j - 1] = -1.0
                c[j] = delta0[k] - delta0[k - 1]
        rd = math.sqrt(self.r_du)
        rows.append(rd * D); rhs.append(-rd * c)
        rw = math.sqrt(self.r_w)
        rows.append(rw * np.eye(m)); rhs.append(np.zeros(m))
        A = np.vstack(rows)
        y = np.concatenate(rhs)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)

        command = float(np.clip(delta0[d] + w[0], -lock, lock))
        self.last_command = command
        return TrackerResult(
            steer_rad=command,
            preview_m=float(v[:d + 1].sum() * dt),
            cross_track_m=e0,
            heading_error_rad=-psi0,                              # tracker convention: path - yaw
            kappa_preview=float(kappa[d]),
        )

    def _profile_at(self, path: Centreline, s: float) -> Optional[float]:
        if self.profile is None:
            return None
        s = s % path.length
        i = int(np.searchsorted(path.s, s, side="right") - 1)
        return float(self.profile[max(i, 0)])


def _curvature_at(path: Centreline, s: float) -> float:
    s = float(s) % path.length
    knots = np.concatenate((path.s, [path.length]))
    closed = np.concatenate((path.curvature, path.curvature[:1]))
    return float(np.interp(s, knots, closed))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
