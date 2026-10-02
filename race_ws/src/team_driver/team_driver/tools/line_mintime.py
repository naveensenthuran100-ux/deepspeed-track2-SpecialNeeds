"""Minimum-lap-time racing line: entry, apex and exit planned for every corner.

    python3 -m team_driver.tools.line_mintime --out maps/icra26_compete_raceline.csv

The minimum-curvature line (line_optimise.py) rounds each corner as gently as
the corridor allows, but it is blind to two things that decide lap time here:

  * the corridor is wider than the centreline-normal bounds let it see. Around
    the inner hairpin there are ~3 m of open track beyond the spit tip, yet
    the curvature QP linearises about the centreline, caps every point within
    2 m of it, and takes the hairpin at R = 0.75 m hugging the tip;
  * the car's binding limit in the slow corners is STEERING LOCK, not grip.
    Measured on the simulator: grip saturates near 9 m/s^2, but the two tight
    hairpins saturated the steering even on clean runs (0.75 m needs 78% of
    lock before any correction), and every collision was the car running wide
    there at full lock.

So the line is optimised for lap time directly. A closed periodic spline is
shifted sideways from the centreline at `n_ctrl` control points; for each
candidate the spline is resampled uniformly, its exact curvature taken from
the spline derivatives, and the lap time computed with the same speed profile
the driver uses (friction circle on). Penalties keep it legal and drivable:

  * wall clearance >= half the car's width + margin everywhere, measured on
    the distance field of FREE cells - the grey outfield counts as wall, so a
    line through a gap between ducts (rule 11) is as penalised as one through
    a duct;
  * |curvature| <= kappa_max, leaving steering authority for corrections;
  * a small curvature-rate cost, because the steering has a finite rate.
"""

import argparse
import math
import os

import numpy as np

from ..core.profile import speed_profile
from ..core.track import Centreline, OccupancyGrid
from ..core.vehicle import Vehicle
from .line_optimise import CAR_HALF_WIDTH


class LapModel:
    """Everything needed to score a candidate line, built once."""

    def __init__(self, centre: Centreline, grid: OccupancyGrid, vehicle: Vehicle,
                 a_lat: float, margin_m: float, kappa_max: float, n_ctrl: int = 90,
                 spacing_m: float = 0.2, v_max: float = 18.0,
                 w_wall: float = 400.0, w_kappa: float = 50.0, w_rate: float = 0.002):
        from scipy import ndimage

        self.centre, self.vehicle = centre, vehicle
        self.a_lat, self.margin_m, self.kappa_max = a_lat, margin_m, kappa_max
        self.spacing_m, self.v_max = spacing_m, v_max
        self.w_wall, self.w_kappa, self.w_rate = w_wall, w_kappa, w_rate

        free = (255.0 - grid.values.astype(np.float64)) / 255.0 < grid.free_thresh
        self.field = ndimage.distance_transform_edt(free) * grid.resolution
        self.grid = grid
        self.need = CAR_HALF_WIDTH + margin_m

        # Control points evenly spaced along the centreline.
        self.ctrl_s = np.linspace(0.0, centre.length, n_ctrl, endpoint=False)
        self.base = np.array([centre.point_at_s(s) for s in self.ctrl_s])
        idx = np.array([int(np.argmin(np.abs(centre.s - s))) for s in self.ctrl_s])
        self.normal = centre.normal[idx]

    # -- geometry ------------------------------------------------------------
    def curve(self, offsets):
        """Uniformly resampled points, arc length, curvature and its rate."""
        from scipy.interpolate import CubicSpline

        pts = self.base + offsets[:, None] * self.normal
        closed = np.vstack([pts, pts[:1]])
        chord = np.r_[0.0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
        spline = CubicSpline(chord, closed, bc_type="periodic")
        # Arc length of the spline itself, then uniform samples along it.
        fine = np.linspace(0.0, chord[-1], 8 * len(pts), endpoint=False)
        d1 = spline(fine, 1)
        speed = np.hypot(d1[:, 0], d1[:, 1])
        arc = np.r_[0.0, np.cumsum(speed[:-1] * np.diff(fine))]
        length = arc[-1] + speed[-1] * (fine[1] - fine[0])
        count = max(int(length / self.spacing_m), 50)
        s = np.linspace(0.0, length, count, endpoint=False)
        u = np.interp(s, arc, fine)
        xy, d1, d2 = spline(u), spline(u, 1), spline(u, 2)
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(
            np.hypot(d1[:, 0], d1[:, 1]) ** 3, 1e-9)
        return xy, s, length, kappa

    def clearance(self, xy):
        from scipy import ndimage

        # OccupancyGrid stores row 0 at the BOTTOM (world y = origin), matching
        # world_to_grid; cell centres sit at +0.5, hence the -0.5.
        cols = (xy[:, 0] - self.grid.origin[0]) / self.grid.resolution - 0.5
        rows = (xy[:, 1] - self.grid.origin[1]) / self.grid.resolution - 0.5
        return ndimage.map_coordinates(self.field, [rows, cols], order=1, mode="constant", cval=0.0)

    # -- scoring -------------------------------------------------------------
    def lap(self, xy, s, length, kappa):
        vehicle = self.vehicle
        profile = speed_profile(kappa, s, vehicle, grip_fraction=self.a_lat / vehicle.a_lat_max,
                                v_max=self.v_max, total_length=length, friction_circle=True)
        ds = np.diff(np.r_[s, length])
        return float(np.sum(ds / profile)), profile

    def cost(self, offsets):
        xy, s, length, kappa = self.curve(offsets)
        time, _ = self.lap(xy, s, length, kappa)
        short = np.maximum(self.need - self.clearance(xy), 0.0)
        over = np.maximum(np.abs(kappa) - self.kappa_max, 0.0)
        rate = np.diff(np.r_[kappa, kappa[:1]]) / self.spacing_m
        return (time + self.w_wall * float(np.sum(short ** 2))
                + self.w_kappa * float(np.sum(over ** 2))
                + self.w_rate * float(np.sum(rate ** 2)) * self.spacing_m)


def initial_offsets(model: LapModel, line: Centreline) -> np.ndarray:
    """Offsets of an existing line from the centreline, at the control points."""
    out = np.zeros(len(model.ctrl_s))
    for k, (p, n) in enumerate(zip(model.base, model.normal)):
        i, _ = line.nearest(p)
        out[k] = float(np.dot(line.xy[i] - p, n))
    return out


def optimise(model: LapModel, start: np.ndarray, iters: int = 400, bound_m: float = 2.5):
    from scipy.optimize import minimize

    result = minimize(model.cost, start, method="L-BFGS-B",
                      bounds=[(-bound_m, bound_m)] * len(start),
                      options={"maxiter": iters, "eps": 1e-3})
    return result.x, result


def corners(s, kappa, profile, threshold=0.3):
    """Each corner as (entry s, apex s, exit s, min radius, entry v, apex v, exit v)."""
    hot = np.abs(kappa) > threshold
    out = []
    if not hot.any():
        return out
    n = len(s)
    start = int(np.argmin(hot)) if hot.all() is False else 0
    order = np.roll(np.arange(n), -start)
    run = []
    for i in list(order) + [order[0]]:
        if hot[i]:
            run.append(i)
        elif run:
            k = run[int(np.argmax(np.abs(kappa[run])))]
            out.append((s[run[0]], s[k], s[run[-1]], 1.0 / abs(kappa[k]),
                        profile[run[0]], profile[k], profile[run[-1]],
                        "L" if kappa[k] > 0 else "R"))
            run = []
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    p.add_argument("--map", default=os.path.join(root, "maps", "icra26_compete"))
    p.add_argument("--centreline", default=os.path.join(root, "maps", "icra26_compete_centerline.csv"))
    p.add_argument("--start", help="line to start from (default: the centreline)")
    p.add_argument("--vehicle", default=os.path.join(root, "race_ws", "src", "team_driver",
                                                     "config", "vehicle.json"))
    p.add_argument("--a-lat", type=float, default=8.0, help="cornering grip to plan with, m/s^2")
    p.add_argument("--margin", type=float, default=0.25, help="wall clearance beyond half the car")
    p.add_argument("--kappa-max", type=float, default=1.1, help="steering authority cap, 1/m")
    p.add_argument("--n-ctrl", type=int, default=90)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    grid = OccupancyGrid.load(a.map + ".pgm", a.map + ".yaml")
    centre = Centreline.load(a.centreline)
    vehicle = Vehicle.load(a.vehicle)
    model = LapModel(centre, grid, vehicle, a_lat=a.a_lat, margin_m=a.margin,
                     kappa_max=a.kappa_max, n_ctrl=a.n_ctrl)
    start = (initial_offsets(model, Centreline.load(a.start)) if a.start
             else np.zeros(len(model.ctrl_s)))
    before = model.lap(*model.curve(start))[0]
    best, result = optimise(model, start, iters=a.iters)
    xy, s, length, kappa = model.curve(best)
    time, profile = model.lap(xy, s, length, kappa)
    clear = model.clearance(xy)
    on_track = grid.is_free_many(xy)

    print(f"model lap {before:.3f} s -> {time:.3f} s   length {length:.2f} m   "
          f"(a_lat {a.a_lat}, margin {a.margin}, kappa_max {a.kappa_max})")
    print(f"min radius {1/np.abs(kappa).max():.2f} m   min wall clearance {clear.min():.3f} m "
          f"(need {model.need:.3f})   all on track: {bool(on_track.all())}   "
          f"optimiser: {result.message}")
    print(f"\n{'corner':>6} {'dir':>3} {'entry s':>8} {'apex s':>7} {'exit s':>7} {'R min':>6} "
          f"{'v entry':>8} {'v apex':>7} {'v exit':>7}")
    for k, c in enumerate(corners(s, kappa, profile), 1):
        print(f"{k:>6} {c[7]:>3} {c[0]:>8.1f} {c[1]:>7.1f} {c[2]:>7.1f} {c[3]:>6.2f} "
              f"{c[4]:>8.2f} {c[5]:>7.2f} {c[6]:>7.2f}")

    header = (f"x_m, y_m\nMinimum-lap-time racing line, team_driver/tools/line_mintime.py\n"
              f"a_lat {a.a_lat}, margin {a.margin}, kappa_max {a.kappa_max}, n_ctrl {a.n_ctrl}, "
              f"from {os.path.basename(a.start or a.centreline)}\n"
              f"Lap length {length:.2f} m, model lap {time:.3f} s")
    np.savetxt(a.out, xy, delimiter=",", header=header)
    print(f"\nwrote {a.out}")
    return 0 if on_track.all() and clear.min() >= model.need - 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
