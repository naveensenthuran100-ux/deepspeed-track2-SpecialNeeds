#!/usr/bin/env python3
"""Steering weave on the straights, in the surrogate, for the shipped config.

    python3 lab/weave.py

The planner believes vehicle.json; the simulated car has the measured map and
grip, and the commands arrive `delay` 25 ms steps late. Reports the same
metrics measured on the real car (lab/ analysis of shipped_c83_12lap_*):
steering reversals/s and yaw-rate spread on the straights (R > 6 m).
"""
import dataclasses, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver")); sys.path.insert(0, os.path.join(REPO, "lab"))
from calib_real import GRID, SENSOR, setup                                  # noqa: E402
from team_driver.core.planner import Planner                                  # noqa: E402
from team_driver.tools.surrogate import Perturbation, simulate                # noqa: E402

YAML = f"{REPO}/race_ws/src/team_driver/config/driver_params.yaml"


def run(delay, **over):
    belief, cfg, path, profile = setup(YAML)
    truth = dataclasses.replace(belief, a_lat_max=9.0)
    cfg = dataclasses.replace(cfg, **over)
    yaw0 = float(np.arctan2(path.tangent[0][1], path.tangent[0][0]))
    p = Planner(belief, cfg); p.load_path(path, profile)
    r = simulate(p, belief, GRID, start=(path.xy[0][0], path.xy[0][1], yaw0),
                 distance_m=path.length * 4, max_time_s=60, sensor=SENSOR, truth_vehicle=truth,
                 perturbation=Perturbation(command_delay_steps=delay))
    tr = r.trace
    idx = np.array([path.nearest((x, y))[0] for x, y in zip(tr["x"], tr["y"])])
    straight = (np.abs(path.curvature[idx]) < 1 / 6.0) & (tr["t"] > 12.0) & (tr["v"] > 4.0)
    st = tr["steer"][straight]
    yaw_rate = np.gradient(np.unwrap(tr["yaw"]), tr["t"])[straight]
    dur = straight.sum() * 0.025
    x = st - np.convolve(st, np.ones(20) / 20, mode="same")          # remove the slow trend
    rev = np.sum(np.diff(np.sign(x[np.abs(x) > 0.005])) != 0) / max(dur, 1e-6)
    return r, rev, float(np.std(yaw_rate - np.convolve(yaw_rate, np.ones(20) / 20, mode="same")))


if __name__ == "__main__":
    print(f"{'delay':>5} {'setting':<32} {'status':>9} {'coll':>4} {'s/lap':>6} {'reversals/s':>11} {'yaw-rate wobble':>15}")
    for delay in (0, 2, 3):
        for name, over in (("shipped (no damping)", {}),
                           ("heading_speed_ref 4", {"heading_speed_ref": 4.0}),
                           ("k_yaw 0.3", {"k_yaw": 0.3}),
                           ("k_yaw 0.6", {"k_yaw": 0.6}),
                           ("ref 4 + k_yaw 0.3", {"heading_speed_ref": 4.0, "k_yaw": 0.3})):
            r, rev, wob = run(delay, **over)
            print(f"{delay:>5} {name:<32} {r.status:>9} {r.collisions:>4} {r.elapsed_s/4:6.2f} {rev:11.1f} {wob:15.3f}")
