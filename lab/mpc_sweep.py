#!/usr/bin/env python3
"""MPC vs the feed-forward tracker in the delay-calibrated surrogate.

    python3 lab/mpc_sweep.py

Shipped config otherwise. Truth car: measured map, 9 m/s^2 grip; commands
arrive `delay` 25 ms steps late. Reports lap, collisions, steering reversals
and yaw-rate wobble on the straights, and the p95 / max distance off the line.
"""
import dataclasses, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver")); sys.path.insert(0, os.path.join(REPO, "lab"))
from calib_real import GRID, SENSOR, setup                                    # noqa: E402
from team_driver.core.planner import Planner                                    # noqa: E402
from team_driver.tools.surrogate import Perturbation, simulate                  # noqa: E402

YAML = f"{REPO}/race_ws/src/team_driver/config/driver_params.yaml"


def evaluate(delay=2, laps=4, grip_fast=None, **over):
    belief, cfg, path, profile = setup(YAML)
    if grip_fast is not None:
        from team_driver.core.build import load_racing_stack
        import yaml
        line = yaml.safe_load(open(YAML))["driver"]["ros__parameters"]["line"]
        path, profile = load_racing_stack(belief, f"{REPO}/maps/icra26_compete_raceline.csv",
                                          grip_fraction=line["grip_fraction"], v_max=line["v_max"],
                                          friction_circle=True, grip_fraction_fast=grip_fast,
                                          fast_radius=line["fast_radius"], fast_buffer_m=line["fast_buffer_m"])
    cfg = dataclasses.replace(cfg, **over)
    truth = dataclasses.replace(belief, a_lat_max=9.0)
    yaw0 = float(np.arctan2(path.tangent[0][1], path.tangent[0][0]))
    p = Planner(belief, cfg); p.load_path(path, profile)
    r = simulate(p, belief, GRID, start=(path.xy[0][0], path.xy[0][1], yaw0), distance_m=path.length * laps,
                 max_time_s=40 * laps, sensor=SENSOR, truth_vehicle=truth,
                 perturbation=Perturbation(command_delay_steps=delay))
    tr = r.trace
    near = np.array([path.nearest((x, y)) for x, y in zip(tr["x"], tr["y"])])
    idx, off = near[:, 0].astype(int), near[:, 1]
    late = tr["t"] > 12.0
    straight = (np.abs(path.curvature[idx]) < 1 / 6.0) & late & (tr["v"] > 4.0)
    st = tr["steer"][straight]
    x = st - np.convolve(st, np.ones(20) / 20, mode="same")
    rev = np.sum(np.diff(np.sign(x[np.abs(x) > 0.005])) != 0) / max(straight.sum() * 0.025, 1e-6)
    yr = np.gradient(np.unwrap(tr["yaw"]), tr["t"])[straight]
    wob = float(np.std(yr - np.convolve(yr, np.ones(20) / 20, mode="same")))
    fast = late & (tr["v"] > 5.5)
    return (r.status, r.collisions, r.elapsed_s / laps, rev, wob,
            float(np.percentile(off[fast], 95)) if fast.any() else float("nan"),
            float(off[late].max()))


if __name__ == "__main__":
    print(f"{'config':<44} {'status':>9} {'coll':>4} {'s/lap':>6} {'rev/s':>5} {'wobble':>6} {'off p95':>7} {'off max':>7}")
    cases = [("feed-forward (shipped)", {}),
             ("MPC default (qe4 qpsi1 rdu2)", {"tracker": "mpc"}),
             ("MPC qe8", {"tracker": "mpc", "mpc_q_e": 8.0}),
             ("MPC qe8 rdu1", {"tracker": "mpc", "mpc_q_e": 8.0, "mpc_r_du": 1.0}),
             ("MPC qe4 rdu5", {"tracker": "mpc", "mpc_r_du": 5.0}),
             ("MPC delay 2 steps", {"tracker": "mpc", "mpc_delay_steps": 2}),
             ("MPC horizon 30", {"tracker": "mpc", "mpc_horizon": 30})]
    for delay in (2, 3):
        for name, over in cases:
            res = evaluate(delay=delay, **over)
            print(f"{name + f' [delay {delay}]':<44} {res[0]:>9} {res[1]:>4} {res[2]:6.2f} {res[3]:5.1f} {res[4]:6.3f} {res[5]:7.3f} {res[6]:7.3f}")
