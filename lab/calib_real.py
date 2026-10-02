#!/usr/bin/env python3
"""Does the surrogate reproduce the REAL runs? One row per clean realsim run.

    python3 lab/calib_real.py [--a-lat 9.0] [--delay 1]

For every realsim run that finished COMPLETE with healthy timing, rebuild the
exact driver configuration from its params yaml, simulate it with a truth car
that has the measured throttle map (from its vehicle_config) and the measured
grip (--a-lat), and compare the flying-lap mean with the real one.
"""
import argparse, glob, json, os, sys
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from dataclasses import replace                                              # noqa: E402
from team_driver.core.build import load_racing_stack                         # noqa: E402
from team_driver.core.params import ALL_GROUPS, build_configs                # noqa: E402
from team_driver.core.planner import Planner                                 # noqa: E402
from team_driver.core.track import OccupancyGrid                             # noqa: E402
from team_driver.core.vehicle import Vehicle                                 # noqa: E402
from team_driver.tools.surrogate import Perturbation, SurrogateSensor, simulate  # noqa: E402

GRID = OccupancyGrid.load(f"{REPO}/maps/icra26_compete.pgm", f"{REPO}/maps/icra26_compete.yaml")
SENSOR = SurrogateSensor(GRID, max_range=10.0, beam_count=181)


def host(path):
    return path.replace("/hackathon", REPO) if path else path


def flatten(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, pre + k + ".")
        else:
            yield pre + k, v


def setup(params_yaml):
    p = dict(flatten(yaml.safe_load(open(params_yaml))["driver"]["ros__parameters"]))
    vpath = host(p.get("vehicle_config") or "") or f"{REPO}/race_ws/src/team_driver/config/vehicle.json"
    belief = Vehicle.load(vpath)
    values = {name: p.get(name, default) for g in ALL_GROUPS for name, (_, default) in g.items()}
    cfg = build_configs(values)
    line_csv = host(p.get("line.raceline_csv") or "") or f"{REPO}/maps/icra26_compete_raceline.csv"
    path, profile = load_racing_stack(belief, line_csv, grip_fraction=float(p.get("line.grip_fraction", 0.7)),
                                      v_max=float(p.get("line.v_max", 12.0)),
                                      friction_circle=bool(p.get("line.friction_circle", False)))
    return belief, cfg, path, profile


def flying(params_yaml, a_lat, delay, laps=(2, 7)):
    belief, cfg, path, profile = setup(params_yaml)
    truth = replace(belief, throttle_v_coeffs=[24.6, 0.0], a_lat_max=a_lat)
    yaw0 = float(np.arctan2(path.tangent[0][1], path.tangent[0][0]))
    out = []
    for n in laps:
        planner = Planner(belief, cfg)
        planner.load_path(path, profile)
        r = simulate(planner, belief, GRID, start=(path.xy[0][0], path.xy[0][1], yaw0),
                     distance_m=path.length * n, max_time_s=40.0 * n, sensor=SENSOR,
                     truth_vehicle=truth, perturbation=Perturbation(command_delay_steps=delay))
        out.append(r)
    ok = all(r.status == "COMPLETE" for r in out)
    t = (out[1].elapsed_s - out[0].elapsed_s) / (laps[1] - laps[0]) if ok else float("nan")
    return t, out[1].collisions, out[1].status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-lat", type=float, default=9.0)
    ap.add_argument("--delay", type=int, default=1)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f"{REPO}/results/realsim/*.result.json"), key=os.path.getmtime):
        d = json.load(open(f))
        params = host(d.get("params", ""))
        if d.get("status") != "COMPLETE" or not d.get("mean_flying") or not os.path.exists(params):
            continue
        if (d.get("odom_gaps_100ms") or 0) > 3:
            continue
        sim, coll, status = flying(params, a.a_lat, a.delay)
        rows.append((d["tag"], d["mean_flying"], d.get("collisions"), sim, coll, status))
        print(f"{d['tag']:<28} real {d['mean_flying']:7.3f} (coll {d.get('collisions')})   "
              f"surrogate {sim:7.3f} (coll {coll}, {status})   err {100*(sim-d['mean_flying'])/d['mean_flying']:+5.1f}%",
              flush=True)
    real = np.array([r[1] for r in rows]); sim = np.array([r[3] for r in rows])
    ok = np.isfinite(sim)
    err = (sim[ok] - real[ok]) / real[ok]
    from scipy.stats import spearmanr
    print(f"\n{ok.sum()} runs: mean error {100*err.mean():+.1f}%, |error| {100*np.abs(err).mean():.1f}%, "
          f"rank correlation {spearmanr(real[ok], sim[ok]).correlation:.2f}")


if __name__ == "__main__":
    main()
