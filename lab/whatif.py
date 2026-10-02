#!/usr/bin/env python3
"""Offline what-if: each controller change, against the MEASURED car.

The planner keeps a belief (vehicle.json, or a corrected map); the surrogate's
car is the one the sysid points actually describe (v = 24.6 u). Lap time is
flying laps only: (12 laps - 2 laps) / 10.
"""
import os, sys
from dataclasses import replace
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from team_driver.core.planner import Planner, PlannerConfig
from team_driver.core.profile import speed_profile
from team_driver.core.track import Centreline, OccupancyGrid
from team_driver.core.vehicle import Vehicle
from team_driver.tools.surrogate import SurrogateSensor, simulate

grid = OccupancyGrid.load(f"{REPO}/maps/icra26_compete.pgm", f"{REPO}/maps/icra26_compete.yaml")
sensor = SurrogateSensor(grid, max_range=10.0, beam_count=181)
OLD = Vehicle.load(f"{REPO}/race_ws/src/team_driver/config/vehicle.json")
TRUTH = replace(OLD, throttle_v_coeffs=[24.6, 0.0])
FIXED = replace(OLD, throttle_v_coeffs=[24.6, 0.0])           # belief with the right map
C480 = dict(tracker="feedforward", tau_preview=0.1892, min_preview=0.5247, k_cross=1.9583,
            k_heading=0.8418, lift_band=0.201, brake_gain=1.4297, kp=0.1041, ki=0.3236,
            curvature_cap=1.0107)
line = Centreline.load(f"{REPO}/lab/shortlist/cem/raceline_m0300.csv")
yaw0 = float(np.arctan2(line.tangent[0][1], line.tangent[0][0]))
START = (line.xy[0][0], line.xy[0][1], yaw0)


def flying(belief, cfg, grip=0.8061, v_max=11.965):
    prof = speed_profile(line.curvature, line.s, belief, grip_fraction=grip, v_max=v_max,
                         total_length=line.length)
    out = []
    for n in (2, 12):
        p = Planner(belief, PlannerConfig(**cfg)); p.load_path(line, prof)
        r = simulate(p, belief, grid, start=START, distance_m=line.length * n,
                     max_time_s=40 * n, sensor=sensor, truth_vehicle=TRUTH)
        out.append(r)
    ok = all(r.status == "COMPLETE" for r in out)
    return ((out[1].elapsed_s - out[0].elapsed_s) / 10.0 if ok else float("nan")), out[1].collisions


if __name__ == "__main__":
    cases = [
        ("#480 as raced (real: 12.44)", OLD, C480),
        ("+ full throttle band 0.3", OLD, dict(C480, full_throttle_band=0.3)),
        ("+ correct map belief", FIXED, C480),
        ("+ map + band 0.3", FIXED, dict(C480, full_throttle_band=0.3)),
        ("+ map + band + preview 0.05", FIXED, dict(C480, full_throttle_band=0.3, speed_preview_s=0.05)),
        ("+ map + band + preview 0.10", FIXED, dict(C480, full_throttle_band=0.3, speed_preview_s=0.10)),
    ]
    for name, belief, cfg in cases:
        t, c = flying(belief, cfg)
        print(f"{name:<32} {t:7.3f} s/lap   collisions(12 laps) {c}")
