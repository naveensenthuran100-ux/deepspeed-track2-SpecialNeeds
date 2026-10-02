#!/usr/bin/env python3
"""Where and why did a real run collide? Each contact, with the second before it.

    python3 lab/collisions.py results/realsim/<tag>.telemetry.csv lab/shortlist/cem/raceline_m0300.csv
"""
import csv, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from team_driver.core.track import Centreline


def main(tel, line_path):
    rows = list(csv.DictReader(open(tel)))
    f = lambda v: float(v) if v not in ("", "None", "nan") else np.nan
    d = {k: np.array([f(r[k]) for r in rows]) for k in rows[0]}
    line = Centreline.load(line_path)
    near = np.array([line.nearest((x, y)) for x, y in zip(d["x"], d["y"])])
    idx, off = near[:, 0].astype(int), near[:, 1]
    # signed offset: + left of the line
    tx, ty = line.tangent[idx, 0], line.tangent[idx, 1]
    dx, dy = d["x"] - line.xy[idx, 0], d["y"] - line.xy[idx, 1]
    signed = np.sign(tx * dy - ty * dx) * off
    kap = line.curvature[idx]
    c = np.nan_to_num(d["collisions"], nan=np.nanmin(d["collisions"]))
    events = np.where(np.diff(c) > 0)[0] + 1
    print(f"{len(events)} contacts.  |offset| overall p50 {np.median(off):.3f} p95 {np.percentile(off,95):.3f} m")
    print(f"{'s':>6} {'R line':>7} {'v':>5} {'off@hit':>8} {'off-0.5s':>8} {'max|steer_cmd| 1s':>18} {'thr_cmd':>7} {'turn':>5}")
    for e in events:
        w = (d["t"] > d["t"][e] - 1.0) & (d["t"] <= d["t"][e])
        w5 = np.argmin(np.abs(d["t"] - (d["t"][e] - 0.5)))
        turn = "L" if kap[e] > 0 else "R"
        side = "outside" if signed[e] * np.sign(kap[e]) < 0 else "inside"
        print(f"{line.s[idx[e]]:>6.1f} {1/max(abs(kap[e]),1e-3):>7.2f} {d['speed'][e]:>5.2f} "
              f"{signed[e]:>+8.3f} {signed[w5]:>+8.3f} {np.nanmax(np.abs(d['steer_cmd'][w])):>18.2f} "
              f"{d['throttle_cmd'][e]:>7.2f} {turn:>5} {side}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
