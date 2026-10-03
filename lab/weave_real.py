#!/usr/bin/env python3
"""Steering weave on the straights, and the result, for real-simulator runs.

    python3 lab/weave_real.py weave_A_1 weave_B_1 ...

Straights are where the shipped line's radius exceeds 6 m, out lap excluded.
  reversals/s   steering-command sign changes about its local mean
  yaw sd        yaw-rate spread, rad/s - the nose swinging
  drift p-p     median peak-to-peak cross-track excursion per straight, m
plus collisions, best lap and the 10-lap total (timed laps 3-12, +10 s per
collision in them) - the numbers the judges score.
"""
import csv, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from team_driver.core.track import Centreline                       # noqa: E402

LINE = Centreline.load(os.path.join(REPO, "maps/icra26_compete_raceline.csv"))
F = lambda v: float(v) if v not in ("", "None", "nan") else np.nan


def weave(tag):
    rows = list(csv.DictReader(open(f"{REPO}/results/realsim/{tag}.telemetry.csv")))
    d = {k: np.array([F(r[k]) for r in rows]) for k in rows[0]}
    near = np.array([LINE.nearest((x, y)) for x, y in zip(d["x"], d["y"])])
    idx = near[:, 0].astype(int)
    off = np.einsum("ij,ij->i", np.stack([d["x"] - LINE.xy[idx, 0], d["y"] - LINE.xy[idx, 1]], 1),
                    LINE.normal[idx])
    straight = (np.abs(LINE.curvature[idx]) < 1 / 6.0) & (d["lap"] > np.nanmin(d["lap"]))
    t, st = d["t"], d["steer_cmd"]
    segs, start = [], None
    for i, s in enumerate(straight):
        if s and start is None:
            start = i
        if (not s or i == len(straight) - 1) and start is not None:
            if t[i] - t[start] > 0.5:
                segs.append((start, i))
            start = None
    rev, dur, yaw, pp = 0, 0.0, [], []
    for a, b in segs:
        x = st[a:b] - np.nanmean(st[a:b])
        rev += int(np.sum(np.diff(np.sign(x[np.abs(x) > 0.02])) != 0))
        dur += t[b] - t[a]
        yaw.append(np.nanstd(d["yaw_rate"][a:b]))
        pp.append(np.ptp(off[a:b]))
    return rev / max(dur, 1e-6), float(np.mean(yaw)), float(np.median(pp)), float(np.nanmean(d["speed"][straight]))


def main(tags):
    print(f"{'run':<12} {'status':<10} {'v str':>5} {'rev/s':>5} {'yaw sd':>6} {'drift':>5} "
          f"{'coll':>4} {'best':>7} {'10-lap total':>12}")
    for tag in tags:
        path = f"{REPO}/results/realsim/{tag}.result.json"
        if not os.path.exists(path):
            print(f"{tag:<12} (no result yet)")
            continue
        d = json.load(open(path))
        timed = d.get("lap_times", [])[2:12]
        ct = sum(d.get("lap_collisions", [])[2:12])
        total = sum(timed) + 10 * ct if len(timed) == 10 else float("nan")
        try:
            r, y, p, v = weave(tag)
        except Exception:
            r = y = p = v = float("nan")
        print(f"{tag:<12} {d['status'][:10]:<10} {v:5.2f} {r:5.1f} {y:6.3f} {p:5.2f} "
              f"{d.get('collisions', 0):>4} {min(timed) if timed else float('nan'):7.3f} {total:12.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
