#!/usr/bin/env python3
"""Where does a real lap lose its time? Telemetry from record_run.py vs the plan.

    python3 lab/analyse_run.py results/telemetry_x.csv \
        --line lab/shortlist/cem/raceline_m0300.csv --grip 0.806 --v-max 11.965

Reports, from the REAL car:

  1. time lost per stretch of track against the planned speed profile, and
     whether that stretch is under-speed (the car could not or did not reach
     the plan) or the plan itself being slow;
  2. the throttle -> speed map, re-identified from straight-line running,
     against the one in vehicle.json that the feed-forward uses;
  3. acceleration and braking actually achieved, from the IMU;
  4. lateral acceleration actually used in the corners, from the IMU.
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws", "src", "team_driver"))

from team_driver.core.profile import speed_profile                   # noqa: E402
from team_driver.core.track import Centreline                        # noqa: E402
from team_driver.core.vehicle import Vehicle                         # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
            for k in rows[0]}
    return cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("telemetry")
    p.add_argument("--line", required=True)
    p.add_argument("--grip", type=float, required=True)
    p.add_argument("--v-max", type=float, required=True)
    p.add_argument("--bins", type=int, default=16)
    a = p.parse_args()

    d = load(a.telemetry)
    veh = Vehicle.load(os.path.join(REPO, "race_ws/src/team_driver/config/vehicle.json"))
    line = Centreline.load(a.line)
    prof = speed_profile(line.curvature, line.s, veh, grip_fraction=a.grip,
                         v_max=a.v_max, total_length=line.length)

    # ---- keep the race: from the first lap-counter change after the reset ----
    lap = d["lap_count"]
    t = d["t"] - d["t"][0]
    change = np.where(np.diff(np.nan_to_num(lap, nan=-1)) != 0)[0] + 1
    print(f"{len(t)} rows over {t[-1]:.1f}s, {len(change)} lap-counter changes")
    if len(change) < 3:
        print("not enough laps recorded"); return
    # First change is the referee's reset; laps are the spans between the rest.
    marks = change[1:]
    laps = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    print(f"{len(laps)} complete laps: " +
          " ".join(f"{t[e]-t[s]:.2f}" for s, e in laps))

    # skip the first two (warmup, standing start) when there are enough
    use = laps[2:] if len(laps) >= 5 else laps
    idx = np.concatenate([np.arange(s, e) for s, e in use])

    x, y, v = d["x"][idx], d["y"][idx], d["speed"][idx]
    tt = t[idx]
    near = np.array([line.nearest((xi, yi)) for xi, yi in zip(x, y)])
    k, off = near[:, 0].astype(int), near[:, 1]
    v_plan = prof[k]
    s = line.s[k]

    # ---- 1. time lost per stretch -------------------------------------------
    print(f"\n== 1. time per lap by stretch (mean over {len(use)} laps), real vs plan ==")
    edges = np.linspace(0, line.length, a.bins + 1)
    ds = np.diff(np.append(line.s, line.length))
    plan_t = np.array([np.sum((ds / prof)[(line.s >= lo) & (line.s < hi)])
                       for lo, hi in zip(edges[:-1], edges[1:])])
    dt = np.gradient(tt)
    dt = np.where((dt > 0) & (dt < 0.2), dt, np.nan)
    real_t = np.array([np.nansum(dt[(s >= lo) & (s < hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
    real_t /= len(use)
    kap = np.abs(line.curvature)
    print(f"{'s from':>7} {'to':>6} {'min R':>6} {'v plan':>7} {'v real':>7} {'plan s':>7} {'real s':>7} {'lost':>6}")
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        seg = (line.s >= lo) & (line.s < hi)
        m = (s >= lo) & (s < hi)
        print(f"{lo:>7.1f} {hi:>6.1f} {1/max(kap[seg].max(),1e-6):>6.2f} "
              f"{prof[seg].mean():>7.2f} {np.nanmean(v[m]) if m.any() else np.nan:>7.2f} "
              f"{plan_t[i]:>7.3f} {real_t[i]:>7.3f} {real_t[i]-plan_t[i]:>+6.3f}")
    print(f"{'lap':>28} {'':>15} {plan_t.sum():>7.3f} {real_t.sum():>7.3f} {real_t.sum()-plan_t.sum():>+6.3f}")

    under = v < v_plan - 0.3
    over = v > v_plan + 0.3
    print(f"\nshare of samples: under plan by >0.3 m/s {100*under.mean():.0f}%, "
          f"over by >0.3 {100*over.mean():.0f}%;  mean speed error "
          f"{np.mean(v - v_plan):+.2f} m/s;  |cross-track| mean {np.mean(np.abs(off)):.3f} "
          f"p95 {np.percentile(np.abs(off),95):.3f} m")

    # ---- 2. throttle -> speed, from the real car ----------------------------
    thr = d["throttle"][idx]
    acc = np.gradient(v, tt)
    steady = (np.abs(acc) < 0.6) & (np.abs(d["yaw_rate"][idx]) < 0.3) & (thr > 0.02) & (v > 1.0)
    print(f"\n== 2. throttle -> steady speed ({steady.sum()} near-steady straight samples) ==")
    if steady.sum() > 10:
        slope = np.sum(thr[steady] * v[steady]) / np.sum(thr[steady] ** 2)
        print(f"real: v ~= {slope:.2f} * u   (through the origin)")
        for u in (0.1, 0.2, 0.3, 0.4, 0.5):
            m = steady & (np.abs(thr - u) < 0.05)
            print(f"  u {u:.1f}: real {np.median(v[m]) if m.sum() > 3 else float('nan'):6.2f} "
                  f"(n={m.sum():3d})   vehicle.json {veh.speed_for_throttle(u):6.2f}")
    # what the feed-forward asked for vs what was needed
    print(f"feed-forward u for 10 m/s: map {veh.throttle_for_speed(10.0):.3f}")

    # ---- 3. longitudinal, from the IMU --------------------------------------
    ax = d["imu_ax"][idx]
    print("\n== 3. longitudinal acceleration (IMU), straight-ish samples ==")
    st = np.abs(d["yaw_rate"][idx]) < 0.5
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 13)):
        m = st & (v >= lo) & (v < hi) & (thr > 0.5)
        mb = st & (v >= lo) & (v < hi) & (thr < -0.3)
        print(f"  v {lo:>2}-{hi:<2}: accel at u>0.5 p90 {np.nanpercentile(ax[m],90) if m.sum()>5 else float('nan'):6.2f} (n={m.sum():3d})  "
              f"brake at u<-0.3 p10 {np.nanpercentile(ax[mb],10) if mb.sum()>5 else float('nan'):6.2f} (n={mb.sum():3d})")
    print(f"  vehicle.json: a_accel_max {veh.a_accel_max:.2f}, a_brake_max {veh.a_brake_max:.2f}")
    print(f"  throttle command share: >0.9 {100*np.mean(thr>0.9):.0f}%  <-0.3 {100*np.mean(thr<-0.3):.0f}%")

    # ---- 4. lateral, from the IMU -------------------------------------------
    ay = np.abs(d["imu_ay"][idx])
    ay_kin = np.abs(v * d["yaw_rate"][idx])
    print("\n== 4. lateral acceleration used ==")
    print(f"  |a_y| IMU p50 {np.nanpercentile(ay,50):.2f} p90 {np.nanpercentile(ay,90):.2f} "
          f"p99 {np.nanpercentile(ay,99):.2f} max {np.nanmax(ay):.2f}")
    print(f"  |v*yaw_rate| p90 {np.nanpercentile(ay_kin,90):.2f} p99 {np.nanpercentile(ay_kin,99):.2f}")
    print(f"  planned a_lat = grip*a_lat_max = {a.grip*veh.a_lat_max:.2f}")


if __name__ == "__main__":
    main()
