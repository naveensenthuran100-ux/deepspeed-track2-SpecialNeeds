#!/usr/bin/env python3
"""What did the real car physically do? Grip, steering, throttle - from a realsim telemetry CSV.

    python3 lab/physics.py results/realsim/<tag>.telemetry.csv
"""
import csv, sys
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda v: float(v) if v not in ("", "None", "nan") else np.nan
    return {k: np.array([f(r[k]) for r in rows]) for k in rows[0]}


def main(path):
    d = load(path)
    lap = d["lap"]
    first = np.nanmin(lap)
    keep = lap > first                    # drop the standing-start lap
    v, ay, ax = d["speed"][keep], d["imu_ay"][keep], d["imu_ax"][keep]
    wz, st, thr = d["yaw_rate"][keep], d["steer"][keep], d["throttle"][keep]
    st_cmd, thr_cmd = d["steer_cmd"][keep], d["throttle_cmd"][keep]
    kin = v * wz                          # lateral accel from motion: v * yaw rate
    t = d["t"][keep]
    print(f"{keep.sum()} samples, {np.nanmedian(np.diff(t))*1000:.0f} ms median period")
    print(f"speed: max {np.nanmax(v):.2f}  p95 {np.nanpercentile(v,95):.2f}  mean {np.nanmean(v):.2f} m/s")
    print(f"lateral (v*yaw_rate): p90 {np.nanpercentile(np.abs(kin),90):.2f}  p99 {np.nanpercentile(np.abs(kin),99):.2f}  max {np.nanmax(np.abs(kin)):.2f}")
    print(f"lateral (IMU a_y):    p90 {np.nanpercentile(np.abs(ay),90):.2f}  p99 {np.nanpercentile(np.abs(ay),99):.2f}  max {np.nanmax(np.abs(ay)):.2f}")
    print(f"longitudinal IMU a_x: p01 {np.nanpercentile(ax,1):.2f}  p99 {np.nanpercentile(ax,99):.2f}")
    print(f"steer actual: |.|>0.95 {100*np.nanmean(np.abs(st)>0.95):.1f}%  >0.8 {100*np.nanmean(np.abs(st)>0.8):.1f}%   "
          f"steer cmd |.|>0.95 {100*np.nanmean(np.abs(st_cmd)>0.95):.1f}%")
    print(f"throttle cmd: =1.0 {100*np.nanmean(thr_cmd>0.99):.0f}%  <0 {100*np.nanmean(thr_cmd<0):.0f}%  <-0.9 {100*np.nanmean(thr_cmd<-0.9):.0f}%")
    # steering lag: actual vs commanded, best-fit delay in samples
    ok = np.isfinite(st) & np.isfinite(st_cmd)
    a, b = st[ok], st_cmd[ok]
    lags = [(k, np.corrcoef(a[k:], b[:len(b)-k])[0, 1]) for k in range(0, 8)]
    best = max(lags, key=lambda x: x[1])
    print(f"steer actual follows command after ~{best[0]} samples (r={best[1]:.3f}); "
          f"actual/cmd gain {np.polyfit(b[:len(b)-best[0]], a[best[0]:], 1)[0]:.2f}")
    # effective curvature per steer: yaw_rate / v vs steer actual
    m = ok & (v > 1.5)
    kap = wz[m] / v[m]
    gain = np.polyfit(st[m], kap, 1)[0]
    print(f"curvature per unit steer: {gain:.3f} 1/m  (full lock -> radius {1/abs(gain):.2f} m; "
          f"kinematic 0.324/tan(0.5236) = 0.561 m)")
    # understeer check: at high lateral accel, does curvature fall short of what steering asks?
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 30)):
        mm = m & (np.abs(v * wz) >= lo) & (np.abs(v * wz) < hi) & (np.abs(st) > 0.1)
        if mm.sum() > 5:
            ratio = np.nanmedian((wz[mm] / v[mm]) / (gain * st[mm]))
            print(f"  |a_lat| {lo:>2}-{hi:<2}: n={mm.sum():4d}  achieved/asked curvature {ratio:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
