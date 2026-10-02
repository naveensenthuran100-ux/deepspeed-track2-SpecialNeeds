#!/usr/bin/env python3
"""Cross-track error and speed vs distance, per lap: where does the car first lose the line?

    python3 lab/offsets.py out.png line.csv tel1.csv [tel2.csv ...]
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from team_driver.core.track import Centreline


def load(p):
    rows = list(csv.DictReader(open(p)))
    f = lambda v: float(v) if v not in ("", "None", "nan") else np.nan
    return {k: np.array([f(r[k]) for r in rows]) for k in rows[0]}


def main(out, line_path, *tels):
    line = Centreline.load(line_path)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for n, tp in enumerate(tels):
        d = load(tp)
        near = np.array([line.nearest((x, y)) for x, y in zip(d["x"], d["y"])])
        idx = near[:, 0].astype(int)
        tx, ty = line.tangent[idx, 0], line.tangent[idx, 1]
        dx, dy = d["x"] - line.xy[idx, 0], d["y"] - line.xy[idx, 1]
        signed = np.sign(tx * dy - ty * dx) * near[:, 1]
        s = line.s[idx]
        lap = d["lap"]
        name = os.path.basename(tp).replace(".telemetry.csv", "")
        for k, L in enumerate(np.unique(lap[np.isfinite(lap)])[1:]):     # skip standing start
            m = lap == L
            order = np.argsort(s[m])
            kw = dict(color=f"C{n}", alpha=0.7, lw=0.9, label=name if k == 0 else None)
            axes[0].plot(s[m], signed[m], ".", ms=1.5, **{k2: v for k2, v in kw.items() if k2 != "lw"})
            axes[1].plot(s[m], d["speed"][m], ".", ms=1.5, color=f"C{n}")
            axes[2].plot(s[m], np.abs(d["steer_cmd"][m]), ".", ms=1.5, color=f"C{n}")
        first = []
        for L in np.unique(lap[np.isfinite(lap)])[1:]:
            m = np.where((lap == L) & (np.abs(signed) > 0.35))[0]
            if len(m):
                first.append(round(float(s[m[0]]), 1))
        print(f"{name}: first point >0.35 m off the line, per lap: s = {first}")
    k = np.abs(line.curvature)
    ax2 = axes[0].twinx(); ax2.fill_between(line.s, 0, k, color="k", alpha=0.08); ax2.set_ylabel("|curvature|")
    axes[0].set_ylabel("cross-track (m, + left)"); axes[0].axhline(0, color="k", lw=0.5); axes[0].legend()
    axes[1].set_ylabel("speed m/s"); axes[2].set_ylabel("|steer cmd|"); axes[2].set_xlabel("s along line (m)")
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
