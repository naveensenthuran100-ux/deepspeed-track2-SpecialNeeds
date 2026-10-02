#!/usr/bin/env python3
"""Draw the occupancy map with racing lines on it.

    python3 lab/draw_track.py out.png line1.csv [line2.csv ...]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws/src/team_driver"))
from team_driver.core.track import Centreline

RES, ORIGIN = 0.05, (-7.960, -13.230)


def main(out, *lines, profiles=None):
    im = np.array(Image.open(os.path.join(REPO, "maps/icra26_compete.pgm")))
    h, w = im.shape
    extent = (ORIGIN[0], ORIGIN[0] + w * RES, ORIGIN[1], ORIGIN[1] + h * RES)
    fig, ax = plt.subplots(figsize=(8, 15), dpi=110)
    shade = np.where(im == 0, 0.0, np.where(im == 254, 1.0, 0.75))
    ax.imshow(shade, cmap="gray", extent=extent, origin="upper", vmin=0, vmax=1)
    colours = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for n, path in enumerate(lines):
        line = Centreline.load(path)
        xy = np.vstack([line.xy, line.xy[:1]])
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.4, color=colours[n % 5],
                label=f"{os.path.basename(path)}  {line.length:.2f} m")
        if n == 0:
            for s in range(0, int(line.length), 5):
                i = int(np.searchsorted(line.s, s))
                ax.annotate(f"{s}", line.xy[i], fontsize=7, color=colours[0])
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("icra26_compete: black = ducts, white = track, grey = outside")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
