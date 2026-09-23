#!/usr/bin/env python3
"""Generate the racing line from the map. Regenerates maps/*_raceline.csv.

    python3 -m team_driver.tools.build_raceline \
        --map ../../../maps/icra26_compete \
        --centreline ../../../maps/icra26_compete_centerline.csv \
        --out ../../../maps/icra26_compete_raceline.csv

Rule 36 requires precomputed artefacts to be reproducible from submitted
code. This is that code: one command, no hand-placed waypoints.
"""

import argparse
import os
import sys

import numpy as np

from ..core.profile import speed_profile
from ..core.track import Centreline, OccupancyGrid
from ..core.vehicle import Vehicle
from .line_optimise import optimise_raceline, save_raceline


def _lap_time(line, vehicle, grip, v_max):
    speeds = speed_profile(line.curvature, line.s, vehicle, grip,
                           v_max=v_max, total_length=line.length)
    steps = np.diff(np.concatenate((line.s, [line.length])))
    return float(np.sum(steps / speeds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="map path without extension")
    parser.add_argument("--centreline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margin", type=float, default=0.30,
                        help="clearance to keep beyond the car's half width, m")
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--grip", type=float, default=0.4,
                        help="grip fraction used only for the reported lap time")
    parser.add_argument("--v-max", type=float, default=6.0)
    args = parser.parse_args(argv)

    grid = OccupancyGrid.load(args.map + ".pgm", args.map + ".yaml")
    centre = Centreline.load(args.centreline)
    vehicle = Vehicle.default()

    line = optimise_raceline(centre, grid, margin_m=args.margin,
                             smoothing=args.smoothing)

    on_track = bool(np.all(grid.is_free_many(line.xy)))
    before = _lap_time(centre, vehicle, args.grip, args.v_max)
    after = _lap_time(line, vehicle, args.grip, args.v_max)

    print(f"centreline : {centre.length:6.2f} m   profile lap {before:6.2f} s")
    print(f"racing line: {line.length:6.2f} m   profile lap {after:6.2f} s "
          f"({before - after:+.2f} s)")
    print(f"every sample on a free cell: {on_track}")
    if not on_track:
        print("REFUSING to write a line that leaves the track", file=sys.stderr)
        return 1

    save_raceline(line, args.out,
                  source=f"margin {args.margin} m, smoothing {args.smoothing}, "
                         f"from {os.path.basename(args.centreline)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
