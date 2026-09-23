#!/usr/bin/env python3
"""Search configurations against the surrogate, then shortlist for the real sim.

A real 12-lap run costs about six minutes and cannot be parallelised far on
one laptop, which caps honest experimentation at roughly 25 configurations a
day. A surrogate lap costs about a quarter of a second. So the search happens
here and only the shortlist ever touches AutoDRIVE.

    python3 lab/sweep.py                          # the default grid
    python3 lab/sweep.py --top 5 --laps 3         # more laps per config
    python3 lab/sweep.py --quick                  # a small grid, for a smoke test

Ranked by net lap time - the raw time plus 10 s per collision - because that
is what the organisers score, and it already prices contact correctly.

What comes out of here is a hypothesis, not a result. Nothing is believed
until it has completed three full runs on the real simulator.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws", "src", "team_driver"))

from team_driver.core.planner import Planner, PlannerConfig          # noqa: E402
from team_driver.core.profile import speed_profile                   # noqa: E402
from team_driver.core.track import Centreline, OccupancyGrid         # noqa: E402
from team_driver.core.vehicle import Vehicle                         # noqa: E402
from team_driver.tools.line_optimise import optimise_raceline        # noqa: E402
from team_driver.tools.scoring import net_lap_time                   # noqa: E402
from team_driver.tools.surrogate import SurrogateSensor, simulate    # noqa: E402

MAPS = os.path.join(REPO, "maps")

DEFAULT_GRID = {
    "margin": [0.25, 0.35, 0.45],
    "grip": [0.30, 0.35, 0.40, 0.45],
    "k_lookahead": [0.25, 0.35],
    "curvature_cap": [0.6, 0.8],
}

QUICK_GRID = {
    "margin": [0.35],
    "grip": [0.35, 0.40],
    "k_lookahead": [0.35],
    "curvature_cap": [0.8],
}


def build_line(centre, grid, margin, cache):
    if margin not in cache:
        cache[margin] = optimise_raceline(centre, grid, margin_m=margin)
    return cache[margin]


def run_one(vehicle, grid, sensor, line, config, laps):
    planner = Planner(vehicle, config["planner"])
    planner.load_path(line, config["profile"])
    yaw = float(np.arctan2(line.tangent[0][1], line.tangent[0][0]))
    return simulate(
        planner, vehicle, grid,
        start=(line.xy[0][0], line.xy[0][1], yaw),
        distance_m=line.length * laps,
        max_time_s=60.0 * laps,
        sensor=sensor,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=os.path.join(REPO, "lab", "shortlist"))
    args = parser.parse_args(argv)

    grid_map = OccupancyGrid.load(
        os.path.join(MAPS, "icra26_compete.pgm"),
        os.path.join(MAPS, "icra26_compete.yaml"),
    )
    centre = Centreline.load(os.path.join(MAPS, "icra26_compete_centerline.csv"))
    vehicle = Vehicle.default()
    sensor = SurrogateSensor(grid_map, max_range=10.0, beam_count=181)

    if not vehicle.measured:
        print("WARNING: vehicle.json is UNMEASURED. Every number below is "
              "relative to assumed grip and braking. Run system identification "
              "before believing any of it.\n")

    space = QUICK_GRID if args.quick else DEFAULT_GRID
    keys = list(space)
    combinations = list(itertools.product(*(space[k] for k in keys)))

    lines = {}
    results = []
    started = time.perf_counter()

    for index, values in enumerate(combinations, 1):
        settings = dict(zip(keys, values))
        line = build_line(centre, grid_map, settings["margin"], lines)
        profile = speed_profile(
            line.curvature, line.s, vehicle,
            grip_fraction=settings["grip"], v_max=6.0, total_length=line.length,
        )
        planner_config = PlannerConfig(
            k_lookahead=settings["k_lookahead"],
            curvature_cap=settings["curvature_cap"],
        )
        outcome = run_one(
            vehicle, grid_map, sensor, line,
            {"planner": planner_config, "profile": profile}, args.laps,
        )

        per_lap = outcome.elapsed_s / args.laps if outcome.status == "COMPLETE" else None
        results.append({
            **settings,
            "status": outcome.status,
            "lap_s": per_lap,
            "collisions": outcome.collisions,
            "net_s": (net_lap_time(per_lap, outcome.collisions / args.laps)
                      if per_lap else None),
            "max_offset": float(outcome.trace["offset"].max()),
        })
        print(f"  [{index:3d}/{len(combinations)}] {settings}  "
              f"{outcome.status:9s} "
              f"{'%6.2fs' % per_lap if per_lap else '   -  '} "
              f"coll {outcome.collisions:2d}")

    finished = [r for r in results if r["net_s"] is not None]
    finished.sort(key=lambda r: r["net_s"])

    print(f"\n{len(finished)}/{len(results)} configurations completed, "
          f"{time.perf_counter() - started:.1f}s total\n")
    print(f"{'rank':>4} {'net/lap':>8} {'raw':>7} {'coll':>5} {'off':>5}  settings")
    for rank, row in enumerate(finished[: args.top], 1):
        print(f"{rank:>4} {row['net_s']:8.2f} {row['lap_s']:7.2f} "
              f"{row['collisions']:5d} {row['max_offset']:5.2f}  "
              + " ".join(f"{k}={row[k]}" for k in keys))

    os.makedirs(args.out, exist_ok=True)
    for rank, row in enumerate(finished[: args.top], 1):
        write_params(os.path.join(args.out, f"candidate_{rank}.yaml"), row, keys)
    with open(os.path.join(args.out, "sweep.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nShortlist written to {args.out}. Confirm on the real simulator:")
    print("  ./scripts/evaluate.sh --team candidate_1 --runs 3 "
          "--driver-params lab/shortlist/candidate_1.yaml")
    return 0


def write_params(path, row, keys):
    """A driver_params.yaml evaluate.sh can take directly, no rebuild needed."""
    with open(path, "w") as handle:
        handle.write(
            "# Generated by lab/sweep.py from a surrogate search.\n"
            "# A hypothesis, not a result: confirm over three full runs on the\n"
            "# real simulator before believing it.\n"
            f"# surrogate: {row['lap_s']:.2f} s/lap raw, {row['collisions']} "
            f"collisions, max cross-track {row['max_offset']:.2f} m\n"
            "#\n"
            f"# NOTE: margin={row['margin']} describes the racing line this was\n"
            "# searched against. Regenerate it with:\n"
            "#   python3 -m team_driver.tools.build_raceline --map maps/icra26_compete \\\n"
            "#       --centreline maps/icra26_compete_centerline.csv \\\n"
            f"#       --out maps/icra26_compete_raceline.csv --margin {row['margin']}\n"
            "driver:\n"
            "  ros__parameters:\n"
            "    line:\n"
            f"      grip_fraction: {row['grip']}\n"
            "    pursuit:\n"
            f"      k_lookahead: {row['k_lookahead']}\n"
            f"      curvature_cap: {row['curvature_cap']}\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
