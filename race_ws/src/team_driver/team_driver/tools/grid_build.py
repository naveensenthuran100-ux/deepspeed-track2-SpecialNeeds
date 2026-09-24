"""Build an occupancy grid by ray-carving recorded LiDAR against known pose.

The simulator ships the circuit as Unity geometry, not as a grid, so the map
in maps/ was traced rather than exported. Building a better one from our own
laps is worth bonus marks (rule 45), and is worth having regardless: the
shipped grid was carved from a reactive driver's wandering, so it is accurate
but does not cover the parts of the track a fast line actually uses.

Method: every beam carves the cells it passed through as free and marks the
cell it stopped in as occupied. Cells no beam ever reached stay unknown - a
distinction that matters, because a planner told the outfield is free space
cheerfully cuts the corner across it.

    ros2 run team_driver map_record                 # inside the container
    python3 -m team_driver.tools.grid_build --log results/mapping.npz \
        --out maps/my_circuit
"""

import argparse
from typing import Optional

import numpy as np

from ..core.track import OccupancyGrid

FREE, BLOCKED, UNKNOWN = 254, 0, 205

#: A cell needs this many more free observations than occupied ones to count
#: as free. Above 1.0 because a single stray return should not carve a hole in
#: a wall, and the boundary here is a row of thin ducts.
FREE_BIAS = 2.0


def build_grid(poses, scans, angles, resolution=0.05, max_range=10.0,
               padding_m=0.5) -> Optional[OccupancyGrid]:
    """An OccupancyGrid from (N,3) poses and (N,B) scans. None if there is nothing.

    Args:
        poses: (x, y, yaw) per scan, in the world frame.
        scans: matching ranges.
        angles: beam angles, in the car's frame.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    angles = np.asarray(angles, dtype=np.float64).ravel()
    # Checked before the reshape below: reshape(0, -1) is ambiguous and
    # raises, which is a crash rather than the "nothing recorded" answer.
    if poses.shape[0] == 0 or angles.shape[0] == 0:
        return None
    scans = np.asarray(scans, dtype=np.float64).reshape(poses.shape[0], -1)
    if scans.shape[1] == 0:
        return None

    hit = scans < max_range - 1e-3
    bearings = angles[None, :] + poses[:, 2:3]
    ends_x = poses[:, 0:1] + scans * np.cos(bearings)
    ends_y = poses[:, 1:2] + scans * np.sin(bearings)

    min_x = min(poses[:, 0].min(), ends_x.min()) - padding_m
    max_x = max(poses[:, 0].max(), ends_x.max()) + padding_m
    min_y = min(poses[:, 1].min(), ends_y.min()) - padding_m
    max_y = max(poses[:, 1].max(), ends_y.max()) + padding_m

    width = max(int(np.ceil((max_x - min_x) / resolution)), 1)
    height = max(int(np.ceil((max_y - min_y) / resolution)), 1)
    origin = (float(min_x), float(min_y))

    free_count = np.zeros((height, width), dtype=np.int32)
    occupied_count = np.zeros((height, width), dtype=np.int32)

    def index(xs, ys):
        cols = np.floor((xs - origin[0]) / resolution).astype(np.int64)
        rows = np.floor((ys - origin[1]) / resolution).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        return rows[inside], cols[inside]

    steps = np.arange(0.0, max_range, resolution)

    for scan_index in range(poses.shape[0]):
        x, y, _ = poses[scan_index]
        ranges = scans[scan_index]
        cos_b = np.cos(bearings[scan_index])
        sin_b = np.sin(bearings[scan_index])

        # Carve free space: every sample short of the return, by half a cell
        # so the hit cell itself is not also claimed free.
        along = steps[None, :]
        reachable = along < (ranges[:, None] - resolution * 0.5)
        sample_x = (x + along * cos_b[:, None])[reachable]
        sample_y = (y + along * sin_b[:, None])[reachable]
        rows, cols = index(sample_x, sample_y)
        np.add.at(free_count, (rows, cols), 1)

        # Mark what stopped the beam.
        stopped = hit[scan_index]
        if np.any(stopped):
            rows, cols = index(ends_x[scan_index][stopped], ends_y[scan_index][stopped])
            np.add.at(occupied_count, (rows, cols), 1)

    values = np.full((height, width), UNKNOWN, dtype=np.uint8)
    seen = (free_count > 0) | (occupied_count > 0)
    values[seen & (free_count > FREE_BIAS * occupied_count)] = FREE
    values[seen & (free_count <= FREE_BIAS * occupied_count)] = BLOCKED

    return OccupancyGrid(values, resolution, origin)


def save_grid(grid: OccupancyGrid, stem: str) -> None:
    """Write `stem`.pgm and `stem`.yaml in the format map_server expects."""
    # The grid is stored bottom-up; PGM files are top-down.
    image = np.flipud(grid.values)
    with open(stem + ".pgm", "wb") as handle:
        handle.write(b"P5\n")
        handle.write(f"{grid.width} {grid.height}\n255\n".encode())
        handle.write(image.astype(np.uint8).tobytes())

    with open(stem + ".yaml", "w") as handle:
        handle.write(
            "# Occupancy grid built by team_driver/tools/grid_build.py from our\n"
            "# own recorded laps: ground-truth pose plus the LiDAR, ray-carved.\n"
            "#\n"
            "# free_thresh is 0.196, not the ROS default of 0.25, and that is\n"
            "# deliberate: the three values are 0 for the boundary, 254 for the\n"
            "# track and 205 for everything outside it, and map_server turns a\n"
            "# pixel into a probability with (255 - value) / 255. 205 gives\n"
            "# exactly 0.19607..., so it lands in the unknown band. At 0.25 the\n"
            "# outfield reads as free space and a planner cuts across it.\n"
            f"image: {stem.split('/')[-1]}.pgm\n"
            "mode: trinary\n"
            f"resolution: {grid.resolution}\n"
            f"origin: [{grid.origin[0]:.4f}, {grid.origin[1]:.4f}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="npz written by map_record")
    parser.add_argument("--out", required=True, help="output stem, no extension")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-range", type=float, default=10.0)
    args = parser.parse_args(argv)

    data = np.load(args.log)
    grid = build_grid(data["poses"], data["scans"], data["angles"],
                      resolution=args.resolution, max_range=args.max_range)
    if grid is None:
        print(f"{args.log} contains no scans.")
        return 1

    free = int(np.sum(grid.values == FREE))
    save_grid(grid, args.out)
    print(f"{grid.width} x {grid.height} at {grid.resolution} m, "
          f"origin {grid.origin}, {free} free cells "
          f"({free * grid.resolution ** 2:.1f} m2 of track)")
    print(f"wrote {args.out}.pgm and {args.out}.yaml")
    print("Check it before trusting it: ./scripts/track_tool.py validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
