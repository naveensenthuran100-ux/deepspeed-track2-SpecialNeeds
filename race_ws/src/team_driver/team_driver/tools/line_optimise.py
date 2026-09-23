"""Minimum-curvature racing line from an occupancy grid.

A centreline driven at one speed is not a lap. The line that is - wide in,
apex, wide out - is shorter and, much more to the point, straighter: the
speed profile's cornering cap goes as sqrt(1/|k|), so taking curvature out of
the path is worth speed everywhere at once.

Method
------
Each sample of the line is the centreline point pushed sideways along its
normal by an unknown offset a_i:

    P_i = C_i + a_i * N_i

Discrete curvature is dominated by the second difference of position, so
minimising total squared curvature becomes an ordinary linear least-squares
problem in the offsets:

    minimise  || sum_i (P_{i-1} - 2 P_i + P_{i+1}) ||^2
    subject to  lower_i <= a_i <= upper_i

which is linear in a, convex, and solved exactly by a bounded least-squares
solve rather than by iterating. The bounds are the measured free distance to
the boundary either side, less the margin we insist on keeping.

This is the standard formulation (Heilmeier et al., "Minimum curvature
trajectory planning and control for an autonomous race car", 2019), reduced
to the part that earns its keep on a 54 m circuit.

Generating the line this way, rather than shipping waypoints as data, is
what rule 36 asks for and what rule 45 gives bonus marks for.
"""

from typing import Optional

import numpy as np

from ..core.track import Centreline, OccupancyGrid

#: Half the car's width. The margin a caller asks for is on top of this.
CAR_HALF_WIDTH = 0.135

#: Half-width of a pinned bound, in metres. Small enough to be no offset at
#: all, wide enough for a solver that insists on strict ordering.
_PIN = 1e-9


def optimise_offsets(centre: Centreline, lower, upper, smoothing: float = 0.0):
    """Lateral offsets minimising total squared curvature, within bounds.

    Args:
        centre: the path to move sideways.
        lower, upper: per-sample bounds on the offset, in metres. Negative is
            to the right, matching the left-hand normal.
        smoothing: optional weight pulling the line back towards the
            centreline. Zero means "purely minimum curvature".
    """
    from scipy.optimize import lsq_linear

    points = centre.xy
    normals = centre.normal
    count = len(points)

    # Rows are the x and y components of the second difference at each sample.
    design = np.zeros((2 * count, count), dtype=np.float64)
    target = np.zeros(2 * count, dtype=np.float64)

    previous = np.roll(np.arange(count), 1)
    following = np.roll(np.arange(count), -1)

    rows = np.arange(count)
    design[2 * rows, previous] += normals[previous, 0]
    design[2 * rows, rows] += -2.0 * normals[rows, 0]
    design[2 * rows, following] += normals[following, 0]

    design[2 * rows + 1, previous] += normals[previous, 1]
    design[2 * rows + 1, rows] += -2.0 * normals[rows, 1]
    design[2 * rows + 1, following] += normals[following, 1]

    second_difference = points[previous] - 2.0 * points + points[following]
    target[2 * rows] = -second_difference[:, 0]
    target[2 * rows + 1] = -second_difference[:, 1]

    if smoothing > 0.0:
        design = np.vstack((design, smoothing * np.eye(count)))
        target = np.concatenate((target, np.zeros(count)))

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # A bound pair that has crossed means the margin exceeds the track width
    # there; pin the line to the centre rather than failing the solve. The
    # bounded solver requires bounds that are STRICTLY ordered, so the pin is
    # a hair wide rather than exact - collapsing them onto a single value
    # raises, and takes the whole sweep with it.
    midpoint = 0.5 * (lower + upper)
    crossed = lower >= upper
    lower = np.where(crossed, midpoint - _PIN, lower)
    upper = np.where(crossed, midpoint + _PIN, upper)

    solution = lsq_linear(design, target, bounds=(lower, upper), method="bvls")
    return solution.x


def raceline_bounds(centre: Centreline, grid: OccupancyGrid, margin_m: float,
                    max_m: float = 2.0):
    """Per-sample offset bounds from the measured free distance either side."""
    widths = centre.half_widths(grid, max_m=max_m)
    keep_out = CAR_HALF_WIDTH + float(margin_m)
    upper = np.maximum(widths[:, 0] - keep_out, 0.0)      # towards the normal
    lower = -np.maximum(widths[:, 1] - keep_out, 0.0)     # away from it
    return lower, upper


def optimise_raceline(
    centre: Centreline,
    grid: OccupancyGrid,
    margin_m: float = 0.30,
    smoothing: float = 0.0,
    max_m: float = 2.0,
) -> Centreline:
    """The racing line, as a Centreline with the same sample count.

    Same count on purpose: a speed profile is computed per sample and the
    planner indexes both by the same index, so the two must agree.
    """
    lower, upper = raceline_bounds(centre, grid, margin_m, max_m=max_m)
    offsets = optimise_offsets(centre, lower, upper, smoothing=smoothing)
    return Centreline(centre.xy + offsets[:, None] * centre.normal)


def save_raceline(line: Centreline, path: str, source: str = "") -> None:
    header = (
        "x_m, y_m\n"
        "Minimum-curvature racing line, generated by "
        "team_driver/tools/line_optimise.py\n"
        f"{source}\n"
        f"Lap length {line.length:.2f} m"
    )
    np.savetxt(path, line.xy, delimiter=",", header=header)
