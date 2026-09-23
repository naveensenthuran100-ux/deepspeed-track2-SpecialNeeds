"""The circuit, as geometry we can plan against.

A scored run never opens a map - the simulator owns the circuit, the lap
counter and the collision detection (docs/05-evaluation.md 5.3). This module
exists for the other half of the problem: the LiDAR sees 10 m of a 54 m lap,
so the only way to brake for a corner *before* it is to know it is there.

The grid in maps/ was traced from the simulator by the organisers. Its
`free_thresh` of 0.196 is deliberate and load-bearing: the three pixel values
are 0 for the boundary, 254 for the track surface and 205 for everything
outside it, and map_server turns a pixel into a probability with
(255 - value) / 255, so 205 gives 0.19607... At the ROS default of 0.25 the
outfield reads as free space and a planner cheerfully cuts the corner across
it.
"""

import math
import os
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


class OccupancyGrid:
    """An occupancy grid in the simulator's world frame.

    Indexing is (row, col) with row measured from the BOTTOM, matching the ROS
    map convention where `origin` is the lower-left corner. PGM files store
    rows top-down, so the image is flipped once at load and never again.
    """

    def __init__(self, values, resolution, origin, free_thresh=0.196, occupied_thresh=0.65):
        self.values = np.asarray(values, dtype=np.uint8)
        self.height, self.width = self.values.shape
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.free_thresh = float(free_thresh)
        self.occupied_thresh = float(occupied_thresh)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, pgm_path: str, yaml_path: Optional[str] = None) -> "OccupancyGrid":
        if yaml_path is None:
            yaml_path = os.path.splitext(pgm_path)[0] + ".yaml"
        import yaml  # already in the judging image; not needed to import core

        with open(yaml_path, "r") as handle:
            meta = yaml.safe_load(handle)

        values = _read_pgm(pgm_path)
        return cls(
            values=np.flipud(values),          # file is top-down, we want bottom-up
            resolution=meta["resolution"],
            origin=meta["origin"],
            free_thresh=meta.get("free_thresh", 0.196),
            occupied_thresh=meta.get("occupied_thresh", 0.65),
        )

    # -- frames ----------------------------------------------------------

    def world_to_grid(self, xy) -> Tuple[int, int]:
        """(x, y) in metres -> (row, col). Not bounds-checked; see `is_free`."""
        col = int(math.floor((float(xy[0]) - self.origin[0]) / self.resolution))
        row = int(math.floor((float(xy[1]) - self.origin[1]) / self.resolution))
        return row, col

    def grid_to_world(self, rc) -> Tuple[float, float]:
        """(row, col) -> the centre of that cell, in metres."""
        row, col = rc
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    # -- occupancy -------------------------------------------------------

    def is_free_value(self, value) -> bool:
        """Free per the ROS trinary rule: p = (255 - value) / 255 < free_thresh."""
        return bool((255.0 - float(value)) / 255.0 < self.free_thresh)

    def is_free(self, xy) -> bool:
        """Is this world point on the track surface? Off-grid is not free.

        Explicitly bounds-checked rather than relying on numpy indexing, which
        would happily wrap a negative index round to the far side of the map
        and report a cell on the other side of the circuit as free.
        """
        row, col = self.world_to_grid(xy)
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return self.is_free_value(self.values[row, col])

    def is_free_many(self, points) -> np.ndarray:
        """Vectorised `is_free` over an (N, 2) array. Off-grid points are False."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cols = np.floor((pts[:, 0] - self.origin[0]) / self.resolution).astype(np.int64)
        rows = np.floor((pts[:, 1] - self.origin[1]) / self.resolution).astype(np.int64)

        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        out = np.zeros(pts.shape[0], dtype=bool)
        if not np.any(inside):
            return out

        sampled = self.values[rows[inside], cols[inside]].astype(np.float64)
        out[inside] = (255.0 - sampled) / 255.0 < self.free_thresh
        return out

    def free_distance(self, origin_xy, direction, max_m, step=None) -> float:
        """March from `origin_xy` along unit `direction` until a cell is not free.

        Used to measure how much track there is either side of a centreline
        point, which is what bounds the racing line.
        """
        step = step or (self.resolution * 0.5)
        count = max(int(math.ceil(float(max_m) / step)), 1)
        offsets = np.arange(1, count + 1) * step
        samples = np.asarray(origin_xy, dtype=np.float64) + np.outer(
            offsets, np.asarray(direction, dtype=np.float64)
        )
        free = self.is_free_many(samples)
        blocked = np.nonzero(~free)[0]
        return float(offsets[blocked[0]] - step) if blocked.size else float(max_m)


class Centreline:
    """A closed path, with the arc length and curvature a speed profile needs."""

    def __init__(self, points):
        self.xy = _collapse_duplicates(np.asarray(points, dtype=np.float64).reshape(-1, 2))
        if len(self.xy) < 3:
            raise ValueError("a centreline needs at least three distinct points")

        deltas = np.roll(self.xy, -1, axis=0) - self.xy      # segment i -> i+1, wrapping
        segments = np.hypot(deltas[:, 0], deltas[:, 1])
        self.length = float(segments.sum())
        self.s = np.concatenate(([0.0], np.cumsum(segments)[:-1]))

        self.tangent = _unit(np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0))
        # Left-hand normal, matching the steering sign convention where
        # positive angles are to the left.
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)
        self.curvature = _menger_curvature(self.xy)

    @classmethod
    def load(cls, csv_path: str) -> "Centreline":
        return cls(np.loadtxt(csv_path, delimiter=",", comments="#"))

    def save(self, csv_path: str, header: str = "") -> None:
        np.savetxt(csv_path, self.xy, delimiter=",", header=header or "x_m, y_m")

    def nearest(self, xy) -> Tuple[int, float]:
        """Index of the closest sample, and the distance to it."""
        offsets = self.xy - np.asarray(xy, dtype=np.float64)
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def point_at_s(self, s) -> np.ndarray:
        """Linearly interpolated point at arc length `s`, wrapping round the lap."""
        s = float(s) % self.length
        knots = np.concatenate((self.s, [self.length]))
        closed = np.vstack((self.xy, self.xy[:1]))
        return np.array(
            [np.interp(s, knots, closed[:, 0]), np.interp(s, knots, closed[:, 1])]
        )

    def half_widths(self, grid: OccupancyGrid, max_m: float = 2.0) -> np.ndarray:
        """Free distance to the boundary either side, as (N, 2) = (left, right).

        This is what bounds the racing line: the line may move sideways by at
        most this much, minus whatever margin you insist on keeping.
        """
        out = np.zeros((len(self.xy), 2), dtype=np.float64)
        for i, (point, normal) in enumerate(zip(self.xy, self.normal)):
            out[i, 0] = grid.free_distance(point, normal, max_m)
            out[i, 1] = grid.free_distance(point, -normal, max_m)
        return out


# -- helpers ----------------------------------------------------------------

def _read_pgm(path: str) -> np.ndarray:
    """Binary PGM (P5). Written by hand because the alternative is Pillow, and
    the header parsing is the only fiddly part."""
    with open(path, "rb") as handle:
        data = handle.read()

    fields, cursor = [], 0
    while len(fields) < 4:
        while cursor < len(data) and data[cursor : cursor + 1].isspace():
            cursor += 1
        if data[cursor : cursor + 1] == b"#":
            while cursor < len(data) and data[cursor : cursor + 1] != b"\n":
                cursor += 1
            continue
        start = cursor
        while cursor < len(data) and not data[cursor : cursor + 1].isspace():
            cursor += 1
        fields.append(data[start:cursor])

    magic, width, height = fields[0], int(fields[1]), int(fields[2])
    if magic != b"P5":
        raise ValueError(f"{path}: expected a binary PGM (P5), got {magic!r}")

    cursor += 1  # exactly one whitespace byte after maxval
    pixels = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=cursor)
    return pixels.reshape(height, width)


def _collapse_duplicates(points: np.ndarray) -> np.ndarray:
    """Drop consecutive repeats, and the closing point if it repeats the first.

    Duplicates are a zero-length segment, and a zero-length segment is a
    division by zero in every tangent and curvature formula there is.
    """
    if len(points) < 2:
        return points
    steps = np.hypot(*(points[1:] - points[:-1]).T)
    keep = np.concatenate(([True], steps > _EPS))
    points = points[keep]
    if len(points) > 1 and np.hypot(*(points[-1] - points[0])) <= _EPS:
        points = points[:-1]
    return points


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.hypot(vectors[:, 0], vectors[:, 1])
    norms = np.where(norms < _EPS, 1.0, norms)
    return vectors / norms[:, None]


def _menger_curvature(points: np.ndarray) -> np.ndarray:
    """Signed curvature through each point and its two neighbours.

    Menger's formula, k = 4 * area / (|AB| |BC| |CA|), which is exact for
    points on a circle and needs no derivative estimate. Sign comes from the
    cross product, positive for a left-hand turn.
    """
    before = np.roll(points, 1, axis=0)
    after = np.roll(points, -1, axis=0)

    ab = points - before
    bc = after - points
    ca = before - after

    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    lengths = (
        np.hypot(ab[:, 0], ab[:, 1])
        * np.hypot(bc[:, 0], bc[:, 1])
        * np.hypot(ca[:, 0], ca[:, 1])
    )
    return np.where(lengths < _EPS, 0.0, 2.0 * cross / np.where(lengths < _EPS, 1.0, lengths))
