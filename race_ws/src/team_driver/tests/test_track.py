"""Occupancy grid and centreline geometry.

Knowing where we are changes the problem: we can brake for a corner before
the 10 m LiDAR can see it, which is most of the lap time on a 54 m circuit.
Everything in the line-following path stands on this module being right, so
it is tested against the real shipped map rather than only against fixtures.
"""
import pathlib

import numpy as np
import pytest

from team_driver.core.track import Centreline, OccupancyGrid

MAPS = pathlib.Path(__file__).resolve().parents[4] / "maps"
PGM = MAPS / "icra26_compete.pgm"
YAML = MAPS / "icra26_compete.yaml"
CENTRELINE = MAPS / "icra26_compete_centerline.csv"


@pytest.fixture(scope="module")
def grid():
    return OccupancyGrid.load(str(PGM), str(YAML))


@pytest.fixture(scope="module")
def centreline():
    return Centreline.load(str(CENTRELINE))


# -- the grid ---------------------------------------------------------------

def test_grid_matches_the_published_dimensions(grid):
    assert grid.width == 220
    assert grid.height == 476
    assert abs(grid.resolution - 0.05) < 1e-9


def test_world_grid_roundtrip(grid):
    for xy in [(0.0, 0.0), (0.8, 3.16), (-3.0, -5.0)]:
        assert np.allclose(grid.grid_to_world(grid.world_to_grid(xy)), xy, atol=0.05)


def test_spawn_point_is_on_the_track(grid):
    # tracks.yaml: the car spawns at (0.80, 3.16) facing -y.
    assert grid.is_free((0.80, 3.16)) is True


def test_offgrid_query_is_safe(grid):
    assert grid.is_free((1e6, 1e6)) is False
    assert grid.is_free((-1e6, -1e6)) is False


def test_unknown_cells_are_not_free(grid):
    # free_thresh is 0.196 precisely so that value 205 (outside the track)
    # lands in the unknown band rather than reading as free space.
    assert grid.is_free_value(205) is False
    assert grid.is_free_value(254) is True
    assert grid.is_free_value(0) is False


def test_is_free_many_matches_is_free(grid):
    points = np.array([[0.80, 3.16], [1e6, 1e6], [0.0, 0.0]])
    expected = [grid.is_free(tuple(p)) for p in points]
    assert list(grid.is_free_many(points)) == expected


# -- the centreline ---------------------------------------------------------

def test_curvature_of_a_circle():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    line = Centreline(np.stack([2.0 * np.cos(t), 2.0 * np.sin(t)], axis=1))
    assert np.allclose(np.abs(line.curvature), 0.5, atol=0.02)


def test_circle_arc_length():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    line = Centreline(np.stack([2.0 * np.cos(t), 2.0 * np.sin(t)], axis=1))
    assert abs(line.length - 2 * np.pi * 2.0) < 0.05


def test_duplicate_points_do_not_divide_by_zero():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    line = Centreline(points)
    assert np.all(np.isfinite(line.curvature))
    assert np.all(np.isfinite(line.s))


def test_closed_loop_last_point_equal_to_first_is_collapsed():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    line = Centreline(points)
    assert len(line.xy) == 3


def test_shipped_centreline_lap_length(centreline):
    # maps/icra26_compete_centerline.csv header: "Lap length 54.43 m".
    assert abs(centreline.length - 54.43) < 0.5


def test_shipped_centreline_lies_on_free_cells(centreline, grid):
    # The map README's own check: every sample of the traced line is on a
    # free cell. A centreline that disagrees with the grid is worse than no
    # map at all.
    assert np.all(grid.is_free_many(centreline.xy))


def test_nearest_finds_the_closest_sample(centreline):
    index, distance = centreline.nearest(tuple(centreline.xy[42]))
    assert index == 42
    assert distance < 1e-9


def test_nearest_off_the_line_reports_a_real_distance(centreline):
    point = centreline.xy[10] + centreline.normal[10] * 0.3
    index, distance = centreline.nearest(tuple(point))
    assert abs(distance - 0.3) < 0.05


def test_point_at_s_wraps_around_the_lap(centreline):
    start = centreline.point_at_s(0.0)
    wrapped = centreline.point_at_s(centreline.length)
    assert np.allclose(start, wrapped, atol=0.05)


def test_tangent_and_normal_are_unit_and_perpendicular(centreline):
    assert np.allclose(np.linalg.norm(centreline.tangent, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(centreline.normal, axis=1), 1.0, atol=1e-6)
    dots = np.sum(centreline.tangent * centreline.normal, axis=1)
    assert np.allclose(dots, 0.0, atol=1e-9)


def test_half_widths_are_positive_and_bounded(centreline, grid):
    widths = centreline.half_widths(grid, max_m=2.0)
    assert widths.shape == (len(centreline.xy), 2)
    assert np.all(widths >= 0.0)
    assert np.all(widths <= 2.0)


def test_half_widths_find_a_corridor_about_two_metres_wide(centreline, grid):
    # docs/02-simulator.md 2.1: about 2 m typical, under 1 m at the tightest.
    total = centreline.half_widths(grid, max_m=2.0).sum(axis=1)
    assert 0.6 < np.median(total) < 3.0
    assert total.min() < 1.6
