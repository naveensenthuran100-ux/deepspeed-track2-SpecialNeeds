"""Building our own occupancy grid from recorded driving.

The simulator ships the circuit as Unity geometry, not as a grid. The one in
maps/ was traced by the organisers from a reactive driver's wandering - it is
accurate but not beautiful, and rule 45 gives bonus marks for building a
better one from our own laps.

Tested as a round trip: generate scans from a KNOWN grid with the surrogate
sensor, rebuild a grid from those scans, and check the rebuild agrees with
the original. That is the only check that means anything before the car has
driven.
"""
import numpy as np
import pytest

from team_driver.core.track import OccupancyGrid
from team_driver.tools.grid_build import build_grid, save_grid
from team_driver.tools.surrogate import SurrogateSensor

FREE, BLOCKED, UNKNOWN = 254, 0, 205


def _corridor(length_m=16.0, width_m=2.0, resolution=0.05):
    cols = int(length_m / resolution)
    rows = int((width_m + 1.0) / resolution)
    values = np.full((rows, cols), BLOCKED, dtype=np.uint8)
    low = int(0.5 / resolution)
    values[low:low + int(width_m / resolution), :] = FREE
    return OccupancyGrid(values, resolution, (0.0, 0.0))


def _drive_and_scan(grid, xs, y=1.5, beams=361):
    sensor = SurrogateSensor(grid, max_range=10.0, beam_count=beams)
    poses, scans = [], []
    for x in xs:
        ranges, angles = sensor.scan(x, y, 0.0)
        poses.append((x, y, 0.0))
        scans.append(ranges)
    return np.array(poses), np.array(scans), angles


def test_rebuilds_a_corridor_as_free_space():
    truth = _corridor()
    poses, scans, angles = _drive_and_scan(truth, np.arange(1.0, 15.0, 0.25))
    rebuilt = build_grid(poses, scans, angles, resolution=0.05, max_range=10.0)

    for x in (3.0, 7.0, 12.0):
        assert rebuilt.is_free((x, 1.5)) is True


def test_rebuilds_the_walls_as_not_free():
    truth = _corridor()
    poses, scans, angles = _drive_and_scan(truth, np.arange(1.0, 15.0, 0.25))
    rebuilt = build_grid(poses, scans, angles, resolution=0.05, max_range=10.0)

    for x in (3.0, 7.0, 12.0):
        assert rebuilt.is_free((x, 2.6)) is False      # beyond the far wall
        assert rebuilt.is_free((x, 0.4)) is False      # beyond the near wall


def test_never_observed_space_is_unknown_not_free():
    """A planner told the outfield is free cheerfully cuts the corner across it."""
    truth = _corridor()
    # Drive only the first two metres. The LiDAR reaches 10 m, so anything
    # past about x = 13 was never seen by any beam.
    poses, scans, angles = _drive_and_scan(truth, np.arange(1.0, 3.0, 0.25))
    rebuilt = build_grid(poses, scans, angles, resolution=0.05, max_range=10.0)
    assert rebuilt.is_free((15.0, 1.5)) is False


def test_grid_covers_the_driven_area():
    truth = _corridor()
    poses, scans, angles = _drive_and_scan(truth, np.arange(1.0, 15.0, 0.25))
    rebuilt = build_grid(poses, scans, angles, resolution=0.05, max_range=10.0)
    assert rebuilt.width > 0 and rebuilt.height > 0
    for pose in poses:
        assert rebuilt.is_free((pose[0], pose[1])) is True


def test_saved_grid_round_trips_through_disk(tmp_path):
    truth = _corridor()
    poses, scans, angles = _drive_and_scan(truth, np.arange(1.0, 15.0, 0.25))
    rebuilt = build_grid(poses, scans, angles, resolution=0.05, max_range=10.0)

    stem = str(tmp_path / "rebuilt")
    save_grid(rebuilt, stem)
    reloaded = OccupancyGrid.load(stem + ".pgm", stem + ".yaml")

    assert reloaded.values.shape == rebuilt.values.shape
    assert abs(reloaded.resolution - rebuilt.resolution) < 1e-9
    assert np.allclose(reloaded.origin, rebuilt.origin)
    assert reloaded.is_free((7.0, 1.5)) is True


def test_an_empty_recording_does_not_crash():
    grid = build_grid(np.zeros((0, 3)), np.zeros((0, 10)), np.linspace(-1, 1, 10),
                      resolution=0.05, max_range=10.0)
    assert grid is None
