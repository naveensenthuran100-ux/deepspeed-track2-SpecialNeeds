"""Minimum-curvature racing line.

Wide in, apex, wide out: shorter and faster than the middle of the track.
Generating the line from the map in code we submit is explicitly worth bonus
marks; hand-typed waypoints from watching the car are explicitly not
(rule 36, rule 45).
"""
import pathlib

import numpy as np
import pytest

from team_driver.core.track import Centreline, OccupancyGrid
from team_driver.tools.line_optimise import optimise_offsets, optimise_raceline

MAPS = pathlib.Path(__file__).resolve().parents[4] / "maps"


@pytest.fixture(scope="module")
def track():
    grid = OccupancyGrid.load(
        str(MAPS / "icra26_compete.pgm"), str(MAPS / "icra26_compete.yaml")
    )
    return grid, Centreline.load(str(MAPS / "icra26_compete_centerline.csv"))


def _total_squared_curvature(line):
    return float(np.sum(line.curvature ** 2))


def test_offsets_respect_the_track_bounds():
    lower = np.full(50, -0.4)
    upper = np.full(50, 0.6)
    centre = Centreline(
        np.stack([np.cos(np.linspace(0, 2 * np.pi, 50, endpoint=False)) * 3.0,
                  np.sin(np.linspace(0, 2 * np.pi, 50, endpoint=False)) * 3.0], axis=1)
    )
    offsets = optimise_offsets(centre, lower, upper)
    assert np.all(offsets >= lower - 1e-6)
    assert np.all(offsets <= upper + 1e-6)


def test_a_circle_is_already_minimum_curvature():
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    centre = Centreline(np.stack([2.0 * np.cos(angles), 2.0 * np.sin(angles)], axis=1))
    offsets = optimise_offsets(centre, np.full(200, -0.3), np.full(200, 0.3))
    # It may drift outward to a marginally larger radius, but must not wander.
    assert np.std(offsets) < 0.05


def test_raceline_stays_on_the_track(track):
    grid, centre = track
    line = optimise_raceline(centre, grid, margin_m=0.30)
    assert np.all(grid.is_free_many(line.xy))


def test_raceline_is_straighter_than_the_centreline(track):
    grid, centre = track
    line = optimise_raceline(centre, grid, margin_m=0.30)
    assert _total_squared_curvature(line) < _total_squared_curvature(centre)


def test_raceline_keeps_its_distance_from_the_boundary(track):
    grid, centre = track
    margin = 0.30
    line = optimise_raceline(centre, grid, margin_m=margin)
    clearances = np.array([
        grid.free_distance(point, direction, 2.0)
        for point, normal in zip(line.xy, line.normal)
        for direction in (normal, -normal)
    ])
    assert clearances.min() >= margin - 0.08


def test_raceline_has_the_same_sample_count(track):
    grid, centre = track
    line = optimise_raceline(centre, grid, margin_m=0.30)
    assert len(line.xy) == len(centre.xy)


def test_raceline_is_deterministic(track):
    grid, centre = track
    first = optimise_raceline(centre, grid, margin_m=0.30)
    second = optimise_raceline(centre, grid, margin_m=0.30)
    assert np.allclose(first.xy, second.xy)


def _profile_lap_time(line, grip=0.4):
    """Lap time the speed profile would achieve on this line, in seconds.

    The only quality metric that means anything here. Total squared curvature
    is not monotone in the room available: a proper racing line straightens
    the straights and tightens the apexes, so it can carry more peak
    curvature than the centreline while being much faster.
    """
    from team_driver.core.profile import speed_profile
    from team_driver.core.vehicle import Vehicle

    speeds = speed_profile(line.curvature, line.s, Vehicle.default(), grip,
                           v_max=6.0, total_length=line.length)
    steps = np.diff(np.concatenate((line.s, [line.length])))
    return float(np.sum(steps / speeds))


def test_raceline_is_faster_than_the_centreline(track):
    """The whole point. Worth about 3.5 s a lap on this circuit."""
    grid, centre = track
    line = optimise_raceline(centre, grid, margin_m=0.30)
    assert _profile_lap_time(line) < _profile_lap_time(centre) - 2.0


def test_a_wider_margin_gives_a_slower_lap(track):
    grid, centre = track
    tight = optimise_raceline(centre, grid, margin_m=0.25)
    cautious = optimise_raceline(centre, grid, margin_m=0.55)
    assert _profile_lap_time(cautious) > _profile_lap_time(tight)


def test_a_margin_wider_than_the_track_falls_back_to_the_centreline(track):
    """Asking for more clearance than exists is a judgement error, not a crash.

    The bounded solver requires strictly lower < upper, so collapsing both
    onto the centreline raised ValueError and took the whole sweep with it.
    """
    grid, centre = track
    line = optimise_raceline(centre, grid, margin_m=1.5)
    assert np.allclose(line.xy, centre.xy, atol=0.02)
