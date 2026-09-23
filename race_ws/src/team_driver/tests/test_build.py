"""Assembling the racing stack from committed artefacts.

Lives in core/ and imports no ROS so it can be tested here rather than only
inside a six-minute simulator run. The node is a two-line caller.
"""
import pathlib

import numpy as np
import pytest

from team_driver.core.build import load_racing_stack
from team_driver.core.vehicle import Vehicle

MAPS = pathlib.Path(__file__).resolve().parents[4] / "maps"
CENTRELINE = str(MAPS / "icra26_compete_centerline.csv")


def test_loads_a_path_and_a_matching_profile():
    path, profile = load_racing_stack(Vehicle.default(), CENTRELINE, grip_fraction=0.4)
    assert path is not None
    assert len(profile) == len(path.xy)


def test_profile_respects_the_speed_ceiling():
    _, profile = load_racing_stack(
        Vehicle.default(), CENTRELINE, grip_fraction=0.4, v_max=3.0
    )
    assert np.all(profile <= 3.0 + 1e-9)


def test_a_lower_grip_fraction_gives_a_slower_profile():
    _, slow = load_racing_stack(Vehicle.default(), CENTRELINE, grip_fraction=0.3)
    _, fast = load_racing_stack(Vehicle.default(), CENTRELINE, grip_fraction=0.6)
    assert slow.mean() < fast.mean()


def test_a_missing_file_degrades_instead_of_raising():
    """The node must still drive. Gap mode needs no path at all, and a run
    that starts and laps slowly beats a node that dies on the grid."""
    path, profile = load_racing_stack(Vehicle.default(), "/nonexistent/line.csv")
    assert path is None and profile is None


def test_a_corrupt_file_degrades_instead_of_raising(tmp_path):
    bad = tmp_path / "line.csv"
    bad.write_text("this is not a centreline\n")
    path, profile = load_racing_stack(Vehicle.default(), str(bad))
    assert path is None and profile is None
