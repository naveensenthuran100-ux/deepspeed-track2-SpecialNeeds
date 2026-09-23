"""Scoring, as the organisers compute it.

Optimising against anything else optimises against the wrong thing. A
collision is 10 s on the lap it happened on, and both halves of the score are
ratios against the best team, so the objective is expected net time - which
already prices contact correctly and needs no separate safety weight.
"""
import numpy as np
import pytest

from team_driver.tools.scoring import (leaderboard_points, net_lap_time,
                                       summarise_runs)


def test_a_collision_costs_ten_seconds():
    assert net_lap_time(20.0, 1) == 30.0
    assert net_lap_time(20.0, 0) == 20.0


def test_the_fastest_team_scores_fifty_in_each_half():
    points = leaderboard_points(20.5, 20.5, 228.9, 228.9)
    assert abs(points - 100.0) < 1e-9


def test_points_scale_as_the_ratio_of_times():
    # The guide's own worked example: Alpha, 21.80 best lap, 228.90 total.
    points = leaderboard_points(21.80, 20.50, 228.90, 228.90)
    assert abs(points - 97.02) < 0.01


def test_consistency_beats_a_single_quick_lap():
    # Bravo has the fastest lap of anyone and still finishes behind Alpha.
    alpha = leaderboard_points(21.80, 20.50, 228.90, 228.90)
    bravo = leaderboard_points(20.50, 20.50, 245.00, 228.90)
    assert alpha > bravo


def test_a_disqualified_run_scores_nothing():
    assert leaderboard_points(None, 20.5, None, 228.9) == 0.0


def test_summarise_reports_completion_rate():
    runs = [
        {"status": "COMPLETE", "scored": True, "best_lap_time": 20.0,
         "total_time": 220.0, "collisions": 0},
        {"status": "DISQUALIFIED", "scored": False, "best_lap_time": None,
         "total_time": None, "collisions": 11},
        {"status": "COMPLETE", "scored": True, "best_lap_time": 21.0,
         "total_time": 230.0, "collisions": 1},
    ]
    summary = summarise_runs(runs)
    assert summary["runs"] == 3
    assert abs(summary["completion_rate"] - 2 / 3) < 1e-9
    assert abs(summary["best_lap_mean"] - 20.5) < 1e-9
    assert summary["best_lap_best"] == 20.0


def test_summarise_reports_expected_best_of_three():
    """The judges take the best of three, so a config that fails sometimes is
    worth more than its average run suggests."""
    runs = [{"status": "COMPLETE", "scored": True, "best_lap_time": 20.0,
             "total_time": 220.0, "collisions": 0}] * 3
    assert summarise_runs(runs)["expected_scoring_run_in_three"] > 0.99


def test_summarise_of_nothing_does_not_divide_by_zero():
    summary = summarise_runs([])
    assert summary["runs"] == 0
    assert summary["completion_rate"] == 0.0
    assert summary["best_lap_mean"] is None
