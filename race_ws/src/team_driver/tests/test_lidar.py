"""LiDAR preprocessing.

The circuit is bounded by a row of separate 33 cm air ducts, so beams slip
between them and report the far side of the track or nothing at all. Every
test here exists because of that: without disparity extension a gap-follower
aims straight at a duct, and without cleaning, NaN propagates silently
through every arithmetic operation downstream.
"""
import numpy as np
import pytest

from team_driver.core.lidar import beam_angles, clean, disparity_extend, safety_bubble


def test_clean_replaces_nan_and_inf():
    out = clean(np.array([np.nan, np.inf, -np.inf, 5.0]), 10.0)
    assert np.all(np.isfinite(out))
    assert out[1] == 10.0
    assert out[0] == 0.0
    assert out[2] == 0.0


def test_clean_clips_above_max_range():
    assert clean(np.array([25.0]), 10.0)[0] == 10.0


def test_clean_does_not_mutate_its_input():
    original = np.array([np.nan, 5.0])
    clean(original, 10.0)
    assert np.isnan(original[0])


def test_beam_angles_match_the_devkit_scan():
    angles = beam_angles(-2.356, 0.25 * np.pi / 180.0, 1080)
    assert len(angles) == 1080
    assert abs(angles[0] - (-2.356)) < 1e-9
    assert angles[539] < 0.0 < angles[540]      # straight ahead near the middle


def test_disparity_extension_closes_a_duct_gap():
    ranges = np.full(101, 1.0)
    ranges[50] = 9.0                            # a beam slipping between ducts
    angles = np.linspace(-1.0, 1.0, 101)
    out = disparity_extend(ranges, angles, car_half_width=0.2, threshold=0.3)
    assert out[50] < 9.0


def test_disparity_extension_leaves_a_smooth_scan_alone():
    ranges = np.linspace(1.0, 1.5, 101)
    angles = np.linspace(-1.0, 1.0, 101)
    out = disparity_extend(ranges, angles, car_half_width=0.2, threshold=0.3)
    assert np.allclose(out, ranges)


def test_disparity_extension_never_increases_a_range():
    rng = np.random.RandomState(1)
    ranges = rng.uniform(0.2, 10.0, 200)
    angles = np.linspace(-2.356, 2.356, 200)
    out = disparity_extend(ranges, angles, car_half_width=0.2, threshold=0.3)
    assert np.all(out <= ranges + 1e-9)


def test_disparity_extension_keeps_a_genuine_opening_open():
    # A corridor wall that genuinely stops: 40 beams of open track beyond it
    # must not be closed off entirely, or the car has nowhere to aim.
    ranges = np.full(201, 1.0)
    ranges[120:] = 8.0
    angles = np.linspace(-2.0, 2.0, 201)
    out = disparity_extend(ranges, angles, car_half_width=0.135, threshold=0.3)
    assert np.max(out[120:]) > 5.0


def test_safety_bubble_blanks_around_closest_return():
    ranges = np.full(101, 5.0)
    ranges[20] = 0.3
    angles = np.linspace(-1.0, 1.0, 101)
    out = safety_bubble(ranges, angles, radius_m=0.5)
    assert out[20] == 0.0
    assert out[21] == 0.0


def test_safety_bubble_leaves_the_far_side_usable():
    ranges = np.full(201, 5.0)
    ranges[10] = 0.3
    angles = np.linspace(-2.0, 2.0, 201)
    out = safety_bubble(ranges, angles, radius_m=0.5)
    assert np.max(out) > 0.0        # a bubble that blanks everything is useless


def test_safety_bubble_on_an_empty_scan_is_a_noop():
    ranges = np.zeros(101)
    angles = np.linspace(-1.0, 1.0, 101)
    out = safety_bubble(ranges, angles, radius_m=0.5)
    assert np.all(np.isfinite(out))
