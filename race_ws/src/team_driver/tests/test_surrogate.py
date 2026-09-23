"""The surrogate simulator.

The whole optimisation strategy rests on this being both fast and honest. A
real 12-lap run costs about six minutes of wall clock and cannot be
parallelised far on one laptop; the surrogate has to make configuration
search cheap enough that we search analytically instead of guessing.

It is a model, so it can lie. These tests pin what it must get right; the
correlation against real simulator runs is measured separately by
lab/report.py and is the number that decides how far it may be trusted.
"""
import pathlib
import time

import numpy as np
import pytest

from team_driver.core.planner import Planner, PlannerConfig
from team_driver.core.profile import speed_profile
from team_driver.core.track import Centreline, OccupancyGrid
from team_driver.core.vehicle import Vehicle
from team_driver.tools.surrogate import SurrogateSensor, simulate

MAPS = pathlib.Path(__file__).resolve().parents[4] / "maps"

FREE, BLOCKED = 254, 0


def _corridor(length_m=20.0, width_m=2.0, resolution=0.05):
    """A straight east-west corridor with its lower-left corner at the origin."""
    cols = int(length_m / resolution)
    rows = int((width_m + 1.0) / resolution)
    values = np.full((rows, cols), BLOCKED, dtype=np.uint8)
    low = int(0.5 / resolution)
    high = low + int(width_m / resolution)
    values[low:high, :] = FREE
    return OccupancyGrid(values, resolution, (0.0, 0.0))


@pytest.fixture(scope="module")
def real_track():
    grid = OccupancyGrid.load(
        str(MAPS / "icra26_compete.pgm"), str(MAPS / "icra26_compete.yaml")
    )
    line = Centreline.load(str(MAPS / "icra26_compete_centerline.csv"))
    return grid, line


# -- the synthetic LiDAR ----------------------------------------------------

def test_sensor_measures_a_corridor_wall():
    grid = _corridor()
    sensor = SurrogateSensor(grid, max_range=10.0, beam_count=181)
    ranges, angles = sensor.scan(5.0, 1.5, 0.0)
    # The corridor spans y in [0.5, 2.5], so the middle of it has 1.0 m of
    # clearance to either side.
    left = int(np.argmin(np.abs(angles - np.pi / 2)))
    right = int(np.argmin(np.abs(angles + np.pi / 2)))
    assert abs(ranges[left] - 1.0) < 0.15
    assert abs(ranges[right] - 1.0) < 0.15


def test_sensor_reports_max_range_down_an_open_corridor():
    sensor = SurrogateSensor(_corridor(length_m=40.0), max_range=10.0, beam_count=181)
    ranges, angles = sensor.scan(2.0, 1.5, 0.0)
    ahead = int(np.argmin(np.abs(angles)))
    assert ranges[ahead] > 9.0


def test_sensor_ranges_are_bounded_and_finite():
    sensor = SurrogateSensor(_corridor(), max_range=10.0, beam_count=181)
    ranges, _ = sensor.scan(5.0, 1.5, 0.7)
    assert np.all(np.isfinite(ranges))
    assert np.all(ranges >= 0.0) and np.all(ranges <= 10.0)


def test_sensor_off_grid_reports_contact_not_open_space():
    sensor = SurrogateSensor(_corridor(), max_range=10.0, beam_count=181)
    ranges, _ = sensor.scan(-5.0, -5.0, 0.0)
    assert np.all(ranges < 1.0)


# -- the simulator ----------------------------------------------------------

def test_driving_straight_down_a_corridor_is_collision_free():
    grid = _corridor(length_m=30.0)
    result = simulate(
        Planner(Vehicle.default(), PlannerConfig()),
        Vehicle.default(), grid,
        start=(1.0, 1.5, 0.0), distance_m=15.0, max_time_s=30.0,
    )
    assert result.collisions == 0


def test_reports_contact_when_aimed_at_a_wall():
    grid = _corridor(length_m=30.0)
    result = simulate(
        Planner(Vehicle.default(), PlannerConfig()),
        Vehicle.default(), grid,
        start=(1.0, 1.5, 1.2),          # pointed into the corridor wall
        distance_m=15.0, max_time_s=10.0,
        steer_override=0.6,
    )
    assert result.collisions > 0


def test_is_deterministic():
    grid = _corridor(length_m=30.0)
    runs = [
        simulate(
            Planner(Vehicle.default(), PlannerConfig()),
            Vehicle.default(), grid,
            start=(1.0, 1.5, 0.0), distance_m=10.0, max_time_s=30.0,
        )
        for _ in range(2)
    ]
    assert runs[0].collisions == runs[1].collisions
    assert abs(runs[0].elapsed_s - runs[1].elapsed_s) < 1e-9


def test_completes_a_clean_lap_of_the_real_circuit(real_track):
    """A full lap of the real circuit, on the line, without touching anything.

    Grip fraction 0.4 rather than something braver on purpose: above about
    0.5 the car can corner that fast but cannot *track* the line there, because
    the 3.2 rad/s steering rate limit costs roughly 0.8 m of travel to reach
    lock at 5 m/s on a 1.1 m radius corner. Raising that ceiling - curvature
    feed-forward on the steering is the obvious first move - is the single
    biggest lap-time lever left, and belongs to tuning, not here.
    """
    grid, line = real_track
    vehicle = Vehicle.default()
    planner = Planner(vehicle, PlannerConfig())
    planner.load_path(
        line,
        speed_profile(line.curvature, line.s, vehicle, 0.4, v_max=6.0,
                      total_length=line.length),
    )
    start = (line.xy[0][0], line.xy[0][1], float(np.arctan2(*line.tangent[0][::-1])))
    result = simulate(planner, vehicle, grid, start=start,
                      distance_m=line.length, max_time_s=120.0)
    assert result.status == "COMPLETE"
    assert result.collisions == 0
    assert result.elapsed_s < 30.0


def test_a_lap_is_fast_enough_to_sweep_with(real_track):
    """The throughput claim the whole design rests on."""
    grid, line = real_track
    vehicle = Vehicle.default()
    planner = Planner(vehicle, PlannerConfig())
    planner.load_path(
        line,
        speed_profile(line.curvature, line.s, vehicle, 0.7, v_max=6.0,
                      total_length=line.length),
    )
    start = (line.xy[0][0], line.xy[0][1], float(np.arctan2(*line.tangent[0][::-1])))

    started = time.perf_counter()
    simulate(planner, vehicle, grid, start=start,
             distance_m=line.length, max_time_s=120.0)
    wall = time.perf_counter() - started
    # A real 12-lap run costs ~6 minutes. Anything under a second a lap makes
    # a several-hundred-config sweep an overnight job instead of a fortnight.
    assert wall < 1.5, f"one surrogate lap took {wall:.2f} s"


def test_continuous_contact_keeps_counting():
    """A car wedged against a wall must not register a single collision.

    Under-counting contact makes every sweep prefer configurations that
    scrape, which is the opposite of what a +10 s penalty should teach.
    """
    grid = _corridor(length_m=30.0)
    result = simulate(
        Planner(Vehicle.default(), PlannerConfig()),
        Vehicle.default(), grid,
        start=(1.0, 1.5, 1.4), distance_m=40.0, max_time_s=12.0,
        steer_override=0.9,
    )
    assert result.collisions > 3


def test_driving_off_the_map_is_not_a_completed_lap(real_track):
    """Progress must be measured along the path, not as distance travelled.

    Otherwise a car that leaves the circuit and drives in a straight line
    accumulates a lap length of travel and is recorded COMPLETE.
    """
    grid, line = real_track
    vehicle = Vehicle.default()
    planner = Planner(vehicle, PlannerConfig())
    planner.load_path(line, np.full(len(line.xy), 6.0))
    # Pointed straight across the circuit and out of it.
    start = (line.xy[0][0], line.xy[0][1],
             float(np.arctan2(line.tangent[0][1], line.tangent[0][0])) + np.pi / 2)
    result = simulate(planner, vehicle, grid, start=start,
                      distance_m=line.length, max_time_s=60.0,
                      steer_override=0.0)
    assert result.status != "COMPLETE"


def test_line_mode_holds_almost_everywhere_on_a_clear_circuit(real_track):
    """The regression that cost 26 seconds a lap.

    Standing on the racing line, pointed along it, at racing speed, with
    nothing on the track: the planner should be following the line. It was
    not. `_path_blocked` tested line-of-sight to a path point, and on corners
    of ~1 m radius the straight chord to a point one braking distance ahead
    leaves the corridor by geometry - so the car dropped to gap mode at every
    corner entry, which is exactly where the speed profile earns its keep.
    """
    grid, line = real_track
    vehicle = Vehicle.default()
    sensor = SurrogateSensor(grid, max_range=10.0, beam_count=181)

    in_line = 0
    tested = range(0, len(line.xy), 4)
    for index in tested:
        planner = Planner(vehicle, PlannerConfig(block_enter_frames=1))
        planner.load_path(line, np.full(len(line.xy), 5.0))
        x, y = line.xy[index]
        yaw = float(np.arctan2(line.tangent[index][1], line.tangent[index][0]))
        ranges, angles = sensor.scan(x, y, yaw)
        result = planner.step(ranges, angles, (x, y), yaw, 5.3, 0.025)
        in_line += result.mode == "line"

    fraction = in_line / len(list(tested))
    assert fraction > 0.9, f"only {fraction:.0%} of the clear circuit stayed in line mode"


def test_coasting_decelerates_more_gently_than_braking():
    """Lifting off and standing on the brakes are not the same thing.

    The first model made throttle 0 decelerate at the full braking limit,
    which made lifting off *stronger* than braking and hid the controller's
    missing brake authority.
    """
    from team_driver.tools.surrogate import longitudinal_accel

    vehicle = Vehicle.default()
    coast = longitudinal_accel(vehicle, 0.0, 5.0)
    brake = longitudinal_accel(vehicle, -1.0, 5.0)
    assert brake < coast < 0.0
    assert abs(coast) <= vehicle.a_coast + 1e-9
    assert abs(brake - -vehicle.a_brake_max) < 1e-9


def test_accelerating_is_bounded_by_the_traction_limit():
    from team_driver.tools.surrogate import longitudinal_accel

    vehicle = Vehicle.default()
    assert longitudinal_accel(vehicle, 1.0, 0.0) <= vehicle.a_accel_max + 1e-9
    assert longitudinal_accel(vehicle, 1.0, 0.0) > 0.0
