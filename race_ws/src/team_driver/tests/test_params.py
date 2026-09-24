"""The node's parameter wiring, guarded without needing ROS.

driver.py maps ROS parameter names onto dataclass fields. Nothing checked
that the fields existed, and they can only be exercised by starting the node,
so a drift between the two is invisible until the car is on the grid on
judging day. `pursuit.curvature_cap` was exactly that: mapped, declared,
documented in driver_params.yaml, and not a field - the node would have
raised TypeError on startup and scored zero for both components.

Keeping the maps in core/ (ROS-free) is what makes this testable at all.
"""
import dataclasses

import pytest

from team_driver.core.gap import GapConfig
from team_driver.core.params import (GAP_PARAMS, MODE_PARAMS, PURSUIT_PARAMS,
                                     SPEED_PARAMS, build_configs)
from team_driver.core.planner import Planner, PlannerConfig
from team_driver.core.vehicle import Vehicle


def _fields(cls):
    return {f.name for f in dataclasses.fields(cls)}


def test_every_gap_parameter_is_a_real_field():
    mapped = {field for field, _ in GAP_PARAMS.values()}
    assert mapped <= _fields(GapConfig), mapped - _fields(GapConfig)


@pytest.mark.parametrize("group", [SPEED_PARAMS, PURSUIT_PARAMS, MODE_PARAMS])
def test_every_planner_parameter_is_a_real_field(group):
    mapped = {field for field, _ in group.values()}
    assert mapped <= _fields(PlannerConfig), mapped - _fields(PlannerConfig)


def test_parameter_names_do_not_collide():
    names = [n for g in (GAP_PARAMS, SPEED_PARAMS, PURSUIT_PARAMS, MODE_PARAMS) for n in g]
    assert len(names) == len(set(names))


def test_build_configs_from_defaults_constructs_a_working_planner():
    """The exact call the node makes, with the exact defaults it declares."""
    values = {name: default
              for group in (GAP_PARAMS, SPEED_PARAMS, PURSUIT_PARAMS, MODE_PARAMS)
              for name, (_, default) in group.items()}
    config = build_configs(values)
    planner = Planner(Vehicle.default(), config)
    assert planner.pursuit.curvature_cap == values["pursuit.curvature_cap"]
    assert planner.gap_follower.config.v_max == values["gap.v_max"]


def test_planner_passes_pursuit_settings_through_to_the_tracker():
    config = PlannerConfig(k_lookahead=0.11, min_lookahead=0.22,
                           max_lookahead=3.3, curvature_cap=0.44)
    planner = Planner(Vehicle.default(), config)
    assert planner.pursuit.k_lookahead == 0.11
    assert planner.pursuit.min_lookahead == 0.22
    assert planner.pursuit.max_lookahead == 3.3
    assert planner.pursuit.curvature_cap == 0.44


def test_parameter_defaults_match_the_dataclass_defaults():
    """The map and the dataclass must agree on every default.

    They did not. `speed.brake_gain` was raised to 1.0 in the map but left at
    0.30 in PlannerConfig, so the ROS node braked at full authority while
    every offline experiment - the whole configuration sweep included - ran
    at one third of it. The sweep's conclusions were measured on a car that
    could not stop, and the guard above did not catch it because the field
    existed; only its value was wrong.
    """
    gap_defaults = {f.name: f.default for f in dataclasses.fields(GapConfig)}
    planner_defaults = {f.name: f.default for f in dataclasses.fields(PlannerConfig)}

    mismatches = []
    for name, (field, default) in GAP_PARAMS.items():
        if gap_defaults[field] != default:
            mismatches.append(f"{name}: map={default} dataclass={gap_defaults[field]}")
    for group in (SPEED_PARAMS, PURSUIT_PARAMS, MODE_PARAMS):
        for name, (field, default) in group.items():
            if planner_defaults[field] != default:
                mismatches.append(
                    f"{name}: map={default} dataclass={planner_defaults[field]}")

    assert mismatches == [], "parameter defaults have drifted: " + "; ".join(mismatches)
