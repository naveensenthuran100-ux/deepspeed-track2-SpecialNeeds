"""ROS parameter names, mapped onto config fields.

Lives in core/ - free of ROS - so the mapping can be tested. It used to live
in driver.py, where nothing could reach it without a ROS environment, and a
name that did not correspond to a real field raised TypeError only when the
node started. On judging day that is a DNF and zero for both components.

Keep the nesting in driver_params.yaml matching the dotted names here: ROS 2
maps `gap: {v_max: 6.0}` onto the parameter `gap.v_max`.
"""

from .gap import GapConfig
from .planner import PlannerConfig

#: name -> (GapConfig field, default)
GAP_PARAMS = {
    "gap.max_range": ("max_range", 10.0),
    "gap.car_half_width": ("car_half_width", 0.20),
    "gap.disparity_threshold": ("disparity_threshold", 0.3),
    "gap.bubble_radius": ("bubble_radius", 0.35),
    "gap.gap_threshold_frac": ("gap_threshold_frac", 0.55),
    "gap.forward_cone_rad": ("forward_cone_rad", 0.26),
    "gap.stop_margin": ("stop_margin", 0.35),
    "gap.v_min": ("v_min", 0.5),
    "gap.v_max": ("v_max", 6.0),
    "gap.steer_speed_gain": ("steer_speed_gain", 0.7),
}

#: name -> (PlannerConfig field, default)
SPEED_PARAMS = {
    "speed.kp": ("kp", 0.05),
    "speed.ki": ("ki", 0.15),
    "speed.integral_limit": ("integral_limit", 0.30),
    "speed.lift_band": ("lift_band", 0.15),
    "speed.brake_gain": ("brake_gain", 1.0),
    "speed.full_throttle_band": ("full_throttle_band", 0.0),
    "speed.preview_s": ("speed_preview_s", 0.0),
}

PURSUIT_PARAMS = {
    "pursuit.k_lookahead": ("k_lookahead", 0.35),
    "pursuit.min_lookahead": ("min_lookahead", 0.45),
    "pursuit.max_lookahead": ("max_lookahead", 2.0),
    "pursuit.curvature_cap": ("curvature_cap", 0.8),
    "pursuit.tracker": ("tracker", "feedforward"),
    "pursuit.tau_preview": ("tau_preview", 0.16),
    "pursuit.min_preview": ("min_preview", 0.25),
    "pursuit.max_preview": ("max_preview", 2.5),
    "pursuit.k_heading": ("k_heading", 0.7),
    "pursuit.k_cross": ("k_cross", 2.2),
}

MODE_PARAMS = {
    "mode.max_path_offset": ("max_path_offset", 1.2),
    "mode.block_radius": ("block_radius", 0.30),
    "mode.block_min_horizon": ("block_min_horizon", 1.5),
    "mode.block_samples": ("block_samples", 8),
    "mode.block_enter_frames": ("block_enter_frames", 3),
    "mode.block_exit_frames": ("block_exit_frames", 10),
}

PLANNER_GROUPS = (SPEED_PARAMS, PURSUIT_PARAMS, MODE_PARAMS)
ALL_GROUPS = (GAP_PARAMS,) + PLANNER_GROUPS


def build_configs(values) -> PlannerConfig:
    """Turn a {parameter name: value} mapping into a PlannerConfig."""
    gap = GapConfig(**{field: values[name] for name, (field, _) in GAP_PARAMS.items()})
    planner = {
        field: values[name]
        for group in PLANNER_GROUPS
        for name, (field, _) in group.items()
    }
    return PlannerConfig(gap=gap, **planner)


def defaults() -> dict:
    """Every parameter at its declared default."""
    return {name: default
            for group in ALL_GROUPS
            for name, (_, default) in group.items()}
