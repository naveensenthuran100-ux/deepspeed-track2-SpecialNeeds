"""Assemble the racing stack from committed artefacts.

The node's job is to call this and hand the result to the planner. Kept here,
free of ROS, so it is testable in milliseconds and so the surrogate and the
sweep build exactly the same stack the car drives.

The speed profile is computed at startup from the line and vehicle.json
rather than shipped as a third artefact. It takes a few milliseconds, and it
keeps the profile honest: change the identified grip and the profile changes
with it, instead of silently describing a car we no longer believe in.
"""

import os
from typing import Optional, Tuple

import numpy as np

from .profile import speed_profile
from .track import Centreline
from .vehicle import Vehicle


def load_racing_stack(
    vehicle: Vehicle,
    raceline_csv: str,
    grip_fraction: float = 0.4,
    v_max: Optional[float] = None,
    v_min: float = 0.3,
    friction_circle: bool = False,
    grip_fraction_fast: float = 0.0,
    fast_radius: float = 2.0,
) -> Tuple[Optional[Centreline], Optional[np.ndarray]]:
    """(path, speed profile), or (None, None) if the line cannot be loaded.

    Degrading rather than raising is deliberate. Gap mode needs no path, and
    a driver that starts and laps slowly scores; a node that dies on the grid
    is a DNF and zero for both components.
    """
    if not raceline_csv or not os.path.exists(raceline_csv):
        return None, None

    try:
        path = Centreline.load(raceline_csv)
        profile = speed_profile(
            path.curvature, path.s, vehicle,
            grip_fraction=grip_fraction,
            v_max=v_max,
            v_min=v_min,
            total_length=path.length,
            friction_circle=friction_circle,
            grip_fraction_fast=grip_fraction_fast,
            fast_radius=fast_radius,
        )
    except Exception:
        return None, None

    return path, profile
