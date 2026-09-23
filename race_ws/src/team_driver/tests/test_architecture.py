"""Boundaries the design depends on.

`core/` importing ROS would mean the whole suite needs a ROS environment to
run, which is exactly the six-minute feedback loop the layering exists to
avoid. Cheap to check, expensive to discover late.
"""
import pathlib
import re

CORE = pathlib.Path(__file__).resolve().parent.parent / "team_driver" / "core"
ROS_IMPORT = re.compile(r"^\s*(?:from|import)\s+(rclpy|ament_index|std_msgs|sensor_msgs|nav_msgs|geometry_msgs|visualization_msgs)\b", re.M)


def test_core_imports_no_ros():
    offenders = [p.name for p in CORE.glob("*.py") if ROS_IMPORT.search(p.read_text())]
    assert offenders == []


def test_core_modules_are_importable_without_ros():
    import team_driver.core.gap          # noqa: F401
    import team_driver.core.lidar        # noqa: F401
    import team_driver.core.planner      # noqa: F401
    import team_driver.core.speed_controller  # noqa: F401
    import team_driver.core.vehicle      # noqa: F401
