"""Construct the node with ROS stubbed out.

driver.py cannot be imported in this venv, so nothing exercised it. Three
edits silently failed to apply and left `_PURSUIT_PARAMS`, `_MODE_PARAMS` and
`_RACELINE_CANDIDATES` referenced but never defined. The module still
imported - they are used inside `Driver.__init__`, not at module level - so
it would have raised NameError the moment the node started. On judging day
that is a DNF and zero for both components.

Importing is not enough; these tests build a Driver. Stubbing ROS is cheap
and catches the whole class: undefined names, a parameter list drifted from
core/params.py, and a racing line that fails to load.
"""
import pathlib
import sys
import types

import pytest

from team_driver.core.params import ALL_GROUPS

RACELINE = str(pathlib.Path(__file__).resolve().parents[4] / "maps"
               / "icra26_compete_raceline.csv")


class _Parameter:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    """Just enough rclpy.node.Node for Driver.__init__ to run."""

    overrides = {}

    def __init__(self, *args, **kwargs):
        self._parameters = {}
        self.published = []
        self.subscriptions_made = []

    def declare_parameter(self, name, default):
        self._parameters[name] = self.overrides.get(name, default)

    def get_parameter(self, name):
        return _Parameter(self._parameters[name])

    def create_publisher(self, *args, **kwargs):
        self.published.append(args)
        return types.SimpleNamespace(publish=lambda *a, **k: None)

    def create_subscription(self, *args, **kwargs):
        self.subscriptions_made.append(args[1])
        return None

    def get_logger(self):
        return types.SimpleNamespace(
            info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)

    def get_clock(self):
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(to_msg=lambda: None))


def _install_stubs():
    def message(*_args, **_kwargs):
        return types.SimpleNamespace()

    stubs = {
        "rclpy": {"init": lambda *a, **k: None, "ok": lambda: False,
                  "shutdown": lambda: None, "spin": lambda *a: None},
        "rclpy.node": {"Node": _FakeNode},
        "rclpy.qos": {
            "QoSProfile": message,
            # Enums, not callables: driver.py reads members off them.
            "QoSDurabilityPolicy": types.SimpleNamespace(
                VOLATILE="volatile", TRANSIENT_LOCAL="transient_local"),
            "QoSHistoryPolicy": types.SimpleNamespace(
                KEEP_LAST="keep_last", KEEP_ALL="keep_all"),
            "QoSReliabilityPolicy": types.SimpleNamespace(
                RELIABLE="reliable", BEST_EFFORT="best_effort"),
        },
        "sensor_msgs": {}, "sensor_msgs.msg": {"LaserScan": message},
        "nav_msgs": {}, "nav_msgs.msg": {"Odometry": message},
        "geometry_msgs": {}, "geometry_msgs.msg": {"Point": message},
        "std_msgs": {}, "std_msgs.msg": {"Float32": message},
        "visualization_msgs": {}, "visualization_msgs.msg":
            {"Marker": message, "MarkerArray": message},
    }
    installed = []
    for name, attributes in stubs.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            sys.modules[name] = module
            installed.append(name)
    return installed


@pytest.fixture
def driver_module():
    installed = _install_stubs()
    sys.modules.pop("team_driver.driver", None)
    import team_driver.driver as module

    yield module

    for name in installed + ["team_driver.driver"]:
        sys.modules.pop(name, None)
    _FakeNode.overrides = {}


def test_node_constructs(driver_module):
    """The test that would have caught three undefined names."""
    node = driver_module.Driver()
    assert node.planner is not None


def test_node_declares_every_shared_parameter(driver_module):
    node = driver_module.Driver()
    for group in ALL_GROUPS:
        for name in group:
            assert name in node._parameters, f"{name} never declared"


def test_node_subscribes_to_lidar_odom_and_ips(driver_module):
    node = driver_module.Driver()
    assert len(node.subscriptions_made) == 3


def test_node_loads_the_racing_line_when_pointed_at_it(driver_module):
    _FakeNode.overrides = {"line.raceline_csv": RACELINE}
    node = driver_module.Driver()
    assert node.planner.path is not None
    assert node.planner.profile is not None
    assert len(node.planner.profile) == len(node.planner.path.xy)


def test_node_without_a_racing_line_still_drives(driver_module):
    _FakeNode.overrides = {"line.raceline_csv": "/nonexistent.csv"}
    node = driver_module.Driver()
    assert node.planner.path is None       # gap mode only, but alive


def test_parameters_reach_the_planner(driver_module):
    _FakeNode.overrides = {"gap.v_max": 3.25, "pursuit.curvature_cap": 0.55}
    node = driver_module.Driver()
    assert node.planner.gap_follower.config.v_max == 3.25
    assert node.planner.pursuit.curvature_cap == 0.55


def test_devkit_qos_is_reliable_keep_last_one(driver_module):
    """Not cosmetic: a BEST_EFFORT subscriber silently receives nothing from
    the bridge's RELIABLE publisher, while ros2 topic list looks fine."""
    source = (pathlib.Path(driver_module.__file__)).read_text()
    assert "QoSReliabilityPolicy.RELIABLE" in source
    assert "QoSHistoryPolicy.KEEP_LAST" in source
    assert "depth=1" in source
