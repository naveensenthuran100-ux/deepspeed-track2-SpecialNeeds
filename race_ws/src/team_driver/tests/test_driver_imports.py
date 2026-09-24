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


# ===========================================================================
# The development nodes. Same blind spot: nothing else imports them.
# ===========================================================================

@pytest.fixture
def tools_modules(driver_module):
    import team_driver.tools.map_record as map_record
    import team_driver.tools.sysid_drive as sysid_drive

    yield sysid_drive, map_record

    for name in ("team_driver.tools.sysid_drive", "team_driver.tools.map_record"):
        sys.modules.pop(name, None)


def test_sysid_node_constructs(tools_modules):
    sysid_drive, _ = tools_modules
    node = sysid_drive.SysIdDrive()
    assert node.vehicle is not None
    assert node.follower is not None


def test_map_record_node_constructs(tools_modules):
    _, map_record = tools_modules
    node = map_record.MapRecord()
    assert node.poses == []


def test_map_record_saves_nothing_when_nothing_was_recorded(tools_modules):
    _, map_record = tools_modules
    node = map_record.MapRecord()
    node.save()          # must warn, not raise, and write no file


def test_sysid_schedule_visits_every_phase_then_stops(tools_modules):
    """The schedule has to end, or the car never stops and the log never writes."""
    sysid_drive, _ = tools_modules
    node = sysid_drive.SysIdDrive()
    node.speed = 2.0
    node.yaw_rate = 0.5

    ended = False
    for step in range(0, 20000):
        node.elapsed = step * 0.025
        if node._throttle_for(node.elapsed) is None:
            ended = True
            break
    assert ended, "the identification schedule never finishes"

    phases = {row["phase"] for row in node.rows}
    assert phases == {"throttle_sweep", "longitudinal", "grip"}


def test_sysid_throttle_stays_in_range(tools_modules):
    sysid_drive, _ = tools_modules
    node = sysid_drive.SysIdDrive()
    node.speed = 2.0

    for step in range(0, 6000):
        node.elapsed = step * 0.025
        throttle = node._throttle_for(node.elapsed)
        if throttle is None:
            break
        assert -1.0 <= throttle <= 1.0


def test_sysid_records_a_steady_speed_for_every_throttle_step(tools_modules):
    sysid_drive, _ = tools_modules
    node = sysid_drive.SysIdDrive()
    node.speed = 2.0

    for step in range(0, 20000):
        node.elapsed = step * 0.025
        if node._throttle_for(node.elapsed) is None:
            break

    sweep = [r for r in node.rows if r["phase"] == "throttle_sweep"]
    assert len(sweep) == len(sysid_drive.THROTTLE_STEPS)
    assert {r["throttle"] for r in sweep} == set(sysid_drive.THROTTLE_STEPS)
