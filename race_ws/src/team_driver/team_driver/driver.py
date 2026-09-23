#!/usr/bin/env python3
"""The node the judges run. A thin adapter over `team_driver.core`.

Deliberately boring. It subscribes, converts messages into plain numbers,
calls `Planner.step`, publishes two Float32s, and draws a debug marker. There
is no driving logic in this file: it all lives in `core/`, which imports no
ROS and is therefore testable in milliseconds instead of in six-minute
simulator runs. That split is the reason the tests exist at all.

Run:
    ros2 run team_driver driver
    ros2 run team_driver driver --ros-args --params-file config/driver_params.yaml

Two things here are load-bearing and easy to break:

  * `devkit_qos()` - the bridge publishes RELIABLE / KEEP_LAST(1) / VOLATILE.
    A BEST_EFFORT subscriber never receives one message from it, silently,
    while `ros2 topic list` still shows the topic.
  * The callback never raises. If it did, the bridge would keep replying with
    the last command it got, which is the car driving on at its last throttle
    into whatever is in front of it.
"""

import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from team_driver.core.gap import GapConfig
from team_driver.core.lidar import beam_angles
from team_driver.core.planner import Planner, PlannerConfig
from team_driver.core.vehicle import Vehicle

#: Warn when a control step eats this much of the 25 ms budget (40 Hz LiDAR).
LOOP_BUDGET_S = 0.020


def devkit_qos() -> QoSProfile:
    """Match the QoS the AutoDRIVE bridge publishes and subscribes with.

    RELIABLE / KEEP_LAST(1) / VOLATILE. Get this wrong and the topics simply do
    not connect: a BEST_EFFORT subscriber will never see a RELIABLE publisher's
    messages, and `ros2 topic list` will still cheerfully show the topic.
    """
    return QoSProfile(
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )


#: ROS parameter name -> (GapConfig field, default). Flat dotted names so a
#: sweep can write one YAML file and never touch code.
_GAP_PARAMS = {
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

_SPEED_PARAMS = {
    "speed.kp": ("kp", 0.05),
    "speed.ki": ("ki", 0.15),
    "speed.integral_limit": ("integral_limit", 0.30),
    "speed.lift_band": ("lift_band", 0.5),
    "speed.brake_gain": ("brake_gain", 0.30),
}


class Driver(Node):

    def __init__(self):
        super().__init__("driver")

        self.declare_parameter("vehicle_ns", "/autodrive/roboracer_1")
        self.declare_parameter("vehicle_config", "")
        for name, (_, default) in _GAP_PARAMS.items():
            self.declare_parameter(name, default)
        for name, (_, default) in _SPEED_PARAMS.items():
            self.declare_parameter(name, default)

        vehicle = self._load_vehicle()
        config = PlannerConfig(
            gap=GapConfig(
                **{
                    field: self.get_parameter(name).value
                    for name, (field, _) in _GAP_PARAMS.items()
                }
            ),
            **{
                field: self.get_parameter(name).value
                for name, (field, _) in _SPEED_PARAMS.items()
            },
        )
        self.planner = Planner(vehicle, config)

        namespace = str(self.get_parameter("vehicle_ns").value).rstrip("/")
        qos = devkit_qos()

        self.throttle_pub = self.create_publisher(Float32, f"{namespace}/throttle_command", qos)
        self.steering_pub = self.create_publisher(Float32, f"{namespace}/steering_command", qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/driver/markers", 1)

        self.create_subscription(LaserScan, f"{namespace}/lidar", self.scan_callback, qos)
        self.create_subscription(Odometry, f"{namespace}/odom", self.odom_callback, qos)
        self.create_subscription(Point, f"{namespace}/ips", self.ips_callback, qos)

        self.position = None
        self.yaw = None
        self.speed = 0.0

        self._angles = None
        self._angle_signature = None
        self._last_stamp = None
        self._last_steer = 0.0
        self._worst_loop_s = 0.0
        self._marker_divisor = 0
        self._logged_failure = False

        if not vehicle.measured:
            self.get_logger().warn(
                "vehicle.json is UNMEASURED - limits are assumed from the published "
                "specification. Run tools/sysid_drive.py before trusting any lap time."
            )

    # -- setup -----------------------------------------------------------

    def _load_vehicle(self) -> Vehicle:
        path = str(self.get_parameter("vehicle_config").value or "")
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory

                path = os.path.join(
                    get_package_share_directory("team_driver"), "config", "vehicle.json"
                )
            except Exception:
                path = ""
        if path and os.path.exists(path):
            try:
                return Vehicle.load(path)
            except Exception as error:
                self.get_logger().error(f"could not read {path}: {error}; using defaults")
        return Vehicle.default()

    # -- state -----------------------------------------------------------

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def ips_callback(self, msg):
        """Position alone. /odom carries this plus velocity, so it wins when both
        arrive; this only fills the gap before the first odometry message."""
        if self.position is None:
            self.position = (msg.x, msg.y)

    # -- the control loop, once per scan (40 Hz) -------------------------

    def scan_callback(self, scan):
        started = time.perf_counter()
        try:
            angles = self._angles_for(scan)
            dt = self._dt_from(scan)
            result = self.planner.step(
                scan.ranges, angles, self.position, self.yaw, self.speed, dt
            )
            self._last_steer = result.steer_norm
            self.publish(result.steer_norm, result.throttle)
            self._draw(result)
        except Exception as error:  # never let the callback die
            if not self._logged_failure:
                self.get_logger().error(f"planner failed, coasting: {error}")
                self._logged_failure = True
            self.publish(self._last_steer, 0.0)

        elapsed = time.perf_counter() - started
        if elapsed > self._worst_loop_s:
            self._worst_loop_s = elapsed
            if elapsed > LOOP_BUDGET_S:
                self.get_logger().warn(
                    f"control step took {elapsed * 1e3:.1f} ms of a 25 ms budget"
                )

    def _angles_for(self, scan):
        """Beam angles, rebuilt only when the scan's geometry changes."""
        signature = (scan.angle_min, scan.angle_increment, len(scan.ranges))
        if signature != self._angle_signature:
            self._angles = beam_angles(scan.angle_min, scan.angle_increment, len(scan.ranges))
            self._angle_signature = signature
        return self._angles

    def _dt_from(self, scan):
        """Seconds since the last scan, on the simulator's clock.

        The message stamp rather than the wall clock, because a machine that
        cannot keep up runs slower in wall-clock terms while the simulation
        advances at a fixed physics timestep. Integral gain applied against
        wall-clock dt would behave differently on a loaded machine.
        """
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        previous, self._last_stamp = self._last_stamp, stamp
        if previous is None:
            return 0.0
        dt = stamp - previous
        return dt if 0.0 < dt < 1.0 else 0.0

    # -- output ----------------------------------------------------------

    def publish(self, steering, throttle):
        self.steering_pub.publish(Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def _draw(self, result):
        """Where the car thinks it is aiming. Add /driver/markers in RViz.

        Every tenth scan: 4 Hz is plenty to watch, and the marker is not worth
        any of the control budget.
        """
        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor:
            return

        angle = self.planner.vehicle.norm_to_steer(result.steer_norm)
        marker = Marker()
        marker.header.frame_id = "roboracer_1"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "team_driver"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.5 + 0.25 * result.v_target
        marker.scale.y, marker.scale.z = 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(angle / 2.0)
        marker.pose.orientation.w = math.cos(angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
