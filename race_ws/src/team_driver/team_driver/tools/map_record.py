#!/usr/bin/env python3
"""Record pose and LiDAR to an npz, for building our own occupancy grid.

    ros2 run team_driver map_record           # drive a few laps, then Ctrl-C
    python3 -m team_driver.tools.grid_build --log results/mapping.npz \
        --out maps/my_circuit

Deliberately not a rosbag: the only consumer is grid_build, an npz needs no
ROS to read, and the whole point of keeping the offline tools ROS-free is
that they can be tested and run anywhere.
"""

import math
import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from team_driver.core.lidar import beam_angles
from team_driver.driver import devkit_qos


class MapRecord(Node):

    def __init__(self):
        super().__init__("map_record")
        self.declare_parameter("vehicle_ns", "/autodrive/roboracer_1")
        self.declare_parameter("out", "/hackathon/results/mapping.npz")
        self.declare_parameter("min_spacing_m", 0.05)

        namespace = str(self.get_parameter("vehicle_ns").value).rstrip("/")
        qos = devkit_qos()
        self.create_subscription(LaserScan, f"{namespace}/lidar", self.scan_callback, qos)
        self.create_subscription(Odometry, f"{namespace}/odom", self.odom_callback, qos)

        self.pose = None
        self.poses = []
        self.scans = []
        self.angles = None
        self._last_kept = None
        self.get_logger().info(
            "recording. Drive a few laps - the driver node can do it - then Ctrl-C.")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        )

    def scan_callback(self, scan):
        if self.pose is None:
            return

        # One scan every few centimetres. At 40 Hz a stationary car would
        # otherwise contribute thousands of identical scans and weight the
        # grid towards wherever it happened to sit.
        spacing = float(self.get_parameter("min_spacing_m").value)
        if self._last_kept is not None:
            moved = math.hypot(self.pose[0] - self._last_kept[0],
                               self.pose[1] - self._last_kept[1])
            if moved < spacing:
                return

        if self.angles is None:
            self.angles = beam_angles(scan.angle_min, scan.angle_increment,
                                      len(scan.ranges))

        self.poses.append(self.pose)
        self.scans.append(np.asarray(scan.ranges, dtype=np.float32))
        self._last_kept = self.pose

        if len(self.poses) % 200 == 0:
            self.get_logger().info(f"{len(self.poses)} scans recorded")

    def save(self):
        if not self.poses:
            self.get_logger().warn("nothing recorded; not writing a file")
            return
        path = str(self.get_parameter("out").value)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            poses=np.array(self.poses, dtype=np.float64),
            scans=np.array(self.scans, dtype=np.float32),
            angles=np.asarray(self.angles, dtype=np.float64),
        )
        self.get_logger().info(
            f"wrote {len(self.poses)} scans to {path}. Now build the grid:\n"
            f"  python3 -m team_driver.tools.grid_build --log {path} "
            f"--out /hackathon/maps/my_circuit")


def main(args=None):
    rclpy.init(args=args)
    node = MapRecord()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
