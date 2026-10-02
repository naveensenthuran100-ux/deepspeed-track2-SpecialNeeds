#!/usr/bin/env python3
"""Record a real-simulator run at the odometry rate, for offline diagnosis.

    # inside the container, alongside a driven or scored run:
    python3 /hackathon/lab/record_run.py --seconds 240 --out /hackathon/results/telemetry_x.csv

One row per /odom message, carrying the latest value of everything else. The
point is to have, on the REAL car:

  * pose and speed along the lap, to compare with the planned profile;
  * commanded AND actual throttle and steering - the simulator publishes both,
    so actuator lag and saturation are visible directly;
  * IMU acceleration, which measures lateral grip and longitudinal
    acceleration outright instead of inferring them from a model.

Identification from a racing run is cleaner than a dedicated sysid drive: it
spends long stretches on straights at varied throttle with nothing in the way.
The dedicated drive's data was taken circling at full lock into a wall.

Lab tooling: needs rclpy, so it runs in the container; not shipped.
"""

import argparse
import csv
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32

COLUMNS = ["t", "x", "y", "yaw", "vx", "vy", "speed", "yaw_rate",
           "imu_ax", "imu_ay", "imu_wz",
           "throttle_cmd", "steer_cmd", "throttle", "steer",
           "lap_count", "collisions"]


def devkit_qos():
    return QoSProfile(durability=QoSDurabilityPolicy.VOLATILE,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=1)


class Recorder(Node):
    def __init__(self, ns):
        super().__init__("run_recorder")
        q = devkit_qos()
        self.latest = {k: float("nan") for k in COLUMNS}
        self.rows = []
        self.create_subscription(Odometry, f"{ns}/odom", self.on_odom, q)
        self.create_subscription(Imu, f"{ns}/imu", self.on_imu, q)
        for topic, key, kind in (("throttle_command", "throttle_cmd", Float32),
                                 ("steering_command", "steer_cmd", Float32),
                                 ("throttle", "throttle", Float32),
                                 ("steering", "steer", Float32),
                                 ("lap_count", "lap_count", Int32),
                                 ("collision_count", "collisions", Int32)):
            self.create_subscription(kind, f"{ns}/{topic}",
                                     lambda m, k=key: self.latest.__setitem__(k, float(m.data)), q)

    def on_imu(self, msg):
        self.latest["imu_ax"] = msg.linear_acceleration.x
        self.latest["imu_ay"] = msg.linear_acceleration.y
        self.latest["imu_wz"] = msg.angular_velocity.z

    def on_odom(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        row = dict(self.latest)
        row.update(t=stamp if stamp > 0 else time.time(),
                   x=msg.pose.pose.position.x, y=msg.pose.pose.position.y, yaw=yaw,
                   vx=vx, vy=vy, speed=math.hypot(vx, vy),
                   yaw_rate=msg.twist.twist.angular.z)
        self.rows.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ns", default="/autodrive/roboracer_1")
    args = parser.parse_args()

    rclpy.init()
    node = Recorder(args.ns)
    started = time.time()
    try:
        while time.time() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(node.rows)
        print(f"wrote {len(node.rows)} rows to {args.out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
