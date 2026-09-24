#!/usr/bin/env python3
"""Drive the identification manoeuvres and log them. Run this FIRST.

    ros2 run team_driver sysid_drive          # about two minutes
    python3 -m team_driver.tools.sysid --log results/sysid.csv \
        --out race_ws/src/team_driver/config/vehicle.json

Until this has run, config/vehicle.json says `measured: false` and every
limit in it is derived from the published specification rather than from the
car. The speed profile, the surrogate and every sweep result are relative to
those assumptions.

Three phases, and the steering is never scripted: the gap follower steers
throughout so the car stays on the circuit while the throttle does something
deliberately unnatural. A scripted straight-line test is not possible on a
6.3 x 18.2 m track.

  1. throttle sweep   hold each throttle until the speed settles, and record
                      the steady state. This is the actuator map.
  2. longitudinal     accelerate, lift, brake, repeated. This is a_accel_max,
                      a_coast and a_brake_max - and the ordering of the last
                      two is what makes "lift, do not stamp" meaningful.
  3. grip             let the gap follower drive with a rising speed cap and
                      watch lateral acceleration plateau. That plateau is
                      a_lat_max, which the whole speed profile hangs on.
"""

import csv
import math
import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from team_driver.core.gap import GapConfig, GapFollower
from team_driver.core.lidar import beam_angles
from team_driver.core.vehicle import Vehicle
from team_driver.driver import devkit_qos

THROTTLE_STEPS = (0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35)
HOLD_S = 3.0
SETTLE_S = 0.5          # average the tail of each hold, once the speed is steady

LONGITUDINAL_CYCLES = 3
ACCELERATE_S, COAST_S, BRAKE_S = 2.5, 2.0, 1.2
ACCELERATE_THROTTLE = 0.5

GRIP_S = 40.0
GRIP_V_START, GRIP_V_END = 1.5, 5.0


class SysIdDrive(Node):

    def __init__(self):
        super().__init__("sysid_drive")
        self.declare_parameter("vehicle_ns", "/autodrive/roboracer_1")
        self.declare_parameter("out", "/hackathon/results/sysid.csv")

        namespace = str(self.get_parameter("vehicle_ns").value).rstrip("/")
        qos = devkit_qos()
        self.throttle_pub = self.create_publisher(Float32, f"{namespace}/throttle_command", qos)
        self.steering_pub = self.create_publisher(Float32, f"{namespace}/steering_command", qos)
        self.create_subscription(LaserScan, f"{namespace}/lidar", self.scan_callback, qos)
        self.create_subscription(Odometry, f"{namespace}/odom", self.odom_callback, qos)

        self.vehicle = Vehicle.default()
        self.follower = GapFollower(self.vehicle, GapConfig())

        self.speed = 0.0
        self.yaw_rate = 0.0
        self.rows = []
        self.started = None
        self.elapsed = 0.0
        self._angles = None
        self._signature = None
        self._hold_samples = []
        self._finished = False

        self.get_logger().info(
            "system identification: about two minutes. The car will drive "
            "oddly on purpose. Do not score this run.")

    # -- state -----------------------------------------------------------

    def odom_callback(self, msg):
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.yaw_rate = msg.twist.twist.angular.z

    # -- the schedule ----------------------------------------------------

    def scan_callback(self, scan):
        if self._finished:
            return

        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        if self.started is None:
            self.started = stamp
        self.elapsed = stamp - self.started

        signature = (scan.angle_min, scan.angle_increment, len(scan.ranges))
        if signature != self._signature:
            self._angles = beam_angles(scan.angle_min, scan.angle_increment,
                                       len(scan.ranges))
            self._signature = signature

        # Steering always comes from the gap follower: the point is to make
        # the throttle do something unnatural, not to leave the circuit.
        steering = self.follower.step(scan.ranges, self._angles).steer_rad
        throttle = self._throttle_for(self.elapsed)

        if throttle is None:
            self._finish()
            return

        self.steering_pub.publish(
            Float32(data=float(np.clip(self.vehicle.steer_to_norm(steering), -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def _throttle_for(self, t):
        sweep_duration = len(THROTTLE_STEPS) * HOLD_S
        cycle = ACCELERATE_S + COAST_S + BRAKE_S
        longitudinal_duration = LONGITUDINAL_CYCLES * cycle

        if t < sweep_duration:
            index = int(t / HOLD_S)
            throttle = THROTTLE_STEPS[min(index, len(THROTTLE_STEPS) - 1)]
            within = t - index * HOLD_S
            if within > HOLD_S - SETTLE_S:
                self._hold_samples.append(self.speed)
            elif self._hold_samples:
                self._record_hold(index - 1)
            return throttle

        if self._hold_samples:
            self._record_hold(len(THROTTLE_STEPS) - 1)

        t -= sweep_duration
        if t < longitudinal_duration:
            within = t % cycle
            if within < ACCELERATE_S:
                throttle = ACCELERATE_THROTTLE
            elif within < ACCELERATE_S + COAST_S:
                throttle = 0.0
            else:
                throttle = -1.0
            self._row("longitudinal", throttle)
            return throttle

        t -= longitudinal_duration
        if t < GRIP_S:
            # Raise the speed cap steadily and let the gap follower drive.
            fraction = t / GRIP_S
            target = GRIP_V_START + fraction * (GRIP_V_END - GRIP_V_START)
            self._row("grip", 0.0, target=target)
            return self.vehicle.throttle_for_speed(target)

        return None

    # -- logging ---------------------------------------------------------

    def _record_hold(self, index):
        if 0 <= index < len(THROTTLE_STEPS) and self._hold_samples:
            self.rows.append({
                "phase": "throttle_sweep", "t": round(self.elapsed, 4),
                "throttle": THROTTLE_STEPS[index], "steering": 0.0,
                "speed": round(float(np.mean(self._hold_samples)), 4),
                "yaw_rate": 0.0,
            })
        self._hold_samples = []

    def _row(self, phase, throttle, target=0.0):
        self.rows.append({
            "phase": phase, "t": round(self.elapsed, 4),
            "throttle": round(float(throttle), 4), "steering": round(float(target), 4),
            "speed": round(float(self.speed), 4),
            "yaw_rate": round(float(self.yaw_rate), 4),
        })

    def _finish(self):
        self._finished = True
        self.throttle_pub.publish(Float32(data=0.0))
        self.steering_pub.publish(Float32(data=0.0))

        path = str(self.get_parameter("out").value)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["phase", "t", "throttle", "steering", "speed", "yaw_rate"])
            writer.writeheader()
            writer.writerows(self.rows)

        self.get_logger().info(
            f"wrote {len(self.rows)} samples to {path}. Now fit them:\n"
            f"  python3 -m team_driver.tools.sysid --log {path} "
            f"--out /hackathon/race_ws/src/team_driver/config/vehicle.json")


def main(args=None):
    rclpy.init(args=args)
    node = SysIdDrive()
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
