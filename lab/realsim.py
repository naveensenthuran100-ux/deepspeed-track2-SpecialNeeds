#!/usr/bin/env python3
"""Evaluate driver configurations on the REAL AutoDRIVE simulator, back to back.

    # inside the container, workspace sourced:
    python3 /hackathon/lab/realsim.py --params my.yaml --laps 5

This is a referee-shaped test bench, not the referee. It exists because
scripts/evaluate.sh restarts the bridge for every run, and after a restart the
simulator can reconnect while silently ignoring every command (it kept
applying a stale 0.52 throttle and drove itself into a wall: an 11-collision
DQ that said nothing about the driver). One persistent bridge, many runs:

  1. stop the driver and hold zero throttle - the simulator keeps applying the
     last command it received, so a killed driver otherwise leaves the car
     running away;
  2. reset the car to the grid exactly as the referee does (reset_command high
     0.5 s, then low) - rule 31 lets the referee publish it, never the entry,
     and the entry here never does;
  3. start the driver with the given parameters and wait until it publishes;
  4. time laps off the simulator's lap counter, count collisions, record
     telemetry at the odometry rate, and watch that the simulator's ACTUAL
     throttle follows the command;
  5. abort early on too many collisions, a stuck car, or ignored commands.

Final numbers still come from scripts/evaluate.sh. This is for finding them.
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32

NS = "/autodrive/roboracer_1"
BASE_PARAMS = "/hackathon/race_ws/src/team_driver/config/driver_params.yaml"
OUT = "/hackathon/results/realsim"

COLUMNS = ["t", "x", "y", "yaw", "speed", "yaw_rate", "imu_ax", "imu_ay",
           "throttle_cmd", "steer_cmd", "throttle", "steer", "lap", "collisions"]


def qos():
    return QoSProfile(durability=QoSDurabilityPolicy.VOLATILE,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def write_params(overrides, path, base=BASE_PARAMS):
    """The shipped yaml with `overrides` ({'line.grip_fraction': 0.8, ...}) applied."""
    doc = yaml.safe_load(open(base))
    params = doc["driver"]["ros__parameters"]
    for dotted, value in overrides.items():
        node = params
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    with open(path, "w") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False)
    return path


class Bench(Node):
    def __init__(self):
        super().__init__("realsim_bench")
        q = qos()
        self.v = {"lap": None, "collisions": None, "speed": 0.0, "throttle": None,
                  "steer": None, "throttle_cmd": None, "steer_cmd": None,
                  "imu_ax": float("nan"), "imu_ay": float("nan")}
        self.cmd_stamp = 0.0       # wall time of the last command NOT sent by us
        self.ours = False          # are we the ones publishing commands right now?
        self.rows = []
        self.recording = False
        self.create_subscription(Odometry, f"{NS}/odom", self._odom, q)
        self.create_subscription(Imu, f"{NS}/imu", self._imu, q)
        self.create_subscription(Int32, f"{NS}/lap_count",
                                 lambda m: self.v.__setitem__("lap", m.data), q)
        self.create_subscription(Int32, f"{NS}/collision_count",
                                 lambda m: self.v.__setitem__("collisions", m.data), q)
        self.create_subscription(Float32, f"{NS}/throttle",
                                 lambda m: self.v.__setitem__("throttle", m.data), q)
        self.create_subscription(Float32, f"{NS}/steering",
                                 lambda m: self.v.__setitem__("steer", m.data), q)
        self.create_subscription(Float32, f"{NS}/throttle_command", self._thr_cmd, q)
        self.create_subscription(Float32, f"{NS}/steering_command",
                                 lambda m: self.v.__setitem__("steer_cmd", m.data), q)
        self.thr_pub = self.create_publisher(Float32, f"{NS}/throttle_command", q)
        self.steer_pub = self.create_publisher(Float32, f"{NS}/steering_command", q)
        self.reset_pub = self.create_publisher(Bool, "/autodrive/reset_command", q)
        self.driver = None

    # -- callbacks ---------------------------------------------------------
    def _thr_cmd(self, msg):
        self.v["throttle_cmd"] = msg.data
        if not self.ours:
            self.cmd_stamp = time.time()

    def _imu(self, msg):
        self.v["imu_ax"] = msg.linear_acceleration.x
        self.v["imu_ay"] = msg.linear_acceleration.y

    def _odom(self, msg):
        tw = msg.twist.twist
        self.v["speed"] = math.hypot(tw.linear.x, tw.linear.y)
        if not self.recording:
            return
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        st = msg.header.stamp
        self.rows.append({
            "t": st.sec + st.nanosec * 1e-9, "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y, "yaw": yaw, "speed": self.v["speed"],
            "yaw_rate": tw.angular.z, "imu_ax": self.v["imu_ax"], "imu_ay": self.v["imu_ay"],
            "throttle_cmd": self.v["throttle_cmd"], "steer_cmd": self.v["steer_cmd"],
            "throttle": self.v["throttle"], "steer": self.v["steer"],
            "lap": self.v["lap"], "collisions": self.v["collisions"]})

    # -- primitives --------------------------------------------------------
    def pump(self, seconds, hold_zero=False):
        end = time.time() + seconds
        while time.time() < end:
            if hold_zero:
                self.ours = True
                self.thr_pub.publish(Float32(data=0.0))
                self.steer_pub.publish(Float32(data=0.0))
            rclpy.spin_once(self, timeout_sec=0.02)
        self.ours = False

    def stop_driver(self):
        if self.driver is not None and self.driver.poll() is None:
            try:
                os.killpg(self.driver.pid, signal.SIGINT)
                self.driver.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self.driver.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.driver = None
        # Any driver started some other way (run_on_mac.sh) must go too.
        subprocess.run("pkill -f '[t]eam_driver/lib/team_driver/driver'", shell=True)
        self.pump(0.8, hold_zero=True)

    def reset_car(self):
        end = time.time() + 0.5
        while time.time() < end:
            self.reset_pub.publish(Bool(data=True))
            self.pump(0.05, hold_zero=True)
        end = time.time() + 1.5
        while time.time() < end:
            self.reset_pub.publish(Bool(data=False))
            self.pump(0.05, hold_zero=True)

    def sim_obeys(self):
        """Command zero and check the ACTUAL throttle follows."""
        self.pump(1.0, hold_zero=True)
        thr = self.v["throttle"]
        return thr is not None and abs(thr) < 0.02

    def start_driver(self, params_path, log_path):
        log = open(log_path, "w")
        self.driver = subprocess.Popen(
            ["ros2", "run", "team_driver", "driver", "--ros-args", "--params-file", params_path],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        self.cmd_stamp = 0.0
        while time.time() - started < 20.0:
            self.pump(0.1)
            if self.cmd_stamp > started:
                return True
            if self.driver.poll() is not None:
                return False
        return False

    # -- one evaluation ----------------------------------------------------
    def evaluate(self, params_path, laps, tag, max_collisions=6, lap_timeout_s=30.0):
        os.makedirs(OUT, exist_ok=True)
        result = {"tag": tag, "params": params_path, "laps_wanted": laps,
                  "status": "?", "lap_times": [], "lap_collisions": []}
        self.stop_driver()
        self.reset_car()
        if not self.sim_obeys():
            result["status"] = "SIM_IGNORING_COMMANDS"
            return self._save(result)
        self.rows = []
        if not self.start_driver(params_path, f"{OUT}/{tag}.driver.log"):
            result["status"] = "DRIVER_DID_NOT_START"
            self.stop_driver()
            return self._save(result)

        self.recording = True
        lap0, coll0 = self.v["lap"] or 0, self.v["collisions"] or 0
        last_lap, last_coll = lap0, coll0
        t_lap = time.time()
        t_start = t_lap
        slow_since = None
        ignored_since = None
        status = "COMPLETE"
        while len(result["lap_times"]) < laps:
            self.pump(0.05)
            now = time.time()
            lap = self.v["lap"] if self.v["lap"] is not None else last_lap
            coll = (self.v["collisions"] or coll0) - coll0
            if lap > last_lap:
                result["lap_times"].append(round(now - t_lap, 3))
                result["lap_collisions"].append(coll - (last_coll - coll0))
                last_lap, last_coll, t_lap = lap, coll + coll0, now
            elif lap < last_lap:          # the simulator reset itself under us
                status = "LAP_COUNTER_WENT_BACKWARDS"
                break
            if coll > max_collisions:
                status = "TOO_MANY_COLLISIONS"
                break
            if now - t_lap > lap_timeout_s:
                status = "LAP_TIMEOUT"
                break
            if self.v["speed"] < 0.1 and now - t_start > 4.0:
                slow_since = slow_since or now
                if now - slow_since > 4.0:
                    status = "STUCK"
                    break
            else:
                slow_since = None
            thr, cmd = self.v["throttle"], self.v["throttle_cmd"]
            if thr is not None and cmd is not None and abs(thr - cmd) > 0.15:
                ignored_since = ignored_since or now
                if now - ignored_since > 2.0:
                    status = "SIM_IGNORING_COMMANDS"
                    break
            else:
                ignored_since = None
        self.recording = False
        # Count collisions NOW, while the run is the run. Once the driver is
        # stopped the car coasts at racing speed with zero throttle into the
        # next wall, and counting that charged every candidate a crash no
        # judged run would contain.
        raced_collisions = (self.v["collisions"] or coll0) - coll0
        self.stop_driver()

        # Timing quality. A Unity window that macOS throttles (App Nap, display
        # sleep, not frontmost) delivers odometry in 100-370 ms bursts; the car
        # then steers blind and runs wide, which reads exactly like a bad
        # config. Measured overnight 2026-10-02: 0 such gaps per run at first,
        # 34-120 per run later. A run like that says nothing about the driver.
        stamps = [r["t"] for r in self.rows]
        if len(stamps) > 10:
            gaps = [b - a for a, b in zip(stamps, stamps[1:])]
            gaps.sort()
            result["odom_median_ms"] = round(1000 * gaps[len(gaps) // 2], 1)
            result["odom_p95_ms"] = round(1000 * gaps[int(0.95 * (len(gaps) - 1))], 1)
            result["odom_gaps_100ms"] = sum(g > 0.1 for g in gaps)
            limit = int(os.environ.get("RS_MAX_GAPS", "3"))
            if result["odom_gaps_100ms"] > limit and status == "COMPLETE":
                status = "INVALID_TIMING"
            elif result["odom_gaps_100ms"] > limit:
                status += "+BAD_TIMING"
        result["status"] = status
        result["collisions"] = raced_collisions
        flying = result["lap_times"][1:]
        result["best"] = min(flying) if flying else None
        result["mean_flying"] = round(sum(flying) / len(flying), 3) if flying else None
        tel = f"{OUT}/{tag}.telemetry.csv"
        with open(tel, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        result["telemetry"] = tel
        return self._save(result)

    def _save(self, result):
        """Every outcome gets a result file - an early failure most of all."""
        with open(f"{OUT}/{result['tag']}.result.json", "w") as handle:
            json.dump(result, handle, indent=2)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default=BASE_PARAMS, help="driver params yaml (full file)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override on top of --params, e.g. line.grip_fraction=0.9 (repeatable)")
    p.add_argument("--laps", type=int, default=5)
    p.add_argument("--tag", default=time.strftime("run_%H%M%S"))
    a = p.parse_args()
    params = a.params
    if a.set:
        overrides = {}
        for item in a.set:
            key, value = item.split("=", 1)
            overrides[key] = yaml.safe_load(value)
        os.makedirs(OUT, exist_ok=True)
        params = write_params(overrides, f"{OUT}/{a.tag}.params.yaml", base=a.params)
    rclpy.init()
    bench = Bench()
    bench.pump(1.0)
    try:
        print(json.dumps(bench.evaluate(params, a.laps, a.tag), indent=2))
    finally:
        bench.stop_driver()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
