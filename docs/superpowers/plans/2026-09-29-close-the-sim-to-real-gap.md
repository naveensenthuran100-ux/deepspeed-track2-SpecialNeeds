# Close the Sim-to-Real Gap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real car faster than today's best (CEM #480: 12.145 s best, 12.44 s mean over 10 laps, 0 collisions) by re-measuring the car from a clean racing trace, teaching the surrogate the difference between what the planner believes and what the car does, and re-searching against that.

**Architecture:** One recorded real run → re-identify throttle map, speed lag and braking from it (`tools/sysid.py`) → surrogate gains a `truth_vehicle` separate from the planner's belief, plus throttle-map and pose-latency perturbations (`tools/surrogate.py`) → a calibration gate proves the surrogate reproduces the real run before anything is searched → CEM re-search → one real run → adopt.

**Tech Stack:** Python 3 (numpy, scipy, pyyaml), ROS 2 Humble (rclpy, container only), pytest, AutoDRIVE simulator (native macOS app).

**Spec:** `docs/HANDOVER.md` (§4 measured vs assumed, §5 grip ladder) and the findings below, which this plan argues from.

## Findings this plan argues from (all measured 2026-09-29)

| # | Finding | Evidence |
|---|---|---|
| F1 | The throttle→speed map in `vehicle.json` (`v = 39.80u − 19.90u²`) is ~60% too steep. Clean sysid points give `v ≈ 24.6u`. | `results/sysid.csv` rows u=0.08/0.12/0.16/0.25 → 1.99/2.94/3.95/6.03 m/s |
| F2 | The rest of `sysid.csv` is contaminated: the "longitudinal" phase logs steering 0.0 while yaw rate implies full lock (R≈0.56 m), and speed resets every 0.65 s (wall). `a_accel_max`, `a_brake_max`, `a_coast` are fitted to that. | same file, rows 9–60 |
| F3 | The surrogate's "true" car uses the same map as the controller's feed-forward (`surrogate.py:182`), and no perturbation touches the map (`scaled_vehicle`, `surrogate.py:157`). Feed-forward is perfect in the surrogate and ~38% short on the real car. | code |
| F4 | Giving the surrogate car the measured map moves #480 from 11.52 → 11.90 s/lap and shipped from 12.20 → 13.15. Real #480: 12.44 mean. ~0.45 s/lap for #480 remains unexplained. | `scratchpad/map_hypothesis.py` output |
| F5 | Odometry arrived at ~18 Hz in the recorder vs the 40 Hz control loop: the planner may steer on a pose up to ~50 ms old. The surrogate never models that. | `results/.rec_test.csv` (91 rows / 5 s) |
| F6 | After a bridge restart the simulator can keep PUBLISHING but stop APPLYING commands: actual throttle froze at 0.519 while the driver commanded −1.0. That run DQ'd with 11 collisions in 10 s. | `results/telemetry_cem480.csv` |

## Global Constraints

- `team_driver/core/` imports no ROS (enforced by `tests/test_architecture.py`).
- No new dependencies: numpy, scipy, pyyaml only. `requirements.txt` / `apt-packages.txt` stay empty.
- `ros2 run team_driver driver` must keep working. `scripts/` is protected — never edit it.
- `PlannerConfig` defaults and `core/params.py` defaults must agree (`tests/test_params.py`).
- `a_lat_max` stays **5.5717** — measured by the grip ladder. A telemetry refit must never overwrite it.
- Real-simulator validation is **one** 10-lap run (`evaluate.sh --runs 1`). A run with real-time factor outside 0.95–1.05 is void.
- Submission: **18 Oct 2026, 23:59 SGT**, commit timestamp on the submitted branch.
- Tests: `cd race_ws/src/team_driver && <venv>/bin/python -m pytest -q`. Any venv works: `python3 -m venv venv && ./venv/bin/pip install numpy scipy pytest pyyaml`.
- Shipped files (`config/driver_params.yaml`, `config/vehicle.json`, `maps/icra26_compete_raceline.csv`) change only in Task 7, after a real run has confirmed the result.

## Review Focus

1. **Simulator ignoring commands** (F6) — a run where actual throttle does not follow the command must be reported invalid, never analysed as "the car is slow". Pinned by Task 1's command-following check in `analyse_run.py`.
2. **Collision and reset transients in the fit** — samples just after a collision or a grid reset are a car in a wall, not dynamics (exactly F2's failure). Pinned by `test_clean_mask_drops_samples_after_collisions_and_resets` (Task 2).
3. **Too little clean data** — a trace with almost no straight running must make the fitter refuse, not write a vehicle.json. Pinned by `test_speed_dynamics_refuses_too_few_samples` (Task 2).
4. **A non-monotonic or negative map** — the `v_max = −60.45` bug must not return through the new fitter. Pinned by `test_speed_dynamics_map_is_monotonic_on_noisy_data` (Task 2).
5. **An old vehicle.json without `speed_lag`** — the file shipped today must still load. Pinned by `test_vehicle_json_without_speed_lag_loads_with_default` (Task 2 Step 0).

---

### Task 0: Prerequisite — simulator accepting commands (human)

The simulator must be connected **and applying commands** (F6).

- [ ] **Step 1:** In the AutoDRIVE window click **Disconnect → Connect**, then set the vehicle to **Autonomous**.
- [ ] **Step 2:** Confirm to the agent that it is done. Nothing else in this plan touches the GUI until Task 1 Step 4 and Task 6.

---

### Task 1: One clean recorded baseline run

**Files:**
- Create: `lab/preflight.py`
- Modify: `lab/analyse_run.py` (add command-following and pose-rate report)
- Commit (already written, untracked): `lab/record_run.py`, `lab/analyse_run.py`
- Output: `results/telemetry_cem480_baseline.csv`, `results/analysis_cem480_baseline.txt`

**Interfaces:**
- Produces: `results/telemetry_cem480_baseline.csv` with columns `t,x,y,yaw,vx,vy,speed,yaw_rate,imu_ax,imu_ay,imu_wz,throttle_cmd,steer_cmd,throttle,steer,lap_count,collisions` — consumed by Tasks 2 and 4.
- Produces: `POSE_STEPS` — the integer "control steps of 25 ms" printed on `analyse_run.py`'s `odometry period` line — consumed by Task 5.

- [ ] **Step 1: Write `lab/preflight.py`**

```python
#!/usr/bin/env python3
"""Is the simulator APPLYING commands, not just publishing? Run before a scored run.

After a bridge restart the simulator can reconnect and keep publishing
telemetry while ignoring every command: its actual throttle stays frozen at
the last value it received and the car drives itself into a wall. The
referee then scores a disqualification that says nothing about the driver.

This publishes zero throttle and zero steering for two seconds and checks the
simulator's ACTUAL throttle and steering follow. It also stops a car that is
still running on a stale command.

    python3 /hackathon/lab/preflight.py        # inside the container
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from std_msgs.msg import Float32

NS = "/autodrive/roboracer_1"


def main():
    q = QoSProfile(durability=QoSDurabilityPolicy.VOLATILE,
                   reliability=QoSReliabilityPolicy.RELIABLE,
                   history=QoSHistoryPolicy.KEEP_LAST, depth=1)
    rclpy.init()
    node = Node("preflight")
    seen = {"throttle": None, "steering": None}
    for key in seen:
        node.create_subscription(Float32, f"{NS}/{key}",
                                 lambda m, k=key: seen.__setitem__(k, m.data), q)
    thr = node.create_publisher(Float32, f"{NS}/throttle_command", q)
    steer = node.create_publisher(Float32, f"{NS}/steering_command", q)
    started = time.time()
    while time.time() - started < 2.0:
        thr.publish(Float32(data=0.0))
        steer.publish(Float32(data=0.0))
        rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()
    if seen["throttle"] is None:
        print("FAIL: no telemetry at all - is the simulator connected?")
        return 1
    ok = abs(seen["throttle"]) < 0.02 and abs(seen["steering"]) < 0.02
    print(f"{'PASS' if ok else 'FAIL'}: actual throttle {seen['throttle']:.3f}, "
          f"steering {seen['steering']:.3f} after commanding 0.0")
    if not ok:
        print("The simulator is ignoring commands: Disconnect -> Connect -> Autonomous.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Add command-following and pose-rate checks to `lab/analyse_run.py`**

In `main()`, directly after the line `print(f"{len(laps)} complete laps: " + ...)`, insert:

```python
    # A run where the simulator ignored the driver is invalid, not slow.
    live = np.isfinite(d["throttle_cmd"]) & np.isfinite(d["throttle"])
    follow = np.abs(d["throttle"][live] - d["throttle_cmd"][live]) < 0.05
    print(f"command following: {100*follow.mean():.0f}% of {live.sum()} samples "
          f"within 0.05 of the command")
    if live.sum() == 0 or follow.mean() < 0.95:
        print("INVALID RUN: the simulator did not apply the driver's commands.")
        return
    odom_dt = np.median(np.diff(d["t"]))
    print(f"odometry period: median {odom_dt*1000:.1f} ms ({1/odom_dt:.1f} Hz); "
          f"the planner's pose can be up to {odom_dt*1000:.0f} ms old "
          f"= {round(odom_dt / 0.025)} control steps of 25 ms")
```

- [ ] **Step 3: Start a bridge, run the preflight**

```bash
docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc "cd /hackathon/race_ws && source install/local_setup.bash && nohup ros2 launch roboracer_referee simulator.launch.py simulator:=false rviz:=false > /hackathon/results/bridge.log 2>&1 &"
```
```bash
docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/lab/preflight.py"
```
Expected: `PASS: actual throttle 0.000, steering 0.000 after commanding 0.0`. On FAIL: Task 0 again, then repeat this step.

- [ ] **Step 4: Recorder first, then one scored run**

```bash
docker exec -d roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/lab/record_run.py --seconds 260 --out /hackathon/results/telemetry_cem480_baseline.csv"
```
```bash
./scripts/evaluate.sh --team cem480 --runs 1 --no-simulator --driver-params /hackathon/lab/shortlist/cem/driver_params_480.yaml
```
Expected: `status : COMPLETE`, `collisions : 0/10`, real-time factor 0.95–1.05. If the referee reports `No telemetry` the simulator dropped the connection on the bridge restart: Task 0, then repeat Steps 3–4.

- [ ] **Step 5: Analyse**

```bash
python3 lab/analyse_run.py results/telemetry_cem480_baseline.csv --line lab/shortlist/cem/raceline_m0300.csv --grip 0.8061 --v-max 11.965 | tee results/analysis_cem480_baseline.txt
```
Expected: `command following: ≥95%`, ≥10 complete laps, the per-stretch time-loss table, and the `odometry period` line. Note `POSE_STEPS` (the "control steps" integer on that line) and the referee result file name (`results/cem480__<id>.json`).

- [ ] **Step 6: Commit**

```bash
git add lab/preflight.py lab/record_run.py lab/analyse_run.py results/telemetry_cem480_baseline.csv results/analysis_cem480_baseline.txt results/cem480__*.json
git commit -m "feat(lab): record and analyse a real run; baseline telemetry for #480"
```

---

### Task 2: Re-identify the car from racing telemetry

**Files:**
- Modify: `race_ws/src/team_driver/team_driver/core/vehicle.py` (one field)
- Modify: `race_ws/src/team_driver/team_driver/tools/sysid.py`
- Test: `race_ws/src/team_driver/tests/test_sysid.py`, `race_ws/src/team_driver/tests/test_vehicle.py`

**Interfaces:**
- Consumes: telemetry CSV from Task 1; `surrogate.longitudinal_accel(vehicle, throttle, speed, lag)` (tests only).
- Produces:
  - `Vehicle.speed_lag: float = 2.0` (1/s) — consumed by Task 3's surrogate
  - `forward_accel(t, v) -> np.ndarray` (same length, last element NaN)
  - `clean_telemetry_mask(t, yaw_rate, collisions, lap_count, settle_s=1.0) -> np.ndarray[bool]`
  - `fit_speed_dynamics(throttle, speed, accel) -> {"throttle_v_coeffs": [c1, c2], "speed_lag": float}` — raises `ValueError` below `MIN_DYNAMICS_SAMPLES`
  - `fit_brake(throttle, speed, accel) -> float` (m/s² at full brake; 0.0 if no samples)
  - `load_telemetry(path) -> dict[str, np.ndarray]`
  - `fit_telemetry(tel, base: Vehicle) -> Vehicle`
  - CLI: `python -m team_driver.tools.sysid --telemetry CSV --base vehicle.json --out vehicle.json`

- [ ] **Step 0: `Vehicle.speed_lag`, test first**

Append to `tests/test_vehicle.py`:

```python
def test_vehicle_json_without_speed_lag_loads_with_default(tmp_path):
    """The vehicle.json shipped before speed_lag existed must still load."""
    import json
    path = tmp_path / "old.json"
    data = Vehicle.default().to_dict()
    data.pop("speed_lag", None)
    path.write_text(json.dumps(data))
    assert Vehicle.load(str(path)).speed_lag == 2.0
```

Run: `<venv>/bin/python -m pytest tests/test_vehicle.py -q` — Expected: FAIL, `AttributeError: 'Vehicle' object has no attribute 'speed_lag'`.

In `core/vehicle.py`, after `v_max: float = TOP_SPEED_MPS`, add:

```python
    #: 1/s. How fast speed approaches the throttle's steady-state speed; the
    #: surrogate's longitudinal model, identified by tools/sysid.fit_speed_dynamics.
    speed_lag: float = 2.0
```

Run the same command — Expected: PASS.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_sysid.py`:

```python
from dataclasses import replace

from team_driver.tools.surrogate import longitudinal_accel
from team_driver.tools.sysid import (clean_telemetry_mask, fit_brake, fit_speed_dynamics,
                                     fit_telemetry, forward_accel)


def _drive(truth, lag, throttles, dt=0.05, hold_s=1.5):
    """Integrate the surrogate's longitudinal model under a throttle schedule."""
    t, v, ts, vs, us = 0.0, 0.0, [], [], []
    for u in throttles:
        for _ in range(int(hold_s / dt)):
            ts.append(t); vs.append(v); us.append(u)
            v = max(0.0, v + longitudinal_accel(truth, u, v, lag) * dt)
            t += dt
    return np.array(ts), np.array(vs), np.array(us)


def _real_like():
    truth = Vehicle.default()
    truth.throttle_v_coeffs = [24.6, 0.0]
    truth.a_accel_max = 5.0
    truth.a_coast = 3.0
    truth.a_brake_max = 4.0
    return truth


def test_speed_dynamics_recovers_map_and_lag():
    truth = _real_like()
    rng = np.random.RandomState(1)
    t, v, u = _drive(truth, 3.0, rng.uniform(0.08, 0.5, 60))
    fit = fit_speed_dynamics(u, v, forward_accel(t, v))
    fitted = replace(truth, throttle_v_coeffs=fit["throttle_v_coeffs"])
    for throttle in (0.15, 0.3, 0.45):
        assert fitted.speed_for_throttle(throttle) == pytest.approx(
            truth.speed_for_throttle(throttle), rel=0.05)
    assert fit["speed_lag"] == pytest.approx(3.0, rel=0.1)


def test_speed_dynamics_map_is_monotonic_on_noisy_data():
    truth = _real_like()
    rng = np.random.RandomState(2)
    t, v, u = _drive(truth, 3.0, rng.uniform(0.08, 0.5, 60))
    v = v + rng.normal(0.0, 0.05, v.shape)
    fit = fit_speed_dynamics(u, v, forward_accel(t, v))
    fitted = replace(truth, throttle_v_coeffs=fit["throttle_v_coeffs"])
    speeds = [fitted.speed_for_throttle(x) for x in np.linspace(0.0, 1.0, 21)]
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))
    assert speeds[-1] > 0.0


def test_speed_dynamics_refuses_too_few_samples():
    truth = _real_like()
    t, v, u = _drive(truth, 3.0, [0.3], hold_s=1.0)   # 20 samples
    with pytest.raises(ValueError):
        fit_speed_dynamics(u, v, forward_accel(t, v))


def test_brake_recovers_full_brake_deceleration():
    truth = _real_like()
    t, v, u = _drive(truth, 3.0, [0.45, -0.3, 0.45, -0.6, 0.45, -1.0] * 4, hold_s=0.6)
    assert fit_brake(u, v, forward_accel(t, v)) == pytest.approx(4.0, rel=0.05)


def test_clean_mask_drops_samples_after_collisions_and_resets():
    t = np.arange(0.0, 20.0, 0.05)
    yaw_rate = np.zeros_like(t)
    yaw_rate[(t > 2.0) & (t < 3.0)] = 1.0                     # cornering
    collisions = np.where(t >= 5.0, 1.0, 0.0)                 # contact at t=5
    lap_count = np.where(t >= 10.0, 0.0, 3.0)                 # reset at t=10
    keep = clean_telemetry_mask(t, yaw_rate, collisions, lap_count, settle_s=1.0)
    assert not keep[(t > 2.0) & (t < 3.0)].any()
    assert not keep[(t >= 5.0) & (t < 6.0)].any()
    assert not keep[(t >= 10.0) & (t < 12.0)].any()
    assert keep[(t > 7.0) & (t < 9.0)].all()


def test_fit_telemetry_keeps_the_measured_grip():
    truth = _real_like()
    rng = np.random.RandomState(3)
    t, v, u = _drive(truth, 3.0, list(rng.uniform(0.08, 0.5, 40)) + [-0.5, 0.4] * 10)
    tel = {"t": t, "speed": v, "throttle": u, "yaw_rate": np.zeros_like(t),
           "collisions": np.zeros_like(t), "lap_count": np.zeros_like(t)}
    base = Vehicle.default()
    base.a_lat_max = 5.5717
    fitted = fit_telemetry(tel, base)
    assert fitted.a_lat_max == 5.5717
    assert fitted.measured
    assert fitted.speed_for_throttle(0.3) == pytest.approx(truth.speed_for_throttle(0.3), rel=0.05)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd race_ws/src/team_driver && <venv>/bin/python -m pytest tests/test_sysid.py -q`
Expected: FAIL — `ImportError: cannot import name 'clean_telemetry_mask'`.

- [ ] **Step 3: Implement** — add to `tools/sysid.py` after `fit_grip` (and `from dataclasses import replace` at the top):

```python
#: Below this many clean samples a dynamics fit is noise, and a vehicle.json
#: fitted from noise is worse than the one it replaces.
MIN_DYNAMICS_SAMPLES = 100

#: Forward differences across a gap longer than this are not acceleration.
_MAX_STEP_S = 0.2


def forward_accel(t, v):
    """dv/dt as (v[k+1] - v[k]) / dt, aligned with sample k; NaN where undefined.

    Forward, not central: the throttle applied at sample k is what produced
    the change to k+1, so pairing them is what makes the regression exact.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    a = np.full(v.shape, np.nan)
    dt = np.diff(t)
    ok = (dt > 0.0) & (dt < _MAX_STEP_S)
    a[:-1][ok] = np.diff(v)[ok] / dt[ok]
    return a


def clean_telemetry_mask(t, yaw_rate, collisions, lap_count, settle_s: float = 1.0):
    """Samples that are free running: straight, and not just after contact or a reset.

    The dedicated sysid drive was ruined by exactly these: it logged a car
    circling at full lock into a wall and fitted its limits to that.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    keep = np.isfinite(t) & (np.abs(np.nan_to_num(yaw_rate, nan=np.inf)) < 0.5)

    def counter(values):
        return np.nan_to_num(np.asarray(values, dtype=np.float64).ravel(), nan=0.0)

    contacts = t[1:][np.diff(counter(collisions)) != 0]
    resets = t[1:][np.diff(counter(lap_count)) < 0]      # a lap completing is normal
    for when, hold in [(w, settle_s) for w in contacts] + [(w, 2.0 * settle_s) for w in resets]:
        keep &= ~((t >= when) & (t < when + hold))
    return keep


def fit_speed_dynamics(throttle, speed, accel):
    """Throttle map and speed lag from free running: a = lag * (c1*u + c2*u^2 - v).

    That is the surrogate's own longitudinal model outside its acceleration
    and coast caps, so the fit identifies exactly the parameters it uses.
    Samples at the caps are dropped: there the acceleration says nothing
    about the map.
    """
    u = np.asarray(throttle, dtype=np.float64).ravel()
    v = np.asarray(speed, dtype=np.float64).ravel()
    a = np.asarray(accel, dtype=np.float64).ravel()
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(a) & (u > 0.02) & (v > 0.3)
    if ok.sum() >= MIN_DYNAMICS_SAMPLES:
        hi, lo = np.percentile(a[ok], 95), np.percentile(a[ok], 5)
        ok &= (a < 0.9 * hi) & (a > 0.9 * lo)
    if ok.sum() < MIN_DYNAMICS_SAMPLES:
        raise ValueError(f"only {int(ok.sum())} free-running samples; "
                         f"need {MIN_DYNAMICS_SAMPLES} to fit the throttle map")
    design = np.stack([u[ok], u[ok] ** 2, v[ok]], axis=1)
    (k1, k2, kv), *_ = np.linalg.lstsq(design, a[ok], rcond=None)
    lag = -float(kv)
    if lag <= 0.1:
        raise ValueError(f"fitted speed lag {lag:.3f} is not a physical first-order lag")
    c1, c2 = float(k1) / lag, float(k2) / lag
    # Same monotonic guard as fit_actuator_map: see the v_max = -60 bug there.
    c1 = max(c1, _MIN_SLOPE)
    return {"throttle_v_coeffs": [c1, max(c2, -c1 / 2.0)], "speed_lag": lag}


def fit_brake(throttle, speed, accel) -> float:
    """Full-brake deceleration, from a = -a_brake * |u| through the origin."""
    u = np.asarray(throttle, dtype=np.float64).ravel()
    v = np.asarray(speed, dtype=np.float64).ravel()
    a = np.asarray(accel, dtype=np.float64).ravel()
    ok = np.isfinite(u) & np.isfinite(a) & (u < -0.05) & (v > 0.5) & (a < 0.0)
    if not ok.any():
        return 0.0
    return float(np.sum(np.abs(u[ok]) * -a[ok]) / np.sum(u[ok] ** 2))


def load_telemetry(path: str) -> Dict:
    """The columns of a lab/record_run.py CSV that identification needs."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    def column(name):
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan
                         for r in rows], dtype=np.float64)

    return {k: column(k) for k in ("t", "speed", "throttle", "yaw_rate",
                                   "collisions", "lap_count")}


def fit_telemetry(tel: Dict, base: Vehicle) -> Vehicle:
    """A Vehicle re-identified from a racing run, keeping `base`'s grip.

    Grip is NOT refitted: a racing run at a safety margin never reaches the
    limit, so what it measures is a floor, and the grip ladder already
    measured the real edge on the simulator.
    """
    a = forward_accel(tel["t"], tel["speed"])
    keep = clean_telemetry_mask(tel["t"], tel["yaw_rate"], tel["collisions"], tel["lap_count"])
    u, v, a = tel["throttle"][keep], tel["speed"][keep], a[keep]

    dynamics = fit_speed_dynamics(u, v, a)
    vehicle = replace(base, throttle_v_coeffs=dynamics["throttle_v_coeffs"],
                      speed_lag=dynamics["speed_lag"], measured=True)
    vehicle.v_max = vehicle.speed_for_throttle(1.0)

    brake = fit_brake(u, v, a)
    if brake > 0.0:
        vehicle.a_brake_max = brake
    driving = np.isfinite(a) & (u > 0.05)
    if driving.sum() >= MIN_DYNAMICS_SAMPLES:
        vehicle.a_accel_max = float(np.percentile(a[driving], LIMIT_PERCENTILE))
    lifted = np.isfinite(a) & (np.abs(u) <= 0.05) & (v > 0.5)
    if lifted.sum() >= 20:
        vehicle.a_coast = float(np.percentile(-a[lifted], LIMIT_PERCENTILE))
    vehicle.a_coast = min(vehicle.a_coast, vehicle.a_brake_max)
    return vehicle
```

Replace the start of `main()` (the parser and `log = load_log(...)` block) with:

```python
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="CSV written by sysid_drive")
    parser.add_argument("--telemetry", help="CSV written by lab/record_run.py during a racing run")
    parser.add_argument("--base", help="with --telemetry: vehicle.json whose grip is kept")
    parser.add_argument("--out", required=True, help="where to write vehicle.json")
    args = parser.parse_args(argv)

    if args.telemetry:
        base = Vehicle.load(args.base) if args.base else Vehicle.default()
        before = Vehicle.load(args.base) if args.base else Vehicle.default()
        vehicle = fit_telemetry(load_telemetry(args.telemetry), base)
        vehicle.save(args.out)
        print(f"re-identified from {args.telemetry} (grip kept from {args.base})\n")
        print(f"{'parameter':<20} {'before':>10} {'after':>10}")
        for name in ("a_accel_max", "a_brake_max", "a_coast", "v_max", "speed_lag", "a_lat_max"):
            print(f"{name:<20} {getattr(before, name):>10.2f} {getattr(vehicle, name):>10.2f}")
        print(f"{'throttle_v_coeffs':<20} {str([round(c, 2) for c in before.throttle_v_coeffs]):>10} "
              f"{str([round(c, 2) for c in vehicle.throttle_v_coeffs]):>10}")
        print(f"\nwrote {args.out}")
        return 0
    if not args.log:
        parser.error("one of --log or --telemetry is required")

    log = load_log(args.log)
```

(The rest of `main()` is unchanged.)

- [ ] **Step 4: Run tests**

Run: `cd race_ws/src/team_driver && <venv>/bin/python -m pytest tests/test_sysid.py -q`
Expected: all pass. Then the full suite: `<venv>/bin/python -m pytest -q` — all pass.

- [ ] **Step 5: Fit the real car**

```bash
cd race_ws/src/team_driver && <venv>/bin/python -m team_driver.tools.sysid --telemetry ../../../results/telemetry_cem480_baseline.csv --base config/vehicle.json --out ../../../results/vehicle_measured.json
```
Expected: a before/after table; the map's slope near 24–26 (F1); `a_lat_max` unchanged at 5.57. Written to `results/`, **not** to `config/` — shipping it is Task 7.

- [ ] **Step 6: Commit**

```bash
git add race_ws/src/team_driver/team_driver/core/vehicle.py race_ws/src/team_driver/team_driver/tools/sysid.py race_ws/src/team_driver/tests/test_sysid.py race_ws/src/team_driver/tests/test_vehicle.py results/vehicle_measured.json
git commit -m "feat(sysid): re-identify throttle map, speed lag and braking from a racing run"
```

---

### Task 3: Surrogate — truth separate from belief, map error, pose latency

**Files:**
- Modify: `race_ws/src/team_driver/team_driver/tools/surrogate.py`
- Test: `race_ws/src/team_driver/tests/test_surrogate.py`

**Interfaces:**
- Consumes: `Vehicle.speed_lag` (Task 2 Step 0).
- Produces:
  - `scaled_vehicle(vehicle, scale, map_scale=1.0) -> Vehicle`
  - `Perturbation.map_scale: float = 1.0`, `Perturbation.pose_delay_steps: int = 0`
  - `simulate(..., speed_lag: Optional[float] = None, truth_vehicle: Optional[Vehicle] = None)`: dynamics from `truth_vehicle` (else `vehicle`), then perturbed; lag from the truth vehicle unless given.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_surrogate.py`:

```python
def _start(line):
    yaw = float(np.arctan2(line.tangent[0][1], line.tangent[0][0]))
    return (line.xy[0][0], line.xy[0][1], yaw)


def _lap(real_track, believed, truth=None, perturbation=None):
    grid, line = real_track
    planner = Planner(believed, PlannerConfig())
    profile = speed_profile(line.curvature, line.s, believed, grip_fraction=0.7,
                            v_max=8.0, total_length=line.length)
    planner.load_path(line, profile)
    return simulate(planner, believed, grid, start=_start(line), distance_m=line.length,
                    max_time_s=60.0, truth_vehicle=truth, perturbation=perturbation)


def test_truth_vehicle_drives_the_dynamics_not_the_planners_belief(real_track):
    """The planner's feed-forward uses ITS map; the car obeys the truth's. A
    truth with a flatter map than believed must be slower - which is the real
    car's situation, and which the surrogate could not express before."""
    believed = Vehicle.default()
    flat = dataclasses.replace(believed, throttle_v_coeffs=[c * 0.6 for c in believed.throttle_v_coeffs])
    honest, fooled = _lap(real_track, believed), _lap(real_track, believed, truth=flat)
    # Both must finish: a run that ends early off the track also has a small elapsed_s.
    assert honest.status == fooled.status == "COMPLETE"
    assert fooled.elapsed_s > honest.elapsed_s


def test_map_scale_scales_only_the_throttle_map():
    from team_driver.tools.surrogate import scaled_vehicle
    base = Vehicle.default()
    v = scaled_vehicle(base, 1.0, map_scale=0.5)
    assert v.throttle_v_coeffs == pytest.approx([c * 0.5 for c in base.throttle_v_coeffs])
    assert v.a_lat_max == base.a_lat_max and v.a_brake_max == base.a_brake_max


def test_pose_delay_changes_the_run_and_stays_finite(real_track):
    from team_driver.tools.surrogate import Perturbation
    believed = Vehicle.default()
    fresh = _lap(real_track, believed, perturbation=Perturbation(seed=1))
    stale = _lap(real_track, believed, perturbation=Perturbation(seed=1, pose_delay_steps=3))
    assert not np.allclose(fresh.trace["x"][:200], stale.trace["x"][:200])
    for key in ("x", "y", "v", "yaw"):
        assert np.all(np.isfinite(stale.trace[key]))


def test_speed_lag_comes_from_the_truth_vehicle():
    quick = dataclasses.replace(Vehicle.default(), speed_lag=8.0)
    slow = dataclasses.replace(Vehicle.default(), speed_lag=0.5)
    grid = _corridor()
    def run(truth):
        return simulate(Planner(Vehicle.default(), PlannerConfig()), Vehicle.default(), grid,
                        start=(1.0, 1.5, 0.0), distance_m=10.0, max_time_s=30.0,
                        truth_vehicle=truth)
    assert run(quick).elapsed_s < run(slow).elapsed_s
```

Add `import dataclasses` to the imports at the top of `tests/test_surrogate.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `<venv>/bin/python -m pytest tests/test_surrogate.py -q`
Expected: FAIL — `TypeError: simulate() got an unexpected keyword argument 'truth_vehicle'`.

- [ ] **Step 3: Implement** — in `tools/surrogate.py`:

`scaled_vehicle` becomes:

```python
def scaled_vehicle(vehicle: Vehicle, scale: float, map_scale: float = 1.0) -> Vehicle:
    """A vehicle with its LIMITS scaled, its throttle map optionally scaled,
    and its geometry untouched.

    `map_scale` is feed-forward error: the throttle the controller believes
    holds a speed holds a different one on the real car. It was the largest
    single sim-to-real error found (the fitted map was ~60% too steep) and no
    perturbation modelled it.

    Wheelbase, width and steering lock are published specification, not
    things we measured, so they carry no error to model.
    """
    data = vehicle.to_dict()
    for field in ("a_lat_max", "a_accel_max", "a_brake_max", "a_coast"):
        data[field] = data[field] * float(scale)
    data["throttle_v_coeffs"] = [c * float(map_scale) for c in data["throttle_v_coeffs"]]
    return Vehicle(**data)
```

In `Perturbation`, after `vehicle_scale: float = 1.0`:

```python
    map_scale: float = 1.0
    #: Planner sees the pose from this many control steps ago. The real
    #: odometry arrives slower than the 40 Hz loop runs on the LiDAR.
    pose_delay_steps: int = 0
```

In `simulate`'s signature: change `speed_lag: float = 2.0,` to `speed_lag: Optional[float] = None,` and add `truth_vehicle: Optional[Vehicle] = None,` after `perturbation`.

Replace the `truth = (...)` assignment with:

```python
    # The planner keeps `vehicle`; the dynamics use `truth`. When they differ,
    # the car is not the car the planner thinks it is - which is the point.
    base = truth_vehicle if truth_vehicle is not None else vehicle
    truth = (scaled_vehicle(base, noise.vehicle_scale, noise.map_scale)
             if (noise.vehicle_scale != 1.0 or noise.map_scale != 1.0) else base)
    lag = truth.speed_lag if speed_lag is None else float(speed_lag)
    seen = collections.deque(maxlen=max(int(noise.pose_delay_steps), 0) + 1)
```

Replace `plan = planner.step(ranges, angles, (x, y), yaw, speed, dt)` with:

```python
        seen.append((x, y, yaw, speed))
        px, py, pyaw, pspeed = seen[0]           # the oldest kept = pose_delay_steps ago
        plan = planner.step(ranges, angles, (px, py), pyaw, pspeed, dt)
```

Replace `accel = longitudinal_accel(truth, throttle_cmd, speed, speed_lag)` with `accel = longitudinal_accel(truth, throttle_cmd, speed, lag)`. Add `import collections` at the top.

- [ ] **Step 4: Run tests**

Run: `<venv>/bin/python -m pytest -q`
Expected: all pass (existing seed/perturbation tests unchanged: with defaults, behaviour is identical because `Vehicle.speed_lag` defaults to the old `speed_lag=2.0`).

- [ ] **Step 5: Commit**

```bash
git add race_ws/src/team_driver/team_driver/tools/surrogate.py race_ws/src/team_driver/tests/test_surrogate.py
git commit -m "feat(surrogate): truth vehicle separate from belief, throttle-map error, pose latency"
```

---

### Task 4: Calibration gate — the surrogate must reproduce the real run

Nothing is searched until the surrogate, given what the planner **believed** during the real run and the car we **measured**, reproduces that run. Otherwise the next search optimises a model that is wrong in an unknown direction.

**Files:**
- Create: `lab/calibrate.py`
- Output: `results/vehicle_belief_20260929.json`

**Interfaces:**
- Consumes: `simulate(..., truth_vehicle=)` (Task 3), `results/vehicle_measured.json` (Task 2), the Task 1 referee result JSON.

- [ ] **Step 1: Save the belief the real run used**

```bash
git show 754f8dd:race_ws/src/team_driver/config/vehicle.json > results/vehicle_belief_20260929.json
```

- [ ] **Step 2: Write `lab/calibrate.py`**

```python
#!/usr/bin/env python3
"""Does the surrogate reproduce a REAL run? The gate before any new search.

    python3 lab/calibrate.py --result results/<run>.json \
        --params lab/shortlist/cem/driver_params_480.yaml \
        --belief results/vehicle_belief_20260929.json --truth results/vehicle_measured.json

The planner is given what it believed during the real run; the simulated car
is the one we measured. If the surrogate then lands within 3% of the real
mean timed lap, it models the car well enough to search against. If not,
something the surrogate does not model is costing time, and searching would
optimise the wrong thing.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws", "src", "team_driver"))

from team_driver.core.planner import Planner, PlannerConfig          # noqa: E402
from team_driver.core.profile import speed_profile                   # noqa: E402
from team_driver.core.track import Centreline, OccupancyGrid         # noqa: E402
from team_driver.core.vehicle import Vehicle                         # noqa: E402
from team_driver.tools.surrogate import SurrogateSensor, simulate    # noqa: E402

TOLERANCE = 0.03


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--belief", required=True)
    p.add_argument("--truth", required=True)
    a = p.parse_args()

    laps = json.load(open(a.result))["laps"]
    real = float(np.mean([lap["raw_time"] for lap in laps]))
    cfg = yaml.safe_load(open(a.params))["driver"]["ros__parameters"]
    line_path = cfg["line"]["raceline_csv"].replace("/hackathon", REPO) \
        or os.path.join(REPO, "maps", "icra26_compete_raceline.csv")

    grid = OccupancyGrid.load(os.path.join(REPO, "maps", "icra26_compete.pgm"),
                              os.path.join(REPO, "maps", "icra26_compete.yaml"))
    sensor = SurrogateSensor(grid, max_range=10.0, beam_count=181)
    belief, truth = Vehicle.load(a.belief), Vehicle.load(a.truth)
    line = Centreline.load(line_path)
    profile = speed_profile(line.curvature, line.s, belief,
                            grip_fraction=cfg["line"]["grip_fraction"],
                            v_max=cfg["line"]["v_max"], total_length=line.length)
    pc = {k: cfg["pursuit"][k] for k in ("tracker", "tau_preview", "min_preview",
                                         "k_cross", "k_heading", "curvature_cap")}
    pc.update({k: cfg["speed"][k] for k in ("kp", "ki", "lift_band", "brake_gain")})
    yaw0 = float(np.arctan2(line.tangent[0][1], line.tangent[0][0]))
    start = (line.xy[0][0], line.xy[0][1], yaw0)

    def elapsed(n_laps):
        planner = Planner(belief, PlannerConfig(**pc))
        planner.load_path(line, profile)
        r = simulate(planner, belief, grid, start=start, distance_m=line.length * n_laps,
                     max_time_s=40.0 * n_laps, sensor=sensor, truth_vehicle=truth)
        assert r.status == "COMPLETE", f"surrogate run {r.status} with {r.collisions} collisions"
        return r.elapsed_s

    # Flying laps only, like the referee's timed laps: 12 minus the first 2.
    surrogate = (elapsed(12) - elapsed(2)) / 10.0
    error = (surrogate - real) / real
    print(f"real mean timed lap {real:.3f} s   surrogate {surrogate:.3f} s   "
          f"error {100*error:+.1f}%   (gate +/-{100*TOLERANCE:.0f}%)")
    print("PASS" if abs(error) <= TOLERANCE else "FAIL - do not search against this model")
    return 0 if abs(error) <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the gate**

```bash
python3 lab/calibrate.py --result results/<Task 1 result file>.json --params lab/shortlist/cem/driver_params_480.yaml --belief results/vehicle_belief_20260929.json --truth results/vehicle_measured.json
```
Expected: `PASS`. Also run it with `--truth results/vehicle_belief_20260929.json`: expected `FAIL` at about −7% (the gap we started from — this proves the gate discriminates).

**If the gate FAILs with the measured truth:** stop. Re-run Task 3's pose-latency perturbation by hand in `calibrate.py` (`perturbation=Perturbation(pose_delay_steps=k)` for k = 1, 2, 3) and read Task 1's per-stretch table for where the remaining time goes. Report both to the human before continuing — the next step depends on which stretch is losing time.

- [ ] **Step 4: Commit**

```bash
git add lab/calibrate.py results/vehicle_belief_20260929.json
git commit -m "feat(lab): calibration gate - surrogate must reproduce the real run within 3%"
```

---

### Task 5: Re-search against the measured car

**Files:**
- Modify: `lab/montecarlo.py` (perturbation constants and `run_rollouts`)
- Output: `lab/shortlist/cem2/` (`cem.json`, `best.yaml`)

**Files (also):**
- Modify: `lab/cem.py` (a `--vehicle` option)

**Interfaces:**
- Consumes: `Perturbation.map_scale`, `Perturbation.pose_delay_steps` (Task 3); `POSE_STEPS` (Task 1); `results/vehicle_measured.json` (Task 2).

- [ ] **Step 1: Let the search take a vehicle file, so the shipped one stays untouched**

In `lab/cem.py`'s `main()`, add to the parser:

```python
    parser.add_argument("--vehicle", default=os.path.join(
        MC.REPO, "race_ws", "src", "team_driver", "config", "vehicle.json"),
        help="vehicle.json the planner believes and the surrogate drives")
```

and replace the `vehicle = Vehicle.load(os.path.join(MC.REPO, ...))` statement with:

```python
    vehicle = Vehicle.load(args.vehicle)
```

The measured car is then both belief and truth for the search; its remaining uncertainty is the perturbations in Step 2.

- [ ] **Step 2: Model the remaining uncertainty** — in `lab/montecarlo.py`, next to `MODEL_SCALE_SD`:

```python
#: Residual throttle-map error after re-identification from a racing run.
#: Belongs to the CAR, so it is shared by the three judged runs.
MAP_SCALE_SD = 0.06

#: The oldest the planner's pose can be, in 25 ms control steps: the
#: odometry period measured in Task 1, rounded. 55 ms -> 2.
POSE_DELAY_MAX = 2   # replace 2 with POSE_STEPS from Task 1 Step 5
```

In `run_rollouts`, add to the `Perturbation(...)` call:

```python
                map_scale=float(np.exp(car.normal(0.0, MAP_SCALE_SD))),
                pose_delay_steps=int(rng.randint(0, POSE_DELAY_MAX + 1)),
```

- [ ] **Step 3: Smoke test, then the full search**

```bash
python3 lab/cem.py --vehicle results/vehicle_measured.json --generations 2 --population 12 --elite 4 --cars 2 --validate 6 --final-top 2 --final-cars 2 --out /tmp/cem2_smoke
python3 -u lab/cem.py --vehicle results/vehicle_measured.json --out lab/shortlist/cem2 | tee lab/shortlist/cem2.log
```
Expected: smoke test completes in ~1–2 min; full search ~25 min, ending with a final table in which SHIPPED and #480-like configs appear, and `best.yaml` written.

- [ ] **Step 4: Commit**

```bash
git add lab/montecarlo.py lab/cem.py lab/shortlist/cem2/ lab/shortlist/cem2.log
git commit -m "feat(lab): search against the re-identified car with map error and pose latency"
```

---

### Task 6: One real run of the new winner

**Files:**
- Output: `lab/shortlist/cem2/driver_params_best.yaml`, `lab/shortlist/cem2/raceline_best.csv`, `results/telemetry_cem2_best.csv`

- [ ] **Step 1: Build the complete params file** — the shipped yaml with the winner's values merged in, pointing at its own line:

```bash
python3 - <<'PY'
import yaml
base = yaml.safe_load(open("race_ws/src/team_driver/config/driver_params.yaml"))
best = yaml.safe_load(open("lab/shortlist/cem2/best.yaml"))
params = base["driver"]["ros__parameters"]
for group, values in best["driver"]["ros__parameters"].items():
    params[group].update(values)
params["line"]["raceline_csv"] = "/hackathon/lab/shortlist/cem2/raceline_best.csv"
params["vehicle_config"] = "/hackathon/results/vehicle_measured.json"
open("lab/shortlist/cem2/driver_params_best.yaml", "w").write(yaml.safe_dump(base, sort_keys=False))
PY
```

`vehicle_config` matters: the searched config assumes the measured car's feed-forward.

- [ ] **Step 2: Build its racing line**

```bash
python3 -c "import json; print(json.load(open('lab/shortlist/cem2/cem.json'))['final'][0]['margin'])"
```
```bash
cd race_ws/src/team_driver && python3 -m team_driver.tools.build_raceline --map ../../../maps/icra26_compete --centreline ../../../maps/icra26_compete_centerline.csv --out ../../../lab/shortlist/cem2/raceline_best.csv --margin <the printed margin>
```
Expected: `every sample on a free cell: True`.

- [ ] **Step 3: Preflight, recorder, one run**

```bash
docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/lab/preflight.py"
```
Expected `PASS` (on FAIL: Task 0, then again).

```bash
docker exec -d roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/lab/record_run.py --seconds 260 --out /hackathon/results/telemetry_cem2_best.csv"
```
```bash
./scripts/evaluate.sh --team cem2 --runs 1 --no-simulator --driver-params /hackathon/lab/shortlist/cem2/driver_params_best.yaml
```
```bash
python3 lab/analyse_run.py results/telemetry_cem2_best.csv --line lab/shortlist/cem2/raceline_best.csv --grip <best grip> --v-max <best v_max>
```
(grip and v_max: the `line:` values in `lab/shortlist/cem2/best.yaml`.)

- [ ] **Step 4: Accept or reject**

Accept if all hold: `COMPLETE`; collisions ≤ 1; real-time factor 0.95–1.05; `analyse_run.py` command following ≥ 95%; mean timed lap **< 12.44 s** (#480's). Otherwise #480 (with the OLD vehicle.json it was run on) is the candidate for Task 7.

- [ ] **Step 5: Commit**

```bash
git add lab/shortlist/cem2/driver_params_best.yaml lab/shortlist/cem2/raceline_best.csv results/telemetry_cem2_best.csv results/cem2__*.json
git commit -m "test(real): cem2 winner, one 10-lap run - <ACCEPTED|REJECTED>, <best> s best, <mean> s mean"
```

---

### Task 7: Ship it

**Files:**
- Modify: `race_ws/src/team_driver/config/driver_params.yaml`, `race_ws/src/team_driver/config/vehicle.json`, `maps/icra26_compete_raceline.csv`, `docs/HANDOVER.md`

- [ ] **Step 1: List exactly which shipped values change**

```bash
python3 - <<'PY'
import yaml
def flat(d, pre=""):
    for k, v in d.items():
        yield from (flat(v, pre + k + ".") if isinstance(v, dict) else [(pre + k, v)])
old = dict(flat(yaml.safe_load(open("race_ws/src/team_driver/config/driver_params.yaml"))))
new = dict(flat(yaml.safe_load(open("lab/shortlist/cem2/driver_params_best.yaml"))))
for k in old:
    if old[k] != new[k] and not k.endswith(("raceline_csv", "vehicle_config")):
        print(f"{k}: {old[k]} -> {new[k]}")
PY
```

- [ ] **Step 2: Edit each listed key in place** in `config/driver_params.yaml` — `yaml.safe_dump` would strip the file's comments, which are its documentation. Keep each key's comment and append one line giving the measured reason, e.g. `# 2026-09-29: 0.70 -> 0.81, CEM on the re-identified car; real run 12.1 s clean.` Leave `line.raceline_csv: ""` and `vehicle_config: ""`.

- [ ] **Step 3: Ship the car and the line**

If Task 6 accepted the cem2 winner, ship the measured car it was tuned against:

```bash
cp results/vehicle_measured.json race_ws/src/team_driver/config/vehicle.json
```

If Task 6 fell back to #480, **leave `vehicle.json` unchanged** — #480's gains were tuned around the old feed-forward and were validated on it — and use Step 1's diff against `lab/shortlist/cem/driver_params_480.yaml` with margin **0.300**.
```bash
cd race_ws/src/team_driver && python3 -m team_driver.tools.build_raceline --map ../../../maps/icra26_compete --centreline ../../../maps/icra26_compete_centerline.csv --out ../../../maps/icra26_compete_raceline.csv --margin <accepted margin, 3 decimals>
```

- [ ] **Step 4:** Full test suite — all pass. `./scripts/verify_judging_env.sh` — intact.

- [ ] **Step 5:** Preflight, then one real run with the **shipped** files and no `--driver-params` (the judges' path): `./scripts/evaluate.sh --team final --runs 1 --no-simulator`. Expected: `COMPLETE`, lap times matching Task 6 within ±0.1 s mean.

- [ ] **Step 6:** `docs/HANDOVER.md`: add findings F1–F6, the tooling (`preflight`, `record_run`, `analyse_run`, `calibrate`, `cem`), the bridge self-kill fix, the DQ-cliff and correlated-runs scoring fixes, and the new real numbers. Replace §4's table with the re-identified values.

- [ ] **Step 7: Commit**

```bash
git add race_ws/src/team_driver/config/ maps/icra26_compete_raceline.csv docs/HANDOVER.md results/final__*.json
git commit -m "perf: ship the re-identified car and the re-searched config - <best> s best, <mean> s mean"
```
