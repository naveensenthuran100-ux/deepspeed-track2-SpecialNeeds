# Track 2 Racing Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a defensible classical autonomous racing stack for the AutoDRIVE ICRA 2026 compete circuit that completes 10 laps reliably and fast, with every artefact reproducible from committed code.

**Architecture:** Three tiers — a thin ROS adapter, a pure-Python core with zero ROS imports, and offline tools that emit committed artefacts. A numpy surrogate simulator makes configuration search cheap despite a six-minute real-simulator feedback loop.

**Tech Stack:** Python 3.10 (judging) / 3.13 (local venv), numpy, scipy, ROS 2 Humble, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-track2-racing-stack-design.md`

## Global Constraints

- Control output is normalised torque and steering, both clipped to `[-1, 1]`. Steering 1.0 == 0.5236 rad.
- Entry point `ros2 run team_driver driver` must keep working.
- Never publish `/autodrive/reset_command`.
- Never modify: `race_ws/src/roboracer_referee`, `external/autodrive_devkit`, `docker`, `maps/tracks.yaml`, `scripts`, `.github/workflows`, `install`, `.dockerignore`.
- Zero declared pip/apt dependencies. numpy, scipy, PyYAML, opencv, PIL are already in the image.
- **numpy 1.21 compatible only.** No `np.trapezoid`, no `copy=False` keyword semantics from numpy 2, no `np.astype` method. Use `np.trapz`, explicit `.astype()`.
- Core modules (`team_driver/core/`) import no ROS packages, ever.
- Control loop budget 25 ms. Vectorised numpy only — no per-beam Python loops.
- Local test command: `/private/tmp/claude-501/-Users-naveen-Deepspeed/7c8ade1f-74ec-4a41-bd57-c9b41876cb76/scratchpad/venv/bin/python -m pytest`
- Vehicle constants (published spec): wheelbase 0.324 m, length 0.500 m, width 0.270 m, mass 3.906 kg, wheel radius 0.059 m, max steer 0.5236 rad, steer rate 3.2 rad/s.
- LiDAR: 1080 beams, angle_min -2.356 rad, angle_max +2.356 rad, range 0.06-10.0 m, 40 Hz, mounted 0.2733 m ahead of the rear axle.
- Map: 220x476 cells at 0.05 m, origin `(-7.960, -13.230)`, PGM values 0 = boundary, 254 = track, 205 = outside track.

## Review Focus

Input classes the spec implies but which no single task's happy path exercises. Each has its test pinned to the owning task.

1. **All-zero or all-inf LiDAR scan** (bridge hiccup, car inside geometry) — planner must return finite commands, not NaN. Owned by Task 4.
2. **Odometry absent or stale at startup** — the first scans arrive before the first odom; the car must still steer sensibly rather than divide by a zero speed. Owned by Task 9.
3. **Pose off the occupancy grid** (spawn outside the traced area, or a bad transform) — grid lookups must not raise IndexError or silently wrap. Owned by Task 6.
4. **Speed controller at zero and negative target speed** — feed-forward inversion must not produce a throttle above 1 or an unbounded integral when `v_target = 0`. Owned by Task 2.
5. **Degenerate centreline geometry** (duplicate consecutive points, a closed loop whose last point equals its first) — curvature and arc-length must not divide by zero. Owned by Task 6.

---

### Task 1: Core package scaffolding and vehicle model

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/__init__.py`
- Create: `race_ws/src/team_driver/team_driver/core/vehicle.py`
- Create: `race_ws/src/team_driver/maps/vehicle.json`
- Create: `race_ws/src/team_driver/tests/__init__.py`
- Create: `race_ws/src/team_driver/tests/test_vehicle.py`
- Create: `race_ws/src/team_driver/pytest.ini`

**Interfaces:**
- Produces: `Vehicle` dataclass with fields `wheelbase, length, width, mass, max_steer_rad, max_steer_rate, throttle_v_coeffs, a_lat_max, a_accel_max, a_brake_max, v_max, measured`. Classmethods `Vehicle.default()` and `Vehicle.load(path)`. Methods `throttle_for_speed(v) -> float`, `speed_for_throttle(u) -> float`, `steer_to_norm(rad) -> float`, `norm_to_steer(n) -> float`.

- [ ] **Step 1: Write the failing test**

```python
def test_throttle_speed_roundtrip():
    v = Vehicle.default()
    for speed in [0.5, 1.0, 3.0, 5.0]:
        u = v.throttle_for_speed(speed)
        assert 0.0 <= u <= 1.0
        assert abs(v.speed_for_throttle(u) - speed) < 1e-6

def test_zero_target_gives_zero_throttle():
    assert Vehicle.default().throttle_for_speed(0.0) == 0.0

def test_default_is_marked_unmeasured():
    assert Vehicle.default().measured is False
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_vehicle.py -v`, expect ImportError.
- [ ] **Step 3: Implement `vehicle.py`** with a monotonic polynomial actuator map `v = c1*u + c2*u^2` inverted analytically, clipped to `[0, 1]`; `measured=False` on defaults.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(core): vehicle model and actuator map`

---

### Task 2: Speed controller

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/speed_controller.py`
- Create: `race_ws/src/team_driver/tests/test_speed_controller.py`

**Interfaces:**
- Consumes: `Vehicle` from Task 1.
- Produces: `SpeedController(vehicle, kp, ki, integral_limit, lift_band)` with `step(v_target, v_measured, dt) -> float` and `reset()`.

- [ ] **Step 1: Write the failing test**

```python
def test_holds_setpoint_from_rest():
    c = SpeedController(Vehicle.default())
    v = 0.0
    for _ in range(400):
        u = c.step(3.0, v, 0.025)
        v += (Vehicle.default().speed_for_throttle(u) - v) * 0.12
    assert abs(v - 3.0) < 0.15

def test_integral_is_clamped():
    c = SpeedController(Vehicle.default(), integral_limit=0.2)
    for _ in range(1000):
        c.step(5.0, 0.0, 0.025)
    assert abs(c.integral) <= 0.2 + 1e-9

def test_slightly_fast_lifts_rather_than_brakes():
    c = SpeedController(Vehicle.default())
    u = c.step(3.0, 3.2, 0.025)
    assert u >= 0.0

def test_badly_fast_brakes():
    c = SpeedController(Vehicle.default())
    u = c.step(1.0, 6.0, 0.025)
    assert u < 0.0

def test_zero_and_negative_target_are_safe():   # Review Focus 4
    c = SpeedController(Vehicle.default())
    for target in [0.0, -1.0]:
        for _ in range(200):
            u = c.step(target, 0.0, 0.025)
            assert -1.0 <= u <= 1.0
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** feed-forward `vehicle.throttle_for_speed(v_target)` plus PI on error, integral clamped, `lift_band` deciding coast vs brake, output clipped.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(core): feed-forward plus PI speed controller`

---

### Task 3: LiDAR processing

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/lidar.py`
- Create: `race_ws/src/team_driver/tests/test_lidar.py`

**Interfaces:**
- Produces: `clean(ranges, max_range) -> np.ndarray`, `beam_angles(angle_min, increment, n) -> np.ndarray`, `safety_bubble(ranges, angles, radius_m) -> np.ndarray`, `disparity_extend(ranges, angles, car_half_width, threshold) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
def test_clean_replaces_nan_and_inf():
    out = clean(np.array([np.nan, np.inf, -np.inf, 5.0]), 10.0)
    assert np.all(np.isfinite(out))
    assert out[1] == 10.0

def test_disparity_extension_closes_a_duct_gap():
    # a near wall, a one-beam gap showing far, then near again
    r = np.full(101, 1.0); r[50] = 9.0
    angles = np.linspace(-1.0, 1.0, 101)
    out = disparity_extend(r, angles, car_half_width=0.2, threshold=0.3)
    assert out[50] < 9.0     # the apparent opening is pulled in

def test_safety_bubble_blanks_around_closest_return():
    r = np.full(101, 5.0); r[20] = 0.3
    angles = np.linspace(-1.0, 1.0, 101)
    out = safety_bubble(r, angles, radius_m=0.5)
    assert out[20] == 0.0 and out[21] == 0.0
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement**, fully vectorised, numpy 1.21-safe.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(core): lidar cleaning, safety bubble, disparity extension`

---

### Task 4: Follow-the-gap

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/gap.py`
- Create: `race_ws/src/team_driver/tests/test_gap.py`

**Interfaces:**
- Consumes: `lidar` from Task 3.
- Produces: `GapFollower(config)` with `step(ranges, angles) -> GapResult(steer_rad, v_target, gap_centre_index)`.

- [ ] **Step 1: Write the failing test**

```python
def test_steers_towards_the_open_side():
    r = np.full(201, 1.0); r[150:180] = 8.0
    angles = np.linspace(-1.5, 1.5, 201)
    out = GapFollower().step(r, angles)
    assert out.steer_rad > 0.0      # positive angles are to the left

def test_picks_widest_not_deepest():
    r = np.full(201, 1.0)
    r[100] = 9.5                    # one deep beam
    r[150:180] = 8.0                # a genuinely wide gap
    angles = np.linspace(-1.5, 1.5, 201)
    out = GapFollower().step(r, angles)
    assert angles[150] <= out.steer_rad <= angles[179]

def test_degenerate_scan_returns_finite(): # Review Focus 1
    angles = np.linspace(-1.5, 1.5, 201)
    for r in [np.zeros(201), np.full(201, np.inf), np.full(201, np.nan)]:
        out = GapFollower().step(r, angles)
        assert np.isfinite(out.steer_rad) and np.isfinite(out.v_target)
        assert out.v_target >= 0.0
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** widest-run selection over a max-range mask, target the run's midpoint, speed scaled by steering magnitude and forward clearance.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(core): follow-the-gap planner`

---

### Task 5: ROS adapter

**Files:**
- Modify: `race_ws/src/team_driver/team_driver/driver.py`
- Modify: `race_ws/src/team_driver/config/driver_params.yaml`
- Modify: `race_ws/src/team_driver/setup.py` (package list for `core`, `tools`)

**Interfaces:**
- Consumes: `GapFollower`, `SpeedController`, `Vehicle`.
- Produces: a node that publishes throttle and steering every scan and never raises out of a callback.

- [ ] **Step 1: Rewrite `driver.py`** as a thin adapter: `devkit_qos()` kept verbatim, odom/scan callbacks, `planner.step(...)`, publish, markers, a wrapped try/except that falls back to the previous steering at reduced throttle and logs once.
- [ ] **Step 2: Declare every tunable in `driver_params.yaml`** so sweeps need no rebuild.
- [ ] **Step 3: Update `setup.py`** to `packages=['team_driver', 'team_driver.core', 'team_driver.tools']`.
- [ ] **Step 4: Verify** `python -c "import ast; ast.parse(open('driver.py').read())"` and that `core` imports no ROS.
- [ ] **Step 5: Commit** — `feat(driver): thin ROS adapter over the core`

---

### Task 6: Track geometry

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/track.py`
- Create: `race_ws/src/team_driver/tests/test_track.py`

**Interfaces:**
- Produces: `OccupancyGrid.load(pgm, yaml)` with `world_to_grid(xy)`, `grid_to_world(ij)`, `is_free(xy)`, `is_free_many(xy)`, `free_radius(xy)`. `Centreline.load(csv)` / `Centreline(points)` with `.xy`, `.s`, `.curvature`, `.length`, `nearest(xy) -> (index, distance)`, `point_at_s(s)`, `width_at(index, grid)`.

- [ ] **Step 1: Write the failing test**

```python
def test_world_grid_roundtrip():
    g = OccupancyGrid.load(PGM, YAML)
    for xy in [(0.0, 0.0), (0.8, 3.16), (-3.0, -5.0)]:
        assert np.allclose(g.grid_to_world(g.world_to_grid(xy)), xy, atol=0.05)

def test_offgrid_query_is_safe():            # Review Focus 3
    g = OccupancyGrid.load(PGM, YAML)
    assert g.is_free((1e6, 1e6)) is False    # no IndexError, no wraparound

def test_curvature_of_a_circle():
    t = np.linspace(0, 2*np.pi, 400, endpoint=False)
    c = Centreline(np.stack([2.0*np.cos(t), 2.0*np.sin(t)], axis=1))
    assert np.allclose(np.abs(c.curvature), 0.5, atol=0.02)   # kappa = 1/R

def test_duplicate_points_do_not_divide_by_zero():   # Review Focus 5
    pts = np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.],[1.,1.]])
    c = Centreline(pts)
    assert np.all(np.isfinite(c.curvature)) and np.all(np.isfinite(c.s))

def test_shipped_centreline_lap_length():
    c = Centreline.load(CENTRELINE_CSV)
    assert abs(c.length - 54.43) < 0.5
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Curvature by finite differences on arc length with duplicate points collapsed first. `is_free` bounds-checked explicitly.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(core): occupancy grid and centreline geometry`

---

### Task 7: Pure pursuit

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/pursuit.py`
- Create: `race_ws/src/team_driver/tests/test_pursuit.py`

**Interfaces:**
- Consumes: `Centreline`, `Vehicle`.
- Produces: `PurePursuit(vehicle, k_lookahead, min_lookahead, max_lookahead)` with `step(pose_xy, yaw, speed, path) -> PursuitResult(steer_rad, target_xy, lookahead)`.

- [ ] **Step 1: Write the failing test**

```python
def test_straight_ahead_gives_zero_steering():
    path = Centreline(np.stack([np.linspace(0, 20, 200), np.zeros(200)], axis=1))
    out = PurePursuit(Vehicle.default()).step((0.0, 0.0), 0.0, 3.0, path)
    assert abs(out.steer_rad) < 1e-6

def test_known_arc_matches_geometry():
    # target at lookahead L, lateral offset y -> delta = atan(2*L_wb*y / L^2)
    ...
def test_lookahead_scales_with_speed():
    p = PurePursuit(Vehicle.default())
    assert p.step(P, 0.0, 6.0, path).lookahead > p.step(P, 0.0, 1.0, path).lookahead
```

- [ ] **Step 2-5:** fail, implement, pass, commit — `feat(core): pure pursuit tracker`

---

### Task 8: Speed profile

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/profile.py`
- Create: `race_ws/src/team_driver/tests/test_profile.py`

**Interfaces:**
- Consumes: `Vehicle`, `Centreline`.
- Produces: `speed_profile(curvature, s, vehicle, grip_fraction) -> np.ndarray` — cornering cap `sqrt(a_lat/|k|)`, then a wrapped forward pass under `a_accel_max` and backward pass under `a_brake_max`.

- [ ] **Step 1: Write the failing test**

```python
def test_constant_radius_gives_constant_speed():
    k = np.full(400, 0.5); s = np.linspace(0, 2*np.pi*2.0, 400)
    v = speed_profile(k, s, Vehicle.default(), 1.0)
    assert np.allclose(v, v[0], rtol=0.02)

def test_brakes_before_the_corner_not_in_it():
    k = np.zeros(400); k[200:] = 1.0
    s = np.linspace(0, 80, 400)
    v = speed_profile(k, s, Vehicle.default(), 1.0)
    assert v[199] < v[100]          # already slowing before the corner starts

def test_respects_deceleration_limit():
    ...  # consecutive samples never exceed a_brake_max
```

- [ ] **Step 2-5:** fail, implement, pass, commit — `feat(core): curvature speed profile with forward/backward passes`

---

### Task 9: Planner composition and mode switching

**Files:**
- Create: `race_ws/src/team_driver/team_driver/core/planner.py`
- Create: `race_ws/src/team_driver/tests/test_planner.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Planner(vehicle, config, path=None, profile=None)` with `step(ranges, angles, pose, yaw, speed, dt) -> PlanResult(steer_norm, throttle, mode)`.

- [ ] **Step 1: Write the failing test**

```python
def test_falls_back_to_gap_without_pose():   # Review Focus 2
    p = Planner(Vehicle.default(), PlannerConfig())
    out = p.step(np.full(201, 3.0), ANGLES, pose=None, yaw=None, speed=0.0, dt=0.025)
    assert out.mode == "gap"
    assert -1.0 <= out.steer_norm <= 1.0 and -1.0 <= out.throttle <= 1.0

def test_uses_line_mode_when_path_and_pose_present():
    ...
def test_blocked_path_switches_to_gap():
    ...
def test_mode_switch_is_hysteretic():
    ...
def test_outputs_always_clipped_and_finite():
    ...
```

- [ ] **Step 2-5:** fail, implement, pass, commit — `feat(core): planner composition with hysteretic gap fallback`

---

### Task 10: Surrogate simulator

**Files:**
- Create: `race_ws/src/team_driver/team_driver/tools/surrogate.py`
- Create: `race_ws/src/team_driver/tests/test_surrogate.py`

**Interfaces:**
- Produces: `simulate(planner, vehicle, grid, start_pose, laps, dt) -> SimResult(lap_times, collisions, trace, status)`.

- [ ] **Step 1: Write the failing test**

```python
def test_straight_corridor_is_collision_free():
    ...
def test_reports_contact_when_driven_into_a_wall():
    ...
def test_runs_a_lap_within_time_budget():
    # the throughput claim the whole design rests on
    t0 = time.perf_counter(); simulate(...); assert time.perf_counter() - t0 < 1.0
```

- [ ] **Step 2-5:** fail, implement, pass, commit — `feat(tools): numpy surrogate simulator`

---

### Task 11: Racing line optimiser

**Files:**
- Create: `race_ws/src/team_driver/team_driver/tools/line_optimise.py`
- Create: `race_ws/src/team_driver/tests/test_line_optimise.py`

**Interfaces:**
- Produces: `optimise(centreline, grid, margin_m, iterations) -> np.ndarray` — minimum-curvature lateral offsets, bounded by measured track width minus margin.

- [ ] **Step 1: Write the failing test**

```python
def test_line_stays_inside_the_track():
    ...
def test_reduces_total_squared_curvature_vs_centreline():
    ...
def test_respects_the_margin_bound():
    ...
```

- [ ] **Step 2-5:** fail, implement, pass, commit — `feat(tools): minimum-curvature racing line`

---

### Task 12: Sweep harness

**Files:**
- Create: `lab/sweep.py`
- Create: `lab/report.py`
- Create: `lab/README.md`

- [ ] **Step 1:** `sweep.py` — load a config grid, evaluate each against the surrogate, rank by `E[10-lap total]`, emit a shortlist as `driver_params.yaml` files ready for `evaluate.sh --driver-params`.
- [ ] **Step 2:** `report.py` — aggregate `results/*.json` into per-config mean/sigma best lap, mean/sigma total, **completion rate**, mean collisions; and the surrogate-vs-sim correlation table.
- [ ] **Step 3: Commit** — `feat(lab): surrogate sweep and result aggregation`

---

### Task 13: System identification and mapping tools

**Files:**
- Create: `race_ws/src/team_driver/team_driver/tools/sysid.py`
- Create: `race_ws/src/team_driver/team_driver/tools/sysid_drive.py`
- Create: `race_ws/src/team_driver/team_driver/tools/grid_build.py`

- [ ] **Step 1:** `sysid_drive.py` — a ROS node that runs the throttle sweep, the acceleration/braking steps and the constant-radius grip test, logging to CSV. This is the first thing we run in the first sim session.
- [ ] **Step 2:** `sysid.py` — fits `vehicle.json` from that CSV and marks `measured: true`.
- [ ] **Step 3:** `grid_build.py` — ray-carves an occupancy grid from recorded pose + LiDAR.
- [ ] **Step 4: Commit** — `feat(tools): system identification and grid building`

---

### Task 14: Submission scaffolding

**Files:**
- Create: `SUBMISSION.md`
- Verify: `requirements.txt` and `apt-packages.txt` stay empty of real entries.

- [ ] **Step 1:** Write `SUBMISSION.md` from the chapter 7 template with everything known so far filled in and the rest clearly marked as pending measurement.
- [ ] **Step 2:** Run `./scripts/verify_judging_env.sh` and confirm the environment is intact.
- [ ] **Step 3: Commit** — `docs: submission scaffolding`
