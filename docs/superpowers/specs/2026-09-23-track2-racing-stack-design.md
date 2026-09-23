# Track 2 Racing Stack — Design

NTU DeepSpeed Recruitment Hackathon 26/27 · Track 2 (RoboRacer Sim Racing on AutoDRIVE)

Date: 2026-09-23 · Submission deadline: 2026-10-18 23:59 SGT

---

## 1. Problem

Drive the RoboRacer digital twin around the ICRA 2026 compete circuit for 10 timed
laps, as fast as possible, without touching the boundary.

Score, out of 100, relative to the best team in the track:

```
score = 50 x (L* / L_ours)  +  50 x (T* / T_ours)
```

where `L` is the fastest single lap and `T` the 10-lap total, both **after** the
+10 s per-collision penalty. More than 10 collisions in a run is a
disqualification and scores zero for both halves. The judges run each entry
three times and take the best attempt.

### Marginal value, around leader pace (L ~ 20 s, T ~ 220 s)

| Change | Points |
|---|---|
| 1 s off every lap | ~ +4.8 |
| One collision | ~ -2.3 (about the same as being 0.5 s/lap slower everywhere) |
| The 11th collision | ~ -100 |

Two consequences drive every decision below:

1. **The cliff dominates until we are reliably clean.** Three collisions a lap
   disqualifies on lap four.
2. **Once clean, collisions are priced correctly at 10 s.** So the single
   optimisation objective is `E[10-lap total including penalties]`, which
   already balances speed against contact. We do not need a separate
   "safety" weight.

Because the judges take the best of three, the target is `E[max of 3 runs]`,
not `E[single run]`. This tolerates more aggression than expected-value alone:
a config that completes 60% of the time delivers a scoring run 94% of the time.
Completion rate is therefore a first-class measured quantity, not an
afterthought.

## 2. Constraints

### Hard (environment)

- Control is **normalised motor torque** in [-1, 1], not a speed request. No
  speed controller exists anywhere between our node and the wheels.
- LiDAR reaches **10 m** on a **54.43 m** lap. Purely reactive driving is
  structurally late on the brakes.
- The boundary is a row of **33 cm air ducts**, not a wall. Beams pass between
  them and report the far side or nothing. Disparity extension is mandatory.
- Track is ~2 m wide typically, **under 1 m at the tightest point**, for a
  0.27 m wide car.
- Control loop budget: **25 ms** (LiDAR at 40 Hz). The bridge runs a strict
  request/response loop, so a slow node slows the whole simulation.
- QoS must be `RELIABLE / KEEP_LAST(1) / VOLATILE` or topics silently never
  connect.
- Entry point must remain `ros2 run team_driver driver`.
- Must not publish `/autodrive/reset_command` (instant DQ).
- Must not modify: `race_ws/src/roboracer_referee`, `external/autodrive_devkit`,
  `docker`, `maps/tracks.yaml`, `scripts`, `.github/workflows`, `install`,
  `.dockerignore`.

### Hard (ours)

- **Development is Apple Silicon only** (M3 Pro, arm64, 18 GB). The Linux
  simulator binary is x86-only, so locally we run the native macOS `.app` on the
  host with the devkit in an arm64 container (`evaluate.sh --no-simulator`).
  Lap times remain comparable because they come from the simulator's own clock
  at a fixed physics timestep, but **we never execute the judged code path
  locally**.
- **No CUDA.** Anything GPU-dependent would ship unvalidated. Excluded by
  decision.
- **Zero declared dependencies.** numpy, scipy, opencv, PyYAML and PIL are
  already in the image, and `requirements.txt` forbids replacing numpy or
  transforms3d. Shipping an empty requirements file removes the largest silent
  failure mode in the competition.
- **numpy 1.21 compatibility.** Our venv has numpy 2.0.2; the judging image has
  Ubuntu 22.04's apt numpy (1.21.x). All core code must work on both. No
  numpy-2-only APIs.
- Feedback loop against the real simulator is **~6 minutes per 12-lap run**,
  real-time-locked, serial. Realistically ~80 runs/day.

### Compensating control

GitHub Actions is the **only** access we have to the judged code path: x86
Linux, real Linux simulator, headless, same referee. Nothing it prints is
scored, but it is where we prove the entry builds and runs on the judges'
architecture. CI is part of the harness, not an afterthought.

## 3. Goals and non-goals

**Goals**

- Ten laps, reliably complete, with a measured completion rate.
- Lap time competitive via a computed racing line and speed profile, not tuning
  by hand.
- Every artefact reproducible from committed code (rule 36).
- Every component explainable at interview by any team member.

**Non-goals**

- Reinforcement learning, imitation learning, end-to-end policies. Excluded:
  the simulator is real-time-locked so sample collection is prohibitive, and we
  cannot validate a GPU policy on the judging hardware.
- Camera use. Headless runs produce no image.
- Own localisation (particle filter). Deferred, not rejected: ground-truth pose
  is explicitly permitted and recommended. Revisit only if the classical stack
  lands early.

## 4. Architecture

Three tiers with hard boundaries. The governing principle: **anything testable
without the simulator is worth a hundred times anything that is not**, because
the simulator costs six minutes and lies about repeatability.

```
tier 3   OFFLINE TOOLS                     tier 2   PURE-PYTHON CORE
  sysid_fit      -> vehicle.json             vehicle    (bicycle model, actuator map)
  grid_build     -> maps/*.pgm               speed_ctl  (feed-forward + PI)
  line_optimise  -> maps/*_raceline.csv      gap        (follow-the-gap + disparity)
  surrogate      -> lap time, collisions     pursuit    (pure pursuit geometry)
       |                                     profile    (curvature -> speed, fwd/bwd)
       | consumes tier 2                     track      (grid, centreline, frames)
       v                                          ^
  lab/sweep.py                                    | consumes
                                             tier 1   ROS ADAPTER
                                               driver.py (subscribe, call, publish)
```

### Tier 1 — ROS adapter (`team_driver/driver.py`)

Thin. Subscribes with `devkit_qos()`, converts messages to plain numpy/floats,
calls the core, publishes two `Float32`s and debug markers. **No algorithm
logic.** Target ~150 lines. Holds the ROS parameter declarations and maps them
onto core config objects.

Rationale: the core cannot be unit-tested if it only exists inside a callback.

### Tier 2 — Pure-Python core (`team_driver/core/`)

Zero ROS imports. Every module is a plain function or small class over numpy
arrays and dataclasses, unit-testable in milliseconds.

| Module | Responsibility | Depends on |
|---|---|---|
| `vehicle.py` | Kinematic bicycle model; actuator map `throttle <-> steady-state speed`; measured accel/brake/grip limits, loaded from `vehicle.json` | - |
| `speed_controller.py` | `(v_target, v_measured, dt) -> throttle`. Feed-forward from the actuator map plus clamped-integral PI; lift-not-brake policy | `vehicle` |
| `lidar.py` | Scan cleaning, angle array, safety bubble, **disparity extension** | - |
| `gap.py` | Follow-the-gap: widest-gap selection, target angle, reactive target speed | `lidar` |
| `track.py` | Occupancy grid load, world<->grid transforms, centreline, arc-length, curvature, track-width sampling, nearest-point queries | - |
| `raceline.py` | Min-curvature line from centreline + width limits (QP over lateral offsets) | `track` |
| `profile.py` | Curvature -> cornering speed from measured grip; forward/backward passes under measured accel/brake limits | `vehicle`, `track` |
| `pursuit.py` | Pure-pursuit geometry; speed-scaled lookahead; steering from arc curvature and wheelbase | `track` |
| `planner.py` | The composition: pick mode (line-following vs gap fallback), produce `(steering, v_target)` | all above |

`planner.py` is the only module that knows about the others; everything below
it is independently comprehensible, which is the property the interview
actually tests.

### Tier 3 — Offline tools (`team_driver/tools/`)

Shipped, because rule 36 requires the code that produced any precomputed
artefact to be submitted.

- `sysid_fit.py` — fits `vehicle.json` from recorded throttle/speed sweeps.
- `grid_build.py` — ray-carves an occupancy grid from recorded pose + LiDAR.
- `line_optimise.py` — emits `maps/icra26_compete_raceline.csv`.
- `surrogate.py` — the offline simulator (below).

### The surrogate

The throughput multiplier. Given the identified bicycle model and the occupancy
grid, simulate a full lap in pure numpy in roughly 10 ms: integrate the model
under the real planner and speed controller, test the car's footprint against
the grid for contact, and return `(lap_time, collisions, trace)`.

This lets us sweep thousands of configurations in seconds and send only the
best handful to the real simulator. It converts a 25-configs-per-day budget
into effectively unlimited coarse search plus a few confirmation runs.

**The surrogate can lie.** A model that disagrees with AutoDRIVE optimises us
into a line the real car cannot hold. Mitigation, and it is not optional:

- Surrogate-vs-simulator lap time correlation is a **measured, reported
  quantity**, refreshed whenever the model changes.
- The surrogate is never trusted outside the speed and curvature range where it
  has been validated.
- Every configuration that ships is confirmed on the real simulator over at
  least three full 10-lap runs before we believe it.

### Artefacts

Committed to `maps/` (ours to write; only `maps/tracks.yaml` is protected):

- `icra26_compete_raceline.csv` — optimised line with per-point target speed.
- `vehicle.json` — identified vehicle parameters.

Each regenerable by one documented command.

### Dev tooling (`lab/`, repo root, not shipped-critical)

Not in any protected directory. Holds `sweep.py` (run a config grid against the
surrogate, then shortlist to the simulator), result aggregation over
`results/*.json`, and the surrogate-vs-sim correlation report.

## 5. Data flow, per scan (40 Hz)

```
LaserScan ---> lidar.clean -> disparity extension
                    |
Odometry -----> pose, speed
                    |
                    v
            planner.step(scan, pose, speed)
                    |
        +-----------+-----------+
        |                       |
  line mode                gap mode (fallback)
  nearest point on         widest gap after
  raceline -> lookahead    disparity extension
  -> pursuit steering      -> steering
  -> profile v_target      -> reactive v_target
        |                       |
        +-----------+-----------+
                    |
              (steering, v_target)
                    |
      speed_controller.step(v_target, v_measured, dt)
                    |
              (steering, throttle)  -> two Float32 publishes
```

**Mode switching.** Line mode is the default. We fall back to gap mode when the
LiDAR says the planned path is blocked within braking distance, or when the
pose is implausible. This is the guide's own recommended answer for a blocked
line and it protects against the DQ cliff. The switch is hysteretic so the car
does not oscillate between modes.

## 6. Error handling

Every failure mode ends in "keep publishing something safe", because a node
that stops publishing leaves the bridge repeating the last command and the car
drives on at the last throttle.

| Failure | Response |
|---|---|
| Scan contains NaN/inf | Cleaned at the boundary (`nan_to_num`, clip to max range) |
| No odometry yet | Gap mode, conservative speed cap |
| Pose implausible (off-grid, jump) | Gap mode until it recovers |
| Planner raises | Caught in the adapter; publish previous steering with reduced throttle; log once |
| Loop over budget | Measured and logged; profile counters exposed in the result |
| Artefact missing at startup | Fall back to centreline, then to gap mode; log loudly |

Never publish `/autodrive/reset_command`. Never raise out of the callback.

## 7. Testing

- **Unit (pytest, milliseconds).** Every tier-2 module, run locally in the venv.
  These are the tests that make TDD viable given a six-minute integration loop.
  Including: controller holds a setpoint; integral clamps; disparity extension
  closes a duct gap; widest-gap selection is stable between near-identical
  directions; curvature from a known circle; speed profile respects both
  limits; pursuit steering matches a known arc; grid transforms round-trip.
- **Property tests.** Steering always within [-1, 1]; throttle always within
  [-1, 1]; planner never returns NaN for any finite scan.
- **Surrogate regression.** A committed reference lap; changes that alter the
  surrogate's lap time by more than a threshold must be deliberate.
- **Integration (simulator, minutes).** Full 10-lap runs, three at a time,
  scored with `evaluate.sh`.
- **x86 validation (CI).** The Judge workflow on every push. Green means the
  entry builds and runs on the judges' architecture. It is a correctness gate,
  not a score.

## 8. Build order

Ordered by lap time gained per hour, which is also dependency order.

1. Core scaffolding, `vehicle.py`, `speed_controller.py` + tests. Everything
   depends on the speed controller.
2. `lidar.py`, `gap.py` + tests, and the ROS adapter. First thing that laps;
   becomes the permanent fallback.
3. `track.py`, `pursuit.py`, `profile.py` + tests. Centreline-following with a
   computed speed profile.
4. Surrogate + `lab/sweep.py`. Throughput unlocked.
5. `raceline.py` — min-curvature optimisation. Where the lap time is.
6. Tune against `E[10-lap total]`, confirm on the simulator, report completion
   rate.
7. Submission hygiene: clean clone, `verify_judging_env.sh`, `results/submitted/`,
   `SUBMISSION.md`, green CI.

Steps 1-5 need no simulator and no team availability. Steps 2, 3, 6 and 7 need
the simulator, and system identification (feeding step 1's `vehicle.json`) needs
a live car.

Until `vehicle.json` is measured it carries documented placeholder values
derived from the published vehicle specification, clearly marked, so the core
is runnable and testable before the first sim session.

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Surrogate disagrees with AutoDRIVE | High | Correlation measured and reported; never extrapolate; confirm on sim before shipping |
| arm64/x86 divergence | Medium | Zero declared deps; conservative numpy subset; CI as the x86 gate |
| numpy 1.21 vs 2.0 API drift | Medium | No numpy-2-only APIs; CI catches the rest |
| Loop exceeds 25 ms on the judging machine | Medium | Budget measured per scan; vectorised numpy only; no per-beam Python loops |
| Placeholder `vehicle.json` never replaced | High | System ID is the first sim session; the file records whether it is measured or assumed |
| Tuned to our macOS sim, different on x86 | Medium | Prefer analytic parameters over searched ones; confirm shortlists on CI |
| Team cannot explain the code | High | Tier boundaries chosen for explainability; every module one page; spec is the interview script |
