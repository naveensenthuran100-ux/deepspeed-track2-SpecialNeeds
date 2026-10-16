# Team SpecialNeeds

## Members

| Name | Matric number | Email |
| --- | --- | --- |
| Naveen | U2522436E | naveense001@e.ntu.edu.sg |
| Sachin | U2522745C | Alag0005@e.ntu.edu.sg |
| Yukesh | U2522453F | yukesh001@e.ntu.edu.sg |

## Approach

A classical racing stack: **plan offline, track online**. Before the race,
`team_driver/tools/line_mintime.py` computes a minimum-lap-time racing line
from the occupancy grid. At start-up the driver turns that line into a speed
profile from the car's measured limits. During the race, about 40 times a
second, it finds the car on the line from ground-truth odometry; a speed
controller sets the throttle and a curvature feed-forward tracker sets the
steering. Follow-the-gap on the LiDAR is the fallback if the pose is lost,
stale (> 0.25 s) or more than 1.2 m off the line.

Why this approach: lap time on this circuit is decided by braking for corners
the 10 m LiDAR cannot see yet — the car brakes from ~10 m/s to ~2 m/s for the
two 0.8 m hairpins. Planning the whole lap against the map does that; a
reactive driver cannot, so it is only the fallback.

The code is in three layers. `driver.py` is a thin ROS 2 adapter with no
racing logic. `team_driver/core/` holds every decision in pure Python with no
ROS imports, so it is unit-tested in milliseconds (228 tests). `team_driver/tools/`
holds the offline generators rule 36 requires.

Components:

- **Racing line** (`tools/line_mintime.py`): a closed periodic spline shifted
  sideways from the centreline at 90 control points. Each candidate is scored
  by the lap time of the same speed profile the car drives, and optimised with
  L-BFGS-B. Penalties enforce wall clearance (half car width + 0.25 m,
  measured on the distance field of *free* cells, so gaps between ducts count
  as wall — rule 11), a curvature cap of 1.3 /m, and a curvature-rate cost.
  53.12 m long. It replaced our first minimum-curvature line
  (`tools/line_optimise.py`) because it was faster on the real simulator:
  11.75 s vs 11.96 s mean lap.
- **Speed profile** (`core/profile.py`): corner limit v = √(a_lat/κ), then a
  forward pass (acceleration limit) and a backward pass (braking limit), both
  wrapping round the lap, with a friction circle so turning and braking share
  grip. Grip has three zones: hairpins (R < 1 m, near full steering lock,
  where the front tyres hold only ~5 m/s²), normal corners, and open corners
  (R ≥ 2 m, with a 1.38 m buffer around tighter corners).
- **Steering** (`core/tracker.py`): δ = atan(L·κ_preview) + k_h·ψ_e +
  atan(−k_c·e/(v+1)). The curvature feed-forward reads the bend 0.15 s × speed
  ahead, so the 3.2 rad/s steering actuator gets its lag up front. The error
  terms are Stanley's. Above 4 m/s the heading gain falls as 4/v, which cut a
  ~1 Hz weave on the straights by 25%.
- **MPC** (`core/mpc.py`, `pursuit.tracker: "mpc"`, off): a 1 s horizon
  (20 × 0.05 s) linearised about the feed-forward steering, with the in-flight
  command modelled and solved as a small least-squares problem. On the real
  simulator it tracked the line to ~5 cm but was ~0.3 s a lap slower, so it
  is kept as an option.

Tuning was done on the **real simulator**, not on our offline model. We first
tried a Monte Carlo search on that model (`lab/montecarlo.py`). It sampled
random configurations, scored each over randomised "worlds" (grip error, sensor
noise, latency, start jitter), and ranked them by expected best-of-three lap
time including collision penalties. We dropped it for two reasons. The model it
searched (`tools/surrogate.py`) proved 9% optimistic with a rank correlation of
only 0.35 against real runs, so its rankings could not be trusted. And pure
random sampling wastes evaluations: affordable at 0.25 s a lap offline, but not
at about a minute per run on the real simulator. The search we kept was a
cross-entropy method
(`lab/real_cem.py`): 10 configurations per generation, the best 3 kept, 15
parameters searched at once. Each candidate was scored as lap time + 10 s per
collision. Elites were re-driven every generation so a lucky run could not win,
and runs where the simulator stuttered were never scored. In total we recorded
286 test-bench runs on the real simulator, plus the official runs.

What did not work: a grip ladder that walked the grip setting up until contact
reported 5.57 m/s² as the limit — it measured our cautious controller, not the
tyres; the IMU later showed ~9. Yaw-rate damping reduced the weave but caused
collisions at hairpin exits. Open-corner grip of 1.5 gave 9 collisions in 3
runs. Pose extrapolation for latency gave no gain. Iteratively reweighted
minimum-time lines were slower than plain minimum curvature. A config that
matched our shipped one for pace (10.68 s) was rejected because only about 1
run in 4 was clean. On 16 Oct we re-tested two assumptions on the real
simulator. Allowing full throttle earlier on corner exits (5.0, 6.5 or no
limit, against the shipped 3.74 m/s²) gave 1.7–2.3 collisions per 12-lap run
against 0.7, and slower averages (11.04–11.07 s against 10.92 s). Wider hairpin
lines (radius ≥ 1 m) were 0.4–0.5 s a lap slower in the model. Neither was
adopted.

## How it uses the inputs

- **LiDAR** (`…/lidar`): only for the follow-the-gap fallback — cleaned, then
  a safety bubble, then disparity extension, so a gap between two ducts never
  looks like a route. The LiDAR "blocked path" check is disabled
  (`mode.block_enter_frames: 1000000`): in a solo time trial nothing can block
  the path, and disabling it changed nothing measurable.
- **Pose** (`…/odom`): the simulator's **ground truth**, as rules 33 and 35
  permit, for the position on the line and for the speed and yaw rate. We have
  not written our own localisation.
- **IMU**: not used at run time. We used it offline to measure grip (~9 m/s²
  in normal corners, ~5 at full lock) and braking (7.8–8.8 m/s²).
- **Camera / encoders**: not used.

## Speed control

AutoDRIVE takes normalised torque, so we wrote the loop
(`core/speed_controller.py`). The feed-forward is the measured throttle map,
v ≈ 24.6·u (`tools/sysid.py`). A PI trim follows, with an integral clamped to
±0.30 that only integrates while unsaturated. More than 0.42 m/s under target
gives full throttle, unless the lateral load v²κ exceeds 3.74 m/s², which
prevents power-on understeer on corner exits. Up to 0.13 m/s over target, the
controller lifts off; beyond that it brakes at 0.96 throttle per m/s of
excess. There is no ABS, and a locked wheel cannot steer. The target speed is
read 0.13 s ahead to cover actuation latency. On the real simulator the car
follows the planned profile closely, within a few tenths of a m/s at the
corner minima.

## Results on our own machine

`results/submitted/` holds the referee's own file, unedited, from
`./scripts/evaluate.sh --team SpecialNeeds --no-simulator` with the submitted
code and no `--driver-params`, which is exactly how it is judged.

| File | Best lap | 10-lap total | Collisions | Status | Real-time factor |
| --- | --- | --- | --- | --- | --- |
| `SpecialNeeds__20261015T171039Z.json` | **10.687 s** | **107.84 s** | 0 | COMPLETE | 1.008 |

Timed laps were 10.69–10.97 s. This matches our own test bench on the same
code: 0 collisions in 4 of 4 runs, best laps 10.69–10.72 s, totals
107.9–108.6 s.

**A bug the official script found, now fixed.** `evaluate.launch.py` starts
the driver with no params file unless `--driver-params` is given, so the node
raced on the fallback literals in `driver.py` instead of the tuned
`config/driver_params.yaml`. Two official runs on 15 Oct were ~13 s a lap for
that reason (`results/SpecialNeeds__20261015T165827Z.json` and
`…170337Z.json`, kept locally and not submitted). The driver now loads its shipped
`driver_params.yaml` as its declared defaults (`shipped_parameters()` in
`driver.py`). A params file passed explicitly still overrides it, and
`tests/test_driver_imports.py` checks both cases.

### The machine that produced them

| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| Apple M3 Pro (arm64) | integrated | 18 GB | macOS | 1.0 (1.008 in the submitted file) |

We develop on Apple Silicon, where the Linux simulator binary cannot run. We
use the native macOS simulator on the host with the devkit in the official
container (`evaluate.sh --no-simulator`).

macOS throttles the simulator ("App Nap") when its window is not frontmost or
the display sleeps. Odometry gaps then grow from ~20 ms to 100–370 ms and the
driver tracks worse. Our test bench flags such runs and discards them. A
judging machine that keeps up in real time should not see this.

## Anything precomputed

| Artefact | Produced by | Regenerate with |
| --- | --- | --- |
| `maps/icra26_compete_raceline.csv` | `tools/build_raceline.py` (minimum-curvature start), then `tools/line_mintime.py` | From `race_ws/src/team_driver`: `python3 -m team_driver.tools.build_raceline --map ../../../maps/icra26_compete --centreline ../../../maps/icra26_compete_centerline.csv --margin 0.30 --out /tmp/m0300.csv`, then `python3 -m team_driver.tools.line_mintime --start /tmp/m0300.csv --kappa-max 1.3 --a-lat 8.0 --margin 0.25 --out ../../../maps/icra26_compete_raceline.csv`. Re-run on 15 Oct: the first step matches exactly, and the result matches the shipped line to within 1.8 cm (about 10 s on a laptop). |
| `config/vehicle.json` throttle map | `team_driver/tools/sysid.py` | `python3 -m team_driver.tools.sysid --log lab/data/sysid_2026-09-24.csv --map-only --base config/vehicle.json --out config/vehicle.json` |
| `config/vehicle.json` `a_brake_max` 8.5 | IMU braking on real runs (7.8–8.8 m/s²) | `lab/physics.py` |
| `config/driver_params.yaml` values | the real-simulator search | `lab/real_cem.py`, then `lab/finalize.py` |

No waypoint was placed by hand. The speed profile is not shipped: it is
computed at start-up from the line and `vehicle.json`, so it can never
describe a car we no longer believe in. No trained weights.

## Dependencies we added

**None.** `requirements.txt` and `apt-packages.txt` declare nothing. numpy,
scipy and PyYAML are already in the image.

| Package | pip or apt | What we use it for |
| --- | --- | --- |
| none | — | — |

## Third-party code and references

No third-party code. Ideas from:

- Heilmeier et al., *Minimum curvature trajectory planning and control for an
  autonomous race car* (2019) — the first racing line and the friction circle.
- Thrun et al., *Stanley: The robot that won the DARPA Grand Challenge* (2006)
  — the heading and cross-track terms of the tracker.
- Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*,
  CMU-RI-TR-92-01 — the alternative tracker.
- Sezer & Gokasan, *A novel obstacle avoidance algorithm: "Follow the Gap
  Method"* (2012), and Otterness, *The "disparity extender" algorithm* (2019)
  — the fallback.
- Rubinstein & Kroese, *The Cross-Entropy Method* (2004) — the parameter search.

## AI assistance

Two assistants, one after the other:

- **Gemini (Google Antigravity)**, early on, by all three of us: first
  experiments with racing-line optimisation and with finding the car's limits.
- **Claude (Anthropic)** for the stack we submit: to read the guides, design
  the architecture, write the code and tests, and run and analyse the
  experiments on the real simulator. The design decisions, measurements and
  negative results are recorded in `docs/HANDOVER.md`, `docs/superpowers/` and
  the commit messages.

What each member did (rule 41):

- **Naveen**: led the project and directed the work with Claude. Ran the
  real-simulator tuning sessions, adjusting parameters run by run (grip,
  braking, throttle, racing-line changes) and judging each change from lap
  times and collisions. Chose between configurations (the always-clean setup
  over the faster crash-prone one), ran the official runs, and caught the slow
  official run that exposed the settings-file bug.
- **Sachin**: worked on racing-line optimisation and on finding the limits of
  the car in the early (Gemini) phase; now testing the runs and tracking down
  bugs.
- **Yukesh**: worked on racing-line optimisation and on finding the limits of
  the car in the early (Gemini) phase; now testing the runs and tracking down
  bugs.

## Known issues

1. **The hairpin exits are the margin.** At the 0.8 m hairpins the steering
   is near full lock and the tyres hold ~5 m/s². In an 18-run test of faster
   settings, 73% of all collisions were at the exit of the second hairpin.
   `line.grip_fraction_tight` 0.90 is what keeps the car clean.
2. **Ground-truth pose dependency.** Without `/odom` the car falls back to
   follow-the-gap, which is far slower; we have not timed that fallback on the
   real simulator. Own localisation (a
   particle filter against our grid) is the obvious next piece.
3. **Real-time sensitivity.** The tuning assumes odometry at ~20–40 ms. A
   judging machine running well below real time would add effective latency.
4. **`a_lat_max` in `vehicle.json` (5.57) is a historical lower bound.** The
   grip fractions multiply it, so some fractions are above 1.0. The effective
   grips (5.0 / 6.2 / 7.8 m/s²) are below the measured ~9.
   `a_accel_max` (6.25) is optimistic too: measured at full throttle the car
   gives 4.5 m/s² at low speed and 3.0 at 8–10 m/s. It doesn't change what the
   car does, because it is already at full throttle wherever it is below target,
   so we left the shipped file alone.
5. Next with more time: re-optimise the line with the three-zone grip built
   in, tune the MPC for pace rather than precision, and build our own
   localisation.
