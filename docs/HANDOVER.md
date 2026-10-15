# Track 2 — Handover

Written 2026-09-25. Everything below is either measured or explicitly flagged
as an assumption. Read §4 before trusting any number.

> **§0 below (2026-10-03) supersedes §2, §4, §5 and §8.** They are kept
> because the reasoning — including what turned out wrong — is interview
> material.

---

## 0. Read this first — the overnight session of 2026-10-02/03

### Update 2026-10-16 — official run, a judged-path bug, and the exit gate re-tested

- **Bug found by the official script, fixed (7a10a9e):** `evaluate.launch.py`
  starts the driver with no params file, so it raced on the fallback literals
  in `driver.py` (~13 s laps, 2–9 collisions). The driver now declares the
  shipped `driver_params.yaml` values as its defaults. Official run after the
  fix: **10.687 s / 107.84 s, 0 collisions** (`results/submitted/`).
- **Exit full-throttle gate re-tested** (`lab/exit_gate_test.sh`,
  `results/realsim/eg_round*.log`): 0.7 s/lap is spent > 0.42 m/s under target
  with full throttle suppressed on exits, so `speed.full_throttle_max_lat`
  3.74 looked stale. Real sim, 12-lap runs, interleaved:

  | gate (m/s²) | runs | collisions/run | mean flying lap |
  |---|---|---|---|
  | **3.74 (shipped)** | 3 | **0.7** | **10.92 s** |
  | 5.0 | 4 | 2.0 | 11.06 s |
  | 6.5 | 3 | 2.3 | 11.04 s |
  | 12 (off) | 3 | 1.7 | 11.07 s |

  Looser gates give slightly better single laps (10.58–10.63 vs 10.66) but
  more collisions and slower averages: the exits are grip-limited, the
  "under target" time is not recoverable. Shipped value stays.
- Also measured: full-throttle acceleration is 4.5 m/s² at 2–6 m/s falling to
  3.0 at 8–10 m/s (vehicle.json says 6.25; harmless, the car is flat out
  there anyway). Wider hairpin lines (R ≥ 1 m) re-optimised with the
  three-zone grip model: 0.4–0.5 s slower in the model.
- **Simulator window size matters on the Mac.** Full Retina (3024×1898):
  render-bound, odometry ~40 ms. Small (960×540): the bridge saturates a core,
  ~12 ms median but 50–70 stalls >100 ms per run. 1920×1080 windowed
  (`-screen-fullscreen 0 -screen-width 1920 -screen-height 1080`) is the
  usable middle (~20 ms). The Mac in use (Claude app, Telegram) adds stalls.

### Update 2026-10-07 — shipped: braking 8.5 + tight-corner grip (config "H")

Real simulator, judged format, shipped files: **0 collisions in 4 of 4 runs**,
best laps 10.69–10.72 s, 10-lap totals **107.9 / 108.6 / 108.3 / 108.2 s**.
The fast c83 config below was clean in ~1 run of 3 (totals 108–129 s).

- `vehicle.json` `a_brake_max` 7.59 → **8.5** (IMU recorded 7.8–8.8 m/s² of
  real braking; alone: best lap −0.15 s, collision rate unchanged).
- `line.grip_fraction_tight` **0.90** for `tight_radius` **1.0 m**: 73% of the
  30 collisions in the 18-run upgrade test were at the exit of the 0.87 m
  hairpin (s 34–39) — near full lock the tyres hold ~5 m/s² — MPC included.
- Tested and not adopted: open-corner grip 1.5 (9 collisions in 3 runs),
  MPC steering (`tracker: mpc`, most reliable before this fix but ~0.3 s/lap
  slower; kept, off). Results: `results/realsim/up_*`, `tt_*`, `shipped_H_12lap`.

### Where we are — two configs, one decision

Both driven **exactly as judged** (shipped files, no `--driver-params`, out
lap + warm-up + 10 timed laps) on the real simulator:

| | best lap | 10-lap total | collisions per 12-lap run | commit |
|---|---|---|---|---|
| **c83 — fast (shipped at HEAD)** | **10.68–10.73 s** | **107.74 s** clean; 118.7 / 129.9 / 140.2 with 1–3 hits | 0, 2, 2, 3 — about 1 run in 4 clean | `e6a1ece` |
| **c6 — safe** | 11.27 s | 113.53 s | 0 in every run (7 runs) | `d42e18b` |

The judges take the best of three runs. Expected score is higher for c83
(~97 vs ~95 against a 10.7 s / 107.7 s leader) and its best lap alone likely
takes the full 50; c6 never has a bad run. To ship c6:
`git checkout d42e18b -- race_ws/src/team_driver/config/driver_params.yaml`
(the line and vehicle files are identical; the newer open-corner keys fall
back to their off defaults).

Was 12.588 s best (3 laps) on 2026-09-25. Results in `results/realsim/`
(`shipped_c83_12lap_*`, `val12_*`, `shipped_c6_12lap`). Still to do: one
official `./scripts/evaluate.sh` run for `results/submitted/`.

c83's collisions are spread over the lap — wherever the car drifts ~0.45 m
at 6–7.5 m/s (s ≈ 2.6, 26.9) and at the corner-3 exit (s ≈ 37.3). Detuning
(exit buffer 2.0 m, open-corner grip 1.30) did not remove them in 12-lap
tests. The next lever is tracking at speed, not grip.

### The benchmark that matters

**The world best on this exact circuit is 11.00 s** — MonacoF1, ICRA 2026
RoboRacer Sim Racing, Phase 2 final (110.27 s for 10 laps, 0 collisions).
[Results](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2026/).
The sub-8 s laps people quote (VAUL 6.46 s, 7–8 s for most of the top 10)
are from **Phase 1, the practice track** — a different circuit. NTU DeepSpeed
appears in both tables. Against MonacoF1's times our shipped run scores
≈ 48.8 + 48.6 = **97.4 / 100**.

### What was measured, and what it overturned

| Claim in §4/§5 | Measured on the real car | How |
|---|---|---|
| grip limit 5.57–7.24 m/s² | **≈ 9 m/s²** — the ladder measured our controller, not the tyres | IMU + actual steering, ~4,600 cornering samples, achieved/kinematic curvature ratio (`lab/physics.py`) |
| throttle map v = 39.8u − 19.9u² | **v ≈ 24.6u** — ~60% too steep, so feed-forward under-throttled by ~38% | the clean `sysid.csv` sweep points (0.08–0.25) |
| `a_accel/a_brake/a_coast` measured | **contaminated** — the "longitudinal" phase logs steering 0 while yaw rate says full lock (R ≈ 0.56 m), resetting every 0.65 s: a car circling into a wall | same file, rows 9–60 |
| corners grip-limited | the two inner hairpins (s ≈ 30–37) are **steering-lock-limited**: full lock even on clean laps | actual steering feedback |
| 7 s is reachable "at 1.5 g" | 7 s needs ~14.7 m/s² **and** ~10/12 m/s² accel/brake; the car has ~9 | ideal-profile model |

Every collision in the high-speed runs was the car running **~0.3 m wide on
a corner exit** (s = 32.5 and s = 37.3) — `lab/collisions.py`.

### What changed in the driver (all with tests; 213 green)

- `speed.full_throttle_band` — far under target → throttle 1.0. The profile
  plans accelerations at the traction limit; feed-forward + a trim (the
  integral is worth ≤ ki·0.30 = 0.045 throttle) crept up on them.
- `speed.full_throttle_max_lat` — that band only below this lateral load,
  against power-on understeer on exits.
- `speed.preview_s` — target speed read ahead, covering actuation latency.
- `line.friction_circle` — the profile shares grip between turning and
  braking (`a_x ≤ a_x,max·√(1 − (a_y/a_y,max)²)`).
- `tools/line_mintime.py` — racing line optimised for **lap time** with a
  steering-authority curvature cap; wall clearance measured on the free-cell
  distance field, so the grey outfield counts as wall and no line threads a
  gap between ducts (rule 11). Corner-by-corner entry/apex/exit table.
- `pose.extrapolate` / `pose.latency_s` — CTRV pose projection. **Tested, no
  gain on the real car; off.**
- LiDAR "blocked path" check disabled (`mode.block_enter_frames: 1000000`) —
  a solo time trial has nothing to block the path; disabling it changed
  nothing measurable. Follow-the-gap remains the fallback for pose loss.

### Tooling (all in `lab/`, not shipped)

| | |
|---|---|
| `realsim.py` + `rs.sh` | test bench on the real simulator through **one persistent bridge**; resets like the referee, times laps, records telemetry, flags stutter (`INVALID_TIMING`) and ignored commands |
| `real_cem.py` | cross-entropy search run **on the real simulator** (`--resume`, `--restart-bridge-every`) |
| `finalize.py --cid N` | ships a candidate into the config in place (comments survive) |
| `physics.py`, `collisions.py`, `offsets.py`, `draw_track.py` | what the car physically did, where it hit, where it left the line |
| `calib_real.py` | surrogate vs real runs — **9% optimistic, rank correlation 0.35**: do not optimise against the surrogate |

### Operational gotchas (each cost an hour)

1. **macOS throttles the Unity simulator** when its window isn't frontmost or
   the display sleeps: odometry goes from ~17 ms to 100–370 ms bursts and every
   config looks bad. `caffeinate -d -i -w <sim pid>`, keep it frontmost,
   don't use the Mac during runs. `realsim.py` reports the gaps.
2. **The bridge degrades over hours** (one core at 100%, odometry 17 → 40 ms).
   Restart it (`lab/restart_bridge.sh`); the simulator reconnects by itself.
3. **`evaluate.sh` restarts the bridge**, and after a restart the simulator
   can keep publishing while ignoring every command — it once drove a stale
   0.52 throttle into a wall for an 11-collision DQ. The bench checks that
   the actual throttle follows the command before every run.
4. **ROS 2 parameter types are strict**: `v_max: 18` is an INTEGER and the
   driver declares a DOUBLE → the node refuses to start (DNF). Now tested.
5. `pkill -f pattern` inside `bash -lc "..."` matches its own shell — use
   `'[p]attern'`.

---

## 1. The competition, in one box

| | |
|---|---|
| Event | NTU DeepSpeed Recruitment Hackathon 26/27, **Track 2** (RoboRacer Sim Racing on AutoDRIVE) |
| Submission | **18 Oct 2026, 23:59 SGT** — commit timestamp on the submitted branch is what counts |
| Circuit | ICRA 2026 `compete`, 54.43 m lap, 6.3 × 18.2 m, ~1.8 m wide, **tightest radius 0.83 m** |
| Control | Normalised **torque** in [−1,1], not a speed. You own the speed loop. |
| LiDAR | 1080 beams, 270°, **10 m** range, 40 Hz |
| Score | `50·(L*/L) + 50·(T*/T)`, both **after** penalties |
| Penalty | **+10 s per collision**; **>10 collisions = DQ** (zero for both halves) |
| Judging | Best of **3** runs, one machine, headless |
| Entry point | `ros2 run team_driver driver` — must keep working |

Marginal rates near leader pace: 1 s/lap ≈ **4.8 points**; one collision ≈ **2.3
points** (≈ half a second a lap); the 11th collision ≈ **100 points**.

---

## 2. Where we actually are

**Verified on the real AutoDRIVE simulator**, official referee, scored run:

```
status        : COMPLETE
laps          : 3/3
collisions    : 0/10
BEST LAP      : 12.588 s
3-LAP TOTAL   : 38.854 s
real-time factor: 1.01      callback_errors: 0
```

Result file: `results/first_run__20260924T170940Z.json`

**The surrogate predicted 12.28 s. The simulator gave 12.588 s — 2.4% error.**
That is the single most useful fact in this document: offline search is
trustworthy, and a surrogate lap costs 0.25 s against the real simulator's
~6 minutes.

Branch: **`feat/racing-stack`** (19 commits, ~7k lines, **199 tests green**).
`track2` is untouched — `git checkout track2` reverts everything. Nothing pushed.

---

## 3. Architecture

Three tiers, hard boundaries. The point is that anything testable without the
simulator is worth a hundred times anything that is not.

```
race_ws/src/team_driver/
├── team_driver/driver.py        ROS adapter. ~280 lines, NO algorithm logic.
├── team_driver/core/            PURE PYTHON, ZERO ROS IMPORTS (enforced by test)
│   ├── vehicle.py               actuator map + measured limits
│   ├── speed_controller.py      feed-forward + clamped-integral PI
│   ├── lidar.py                 clean / safety bubble / disparity extension
│   ├── gap.py                   follow-the-gap  (permanent fallback)
│   ├── track.py                 occupancy grid + centreline geometry
│   ├── pursuit.py               pure pursuit
│   ├── tracker.py               curvature feed-forward + Stanley  (DEFAULT)
│   ├── profile.py               curvature → speed, fwd/bwd passes
│   ├── planner.py               composition + hysteretic gap fallback
│   ├── params.py                ROS param ↔ dataclass maps (guarded by tests)
│   └── build.py                 loads racing line + builds profile at startup
├── team_driver/tools/           offline, shipped (rule 36 needs the generator)
│   ├── surrogate.py             numpy simulator, ~0.25 s/lap, + Perturbation
│   ├── line_optimise.py         minimum-curvature racing line (QP)
│   ├── build_raceline.py        CLI → maps/icra26_compete_raceline.csv
│   ├── sysid.py / sysid_drive.py   system identification
│   ├── grid_build.py / map_record.py  build our own occupancy grid
│   └── scoring.py               the rules' scoring maths
└── tests/                       199 tests, run in ~17 s
lab/                             dev tooling, NOT shipped
├── sweep.py                     grid search vs surrogate
├── montecarlo.py                random search + perturbed rollouts  ← NEW
├── report.py                    aggregates results/*.json
└── run_on_mac.sh                one-command bring-up
```

Two modes: **line** (racing line + speed profile against ground-truth pose) and
**gap** (reactive, needs only the scan). Switch is hysteretic. Gap is the
permanent fallback for: no pose, stale pose, implausible pose, blocked path.

Run tests:
```bash
cd race_ws/src/team_driver && <venv>/bin/python -m pytest -q
```
venv: `/private/tmp/claude-501/.../scratchpad/venv` — **recreate if gone**:
`python3 -m venv venv && ./venv/bin/pip install numpy scipy pytest pyyaml`

---

## 4. Measured vs assumed — READ THIS

`config/vehicle.json`, from a real identification run (`results/sysid.csv`, 1918 samples):

| parameter | value | status |
|---|---|---|
| `a_brake_max` | 7.59 | measured (52% more than I'd assumed) |
| `a_accel_max` | 6.25 | measured |
| `v_max` | 19.90 | measured, monotonicity-constrained |
| `throttle_v_coeffs` | [39.80, −19.90] | measured over throttle 0.08–0.35 only |
| `a_coast` | 7.59 | **suspect** — fitted 8.37, clamped to ≤ brake. Coast samples contaminated by cornering. |
| **`a_lat_max`** | **5.57** | **LOWER BOUND, NOT A LIMIT** |

`grip_saturated: false`. The car **never reached the edge of grip** during
identification — the gap follower steering it was too conservative — so 5.57
means *at least this much*. Everything downstream inherits this.

**Why it decides the whole project:**

| real grip | theoretical best lap |
|---|---|
| 5.57 (if that IS the limit) | ~11.3 s |
| 9.81 (1.0 g) | 8.5 s |
| **14.7 (1.5 g)** | **6.98 s** |

A teammate reportedly did **~7 s** on this circuit. That needs **1.49 g**, which
is entirely physical for an RC tyre on concrete. So 7 s is achievable *if* the
real grip is near 1.5 g — and we have no evidence either way yet.

---

## 5. The grip limit — MEASURED, and it resets expectations

Walked `line.grip_fraction` up on the real simulator, 5 laps each, until
contact started:

| grip_fraction | a_lat asked | laps | collisions | best lap | verdict |
|---|---|---|---|---|---|
| 0.70 | 3.90 | 5 | **0** | 13.524 s | CLEAN |
| **1.00** | **5.57** | 5 | **0** | **13.096 s** | **CLEAN** |
| 1.30 | 7.24 | 5 | 12 | 12.741 s | over the limit |
| 1.60 | 8.91 | 5 | 20 | 14.689 s | over the limit |
| 2.00 | 11.14 | 5 | 20 | 15.179 s | over the limit |
| 2.50 | 13.93 | 5 | 20 | 15.038 s | over the limit |

**The car's real lateral grip limit is between 5.57 and 7.24 m/s² — about
0.6–0.7 g.** Not the 1.0–1.5 g a real RC tyre manages; this is a Unity
vehicle model, not a tyre.

### What this means for the target

A ~7 s lap on this circuit requires **14.7 m/s² (1.49 g)**. The car delivers
roughly **half that**. **7 s is not physically reachable on the `compete`
circuit with this vehicle**, by any algorithm.

If a 7 s time is genuine it was almost certainly *not* this configuration —
a different AutoDRIVE build (`practice` / `explore` have different circuits),
a sector rather than a full lap, or the simulator's own HUD clock rather than
the referee's. Worth establishing before chasing it.

**Realistic target: ~10.5–11 s.** That is the idealised floor at this grip on
the current racing line, against 12.588 s measured today — so there is roughly
1.5–2 s of real headroom, and it lives in tracking quality and the speed
profile, not in cornering harder.

### The second finding, which is just as useful

Past the limit, lap times get **worse**, not merely riskier: 14.689 / 15.179 /
15.038 s at grip 1.60–2.50, against 13.096 s clean. Exceeding grip makes the
car understeer wide, which costs more time than the extra speed buys — before
counting the +10 s per collision. At grip 1.30 the raw best lap was 12.741 s,
marginally quicker than clean, but with **12 collisions = +120 s and a
disqualification** (limit is 10).

**Operating point: grip_fraction 1.00 on the measured `a_lat_max` of 5.57.**
Anything above ~1.15 should be treated as off the table.

## 6. How to run it

```bash
./lab/run_on_mac.sh              # container → workspace → bridge → sim → driver
./lab/run_on_mac.sh --evaluate   # ...then a scored 3-lap run
```

**Five things that will waste your afternoon otherwise:**

1. **`scripts/run_simulator.sh` is broken on macOS.** Line 89 is
   `exec "${prefix[@]}" "${args[@]}"` under `set -u`; expanding an empty array
   that way is fatal on **bash 3.2, which macOS ships**. Launch the binary
   directly instead (`run_on_mac.sh` does). `scripts/` is protected — don't patch it.
   **→ Report this to ntu-deepspeed@e.ntu.edu.sg. Rule 49 credits genuine bug reports.**

2. **The simulator boots `Disconnected` and you must click Connect in the GUI.**
   `-ip`/`-port` only *prefill* the fields. Symptom if you forget: ESTABLISHED
   socket, devkit prints `Connected!`, full topic list, **zero messages**.
   Also set the vehicle to **Autonomous**.

3. **`evaluate.sh` tears down the shared ROS context** and kills a
   manually-started bridge (`publisher's context is invalid`). Either let
   `evaluate.sh` own the bridge, or restart it after. The sim **does**
   auto-reconnect when the bridge comes back, so this is recoverable unattended.

4. **The `ros2` CLI daemon goes stale** → `xmlrpc.client.Fault: !rclpy.ok()`.
   Use `ros2 daemon stop`, or bypass the CLI with a small rclpy script
   (`/tmp/_probe.py` in the container is exactly that).

5. **Apple Silicon is fine.** The image builds native arm64 (greenlet/gevent
   compile cleanly); the simulator is a universal binary with a native arm64
   slice. No emulation needed. But the **judging path is x86 Linux**, and
   GitHub Actions is the only place we can exercise it.

---

## 7. Bugs found and fixed (all have regression tests)

The ones worth knowing, because they shaped the design:

- **`brake_gain` drift** — `PlannerConfig` said 0.30 while `params.py` said 1.0.
  The ROS node braked at full authority; **every offline experiment braked at a
  third of it**. Invalidated a whole 48-config sweep and manufactured a false
  conclusion ("tracking is the binding constraint, grip cliffs above 0.35" — it
  isn't and it doesn't). Now guarded by a test that the **defaults agree**.
- **`v_max` = −60.45** from extrapolating the throttle fit past its sampled
  range → `throttle_for_speed` returns 1.0 for everything → **permanent full
  throttle**. Fit is now monotonicity-constrained.
- **Blocked-path check used line-of-sight** to a path point. On 0.83 m corners
  the chord leaves the corridor by geometry, so it reported an obstruction at
  every corner entry on a clear track. Now tests whether anything sits *on* the path.
- **Collision debounce never re-triggered** — a car wedged on a wall scored one
  collision, teaching sweeps to prefer scraping.
- **Progress measured as trajectory length** — a car that left the circuit and
  drove straight accumulated a "lap" and was recorded COMPLETE.
- **Pure-pursuit lookahead exceeded the corner radius** (2.2 m into a 1.1 m
  corner), tracking a chord through the boundary. Now capped at a fraction of
  the local radius.
- **`driver.py` had three undefined names and never called `load_path`** — the
  whole line-following stack was dead code in the shipped entry point. Nothing
  imported the node because it needs ROS. Now there are stubbed-ROS tests that
  **construct** it.
- **Stale pose believed indefinitely** — a frozen `/odom` sits exactly *on* the
  line, so the cross-track guard never fires and the car follows the line
  open-loop. Now expires after `pose_timeout_s` (0.25 s) → gap mode.

---

## 8. What I'd do next, ranked

1. **Finish the grip ladder** (§5). Nothing else is worth tuning until
   `a_lat_max` is a limit rather than a floor.
2. **Re-run `lab/montecarlo.py`** with the corrected vehicle. It ranks by
   **expected best-of-three net lap time** under model error, sensor noise,
   actuation latency and start jitter — i.e. the rules' own objective, not raw
   lap time. `--candidates 400 --rollouts 8`, parallel across cores.
3. **Regenerate the racing line** at the winning margin, confirm the top 3
   configs on the real simulator over 3 runs each, and compare against the
   surrogate's prediction (`lab/report.py --surrogate`).
4. **Re-do the grip phase of sysid properly** — fixed steering angle, open-loop
   rising throttle until lateral acceleration plateaus. The current phase lets
   the gap follower steer, which never reaches the limit.
5. **Then get ambitious**: MPPI over the identified model (CPU/numpy, fits the
   25 ms budget, no CUDA needed — we can't validate GPU code on the judging
   hardware anyway). That's also the "ambitious algorithm" bonus.

**Measured negative result, don't redo it:** iteratively-reweighted
*minimum-time* line optimisation is **slower** here than plain minimum
curvature (7.51/7.54/7.70/8.03 s vs 7.49 as the exponent rises). The circuit is
too narrow for the weighting to help. Code is kept and tested; the finding is
pinned in `test_line_optimise.py`.

---

## 9. Submission status

- [x] Driver runs from `ros2 run team_driver driver`
- [x] **Zero declared dependencies** (`requirements.txt` / `apt-packages.txt`
      empty of real entries) — removes the single most common silent-zero
- [x] `./scripts/verify_judging_env.sh` — **intact**, 64 protected files match
- [x] Artefacts reproducible from committed code (rule 36)
- [x] `SUBMISSION.md` drafted
- [ ] **Team members / matric numbers** — TO FILL
- [ ] `results/submitted/` — needs real 10-lap `COMPLETE` runs (we only have a 3-lap)
- [ ] Judge workflow green on the pushed branch
- [ ] Clean-clone test (`setup.sh --no-cache`)

**Interview prep:** every design decision, ruling and measured negative result
is in the git history, `docs/superpowers/specs/` and `docs/superpowers/plans/`.
Any member can be asked about any part (rule 41) — the commit messages are
written to be readable as explanations, not changelogs.
