# Track 2 — Handover

Written 2026-09-25. Everything below is either measured or explicitly flagged
as an assumption. Read §4 before trusting any number.

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
