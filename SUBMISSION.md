# Team &lt;name&gt;

> **Status: draft.** Everything marked TO FILL needs the team's own details or
> a measurement we have not taken yet. Nothing here has run against the real
> AutoDRIVE simulator: all numbers are from our offline surrogate, against
> assumed vehicle limits. See "Known issues" before reading any of them.

## Members

TO FILL — 3 to 5 members, roster fixed at submission (rule 3).

| Name | Matric number | Email |
| --- | --- | --- |
| ... | ... | ... |

## Approach

A classical racing stack in three tiers. The ROS node
(`race_ws/src/team_driver/team_driver/driver.py`) is a thin adapter: it
converts messages, calls one function, publishes two `Float32`s, and never
raises out of the callback. Everything that decides anything lives in
`team_driver/core/`, which imports no ROS and is therefore unit-tested in
milliseconds rather than in six-minute simulator runs. Offline tools in
`team_driver/tools/` generate the artefacts the driver loads.

The driver has two modes. **Line mode** follows a minimum-curvature racing
line against ground-truth pose, at a speed the offline profile computed for
that point — so it brakes for a corner before the 10 m LiDAR can see it,
which on a 54 m lap is most of the available lap time. **Gap mode** is
follow-the-gap with disparity extension, needs nothing but the scan, and
takes over whenever the line cannot be trusted: no pose, an implausible pose,
too far off the line, or something sitting on the path ahead. The switch is
hysteretic, because a mode that flips every frame steers worse than either
mode alone.

What did not work, and why it is worth saying: our first surrogate lap took
40 seconds with the car in gap mode 92% of the time. The blocked-path check
was testing line-of-sight to a path point, and on corners of about a metre
radius the straight chord to a point one braking distance ahead leaves the
corridor by geometry — so it reported an obstruction at every corner entry on
a completely clear track. Three further defects came out of the same trace
(see Known issues). That is what the offline simulator is for.

## How it uses the inputs

- **LiDAR** (`.../lidar`): cleaned, then a safety bubble around the closest
  return, then disparity extension pulling the far side of every near/far
  jump in by half a car width. That last step is not optional here — the
  boundary is a row of separate 33 cm ducts, so beams slip between them and
  report the far side of the track, and without it a gap follower steers
  between two ducts, which is both a collision and leaving the course.
- **Pose** (`.../odom`): the simulator's ground truth, which rule 35 permits
  and §2.6 recommends. Used for the nearest point on the racing line, the
  pure-pursuit target, and the speed profile lookup. We have **not** written
  our own localisation.
- **Camera / IMU / encoders**: not used. The camera produces no image in a
  headless run, and speed from `/odom` is already exact.

## Speed control

AutoDRIVE takes normalised torque, not a speed, so we wrote the loop.
Feed-forward from the measured actuator map (`Vehicle.throttle_for_speed`,
solved in closed form) carries most of the load; a PI trim with a clamped
integral cleans up the rest. Past a lift band the car coasts rather than
brakes, because there is no ABS and a locked wheel steers nowhere; past a
larger band it brakes properly.

Getting the brake authority right mattered more than expected. The speed
profile plans its braking at `a_brake_max`, and our first controller capped
real deceleration at about a third of that, so the car arrived at every
corner too fast and ran wide. The gain and the profile now describe the same
car.

## Results on our own machine

**None yet.** `results/submitted/` is empty: we have not run
`./scripts/evaluate.sh` against the real simulator. TO FILL before submitting
— at least one run with `"status": "COMPLETE"` is required (chapter 7).

Offline surrogate results, for context only, from a 48-configuration sweep:

| | best net lap | collisions | completed |
| --- | --- | --- | --- |
| Minimum-curvature racing line, grip 0.30 | 15.42 s | 0 | 12/12 |
| Shipped centreline, grip 0.40 | 17.10 s | 0 | — |

These are **not** simulator results and must not be read as any kind of
prediction. See Known issues.

### The machine that produced them

| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| Apple M3 Pro (arm64) | — (no CUDA) | 18 GB | macOS | TO FILL |

Note for the judges: we develop on Apple Silicon, where the Linux simulator
binary cannot run. Locally the native macOS simulator runs on the host with
the devkit in the container (`evaluate.sh --no-simulator`), so **we never
execute the judged code path on our own machines**. The GitHub Actions Judge
workflow is our only access to x86 Linux, and we lean on it as a correctness
gate rather than for its times.

## Anything precomputed

| Artefact | Produced by | Regenerate with |
| --- | --- | --- |
| `maps/icra26_compete_raceline.csv` | `team_driver/tools/line_optimise.py` | `python3 -m team_driver.tools.build_raceline --map maps/icra26_compete --centreline maps/icra26_compete_centerline.csv --out maps/icra26_compete_raceline.csv --margin 0.45` |
| `config/vehicle.json` | `team_driver/tools/sysid.py` | `ros2 run team_driver sysid_drive`, then `python3 -m team_driver.tools.sysid --log results/sysid.csv --out race_ws/src/team_driver/config/vehicle.json` |

The racing line is a bounded linear least-squares solve for the lateral
offsets that minimise total squared curvature, subject to the measured free
distance to the boundary either side less a margin — the standard
minimum-curvature formulation (Heilmeier et al. 2019). No waypoint in it was
placed by hand.

The speed profile is **not** shipped as an artefact: it is computed at
startup from the line and `vehicle.json`, so changing the identified grip
changes the profile with it instead of leaving a stale file describing a car
we no longer believe in.

No trained weights. We have no learned components.

## Dependencies we added

**None.** `requirements.txt` and `apt-packages.txt` are empty of real
entries. numpy, scipy, PyYAML, OpenCV and Pillow are already in the image,
and `requirements.txt` warns against replacing numpy or transforms3d, so
shipping nothing removes the largest silent failure mode in the competition.

| Package | pip or apt | What we use it for |
| --- | --- | --- |
| none | — | — |

## Third-party code and references

No third-party code. Ideas from:

- Sezer & Gokasan, *A novel obstacle avoidance algorithm: "Follow the Gap
  Method"* (2012) — the gap follower.
- Otterness, *The "disparity extender" algorithm* (2019) — disparity extension.
- Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*,
  CMU-RI-TR-92-01 — the tracker.
- Heilmeier et al., *Minimum curvature trajectory planning and control for an
  autonomous race car* (2019) — the racing line formulation.

## AI assistance

Claude (Anthropic) was used throughout: to read the track guides, design the
architecture, and write the code and its tests. The design decisions, the
rulings taken along the way, and the defects found are recorded in
`docs/superpowers/specs/`, `docs/superpowers/plans/` and the git history —
every commit message states what changed and why.

TO FILL — each member should add what they personally worked on and can
explain, since any member may be asked about any part (rule 41).

## Known issues

1. **Nothing has run on the real simulator.** Every number in this document
   comes from our own offline surrogate. It is a model and it can lie; the
   surrogate-versus-simulator correlation that would tell us how far to trust
   it has not been measured. `lab/report.py --surrogate` computes it.
2. **`config/vehicle.json` has `measured: false`.** Its grip, braking and
   coast limits are derived from the published vehicle specification, not
   measured from the car, and the driver logs a warning about this on
   startup. `ros2 run team_driver sysid_drive` is the first thing to run.
3. **Tracking accuracy is the binding constraint, not speed.** Above about
   grip fraction 0.35 the car can corner that fast but cannot *track* the
   line there: the 3.2 rad/s steering rate limit costs roughly 0.8 m of
   travel to reach lock at 5 m/s on a 1.1 m radius corner, so it runs wide
   and leaves the circuit. In the sweep, grip 0.30 and 0.35 completed 24/24
   while 0.40 completed 4/12 and 0.45 completed 0/12. Adding curvature
   feed-forward to the steering — turning into the corner immediately rather
   than waiting for cross-track error to build — is the single biggest
   lap-time item left, and would unlock the racing line's remaining margin.
4. **No own localisation.** We use the simulator's ground-truth pose, which
   the rules permit. A particle filter against our own grid is the obvious
   next piece.
