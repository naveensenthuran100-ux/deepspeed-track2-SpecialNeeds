#!/usr/bin/env python3
"""Monte Carlo search for the fastest configuration that is also robust.

    python3 lab/montecarlo.py --candidates 400 --rollouts 8
    python3 lab/montecarlo.py --quick

WHAT IS BEING OPTIMISED
-----------------------
Not lap time. The rules' own objective:

  * a collision is +10 s on the lap it happened on, so net time already
    prices contact correctly and needs no separate safety weight;
  * more than 10 collisions in a run is a disqualification, worth zero for
    BOTH score components - a cliff, not a slope;
  * the judges run each entry THREE times and take the best attempt.

That last point changes the answer. A configuration completing 60% of the
time delivers a scoring run 94% of the time, so it is worth more than its
average run suggests. Ranking by mean lap time would pick something timid;
ranking by best-case would pick something that crashes two runs in three.
So candidates are ranked by **expected best-of-three net lap time**, with a
failed run counted as infinite.

WHAT IS BEING RANDOMISED
------------------------
Two different things, and conflating them is the usual mistake:

  * the SEARCH samples configurations (racing line margin, grip fraction,
    controller gains...) uniformly from the space below;
  * the EVALUATION samples the world - model error, sensor noise, actuator
    latency, start jitter - so each configuration is scored on a
    distribution of worlds rather than one lucky one.

`vehicle_scale` is the most important of those. The planner keeps the
nominal vehicle from vehicle.json while the dynamics integrate a scaled one,
which asks directly: what if the grip we measured is 15% optimistic? A
configuration that only survives at scale 1.0 is one bad measurement from a
DNF, and this search will not rank it highly.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws", "src", "team_driver"))

from team_driver.core.planner import Planner, PlannerConfig          # noqa: E402
from team_driver.core.profile import speed_profile                   # noqa: E402
from team_driver.core.track import Centreline, OccupancyGrid         # noqa: E402
from team_driver.core.vehicle import Vehicle                         # noqa: E402
from team_driver.tools.line_optimise import optimise_raceline        # noqa: E402
from team_driver.tools.surrogate import (Perturbation, SurrogateSensor,  # noqa: E402
                                         simulate)

MAPS = os.path.join(REPO, "maps")

#: Racing line clearance is discretised because each value needs its own
#: optimisation; the rest are continuous.
MARGINS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

SPACE = {
    "grip":          (0.45, 1.00),   # fraction of a_lat_max the profile asks for
    "v_max":         (6.0, 16.0),    # ceiling on target speed, m/s
    "tau_preview":   (0.06, 0.30),   # seconds of travel to read curvature ahead
    "min_preview":   (0.20, 0.70),
    "k_cross":       (1.0, 4.0),
    "k_heading":     (0.40, 1.80),
    "lift_band":     (0.05, 0.60),
    "brake_gain":    (0.60, 1.50),
    "kp":            (0.02, 0.15),
    "ki":            (0.05, 0.40),
    "curvature_cap": (0.50, 1.20),
}

#: The world we are uncertain about. sd on vehicle_scale is deliberately
#: wide: a_lat_max comes from a short identification run, not a rig.
#: What the referee actually runs (scripts/evaluate.sh defaults), and the
#: rule that makes collisions a cliff rather than a slope. The search
#: simulates fewer laps than this for speed, so the DQ test extrapolates.
JUDGED_LAPS = 10
WARMUP_LAPS = 2
COLLISION_LIMIT = 10

#: A reference leader lap for turning lap times into points. Only the SHAPE
#: matters for ranking - it sets how much lap time one failed draw is worth -
#: and anything near the realistic front of the field gives the same order.
REF_LAP_S = 10.5

MODEL_SCALE_SD = 0.13
SCAN_NOISE_M = 0.04
STEER_NOISE = 0.015
THROTTLE_NOISE = 0.02
START_JITTER_M = 0.04
START_YAW_JITTER = 0.02
MAX_DELAY_STEPS = 2

_G = {}


def _init(vehicle_dict, laps, margins=None):
    """Per-worker setup. The distance transform and the racing lines are
    expensive and identical for every candidate, so they are built once."""
    grid = OccupancyGrid.load(os.path.join(MAPS, "icra26_compete.pgm"),
                              os.path.join(MAPS, "icra26_compete.yaml"))
    centre = Centreline.load(os.path.join(MAPS, "icra26_compete_centerline.csv"))
    _G["grid"] = grid
    _G["sensor"] = SurrogateSensor(grid, max_range=10.0, beam_count=181)
    _G["vehicle"] = Vehicle(**vehicle_dict)
    _G["laps"] = laps
    _G["lines"] = {m: optimise_raceline(centre, grid, margin_m=m)
                   for m in (margins or MARGINS)}


def _evaluate(job):
    index, params, rollouts = job
    outcomes, _ = run_rollouts(params, [10_000 * index + roll for roll in range(rollouts)])
    return {**params, **_summarise(outcomes)}


def run_rollouts(params, seeds):
    """One perturbed run per seed: (net seconds per lap, collisions) lists.

    Each seed fixes one WORLD - model error, noise, latency, start jitter.
    Passing the same seeds for every candidate compares them on identical
    worlds (common random numbers), which is what makes a ranking between
    them mean something rather than measuring whose worlds were kinder.

    A seed may instead be a pair (car_seed, run_seed): the model error then
    comes from car_seed and everything else from run_seed, so several runs
    can share one CAR while their noise differs. That is the shape of the
    judging - three runs of one real vehicle - and see summarise_events for
    why it matters.
    """
    vehicle = _G["vehicle"]
    grid, sensor = _G["grid"], _G["sensor"]
    line = _G["lines"][params["margin"]]
    laps = _G["laps"]

    profile = speed_profile(line.curvature, line.s, vehicle,
                            grip_fraction=params["grip"], v_max=params["v_max"],
                            total_length=line.length)
    yaw0 = float(np.arctan2(line.tangent[0][1], line.tangent[0][0]))
    start = (line.xy[0][0], line.xy[0][1], yaw0)

    outcomes, collisions = [], []
    for seed in seeds:
        car_seed, run_seed = seed if isinstance(seed, tuple) else (seed, seed)
        rng = np.random.RandomState(run_seed)
        car = rng if car_seed == run_seed else np.random.RandomState(car_seed)
        planner = Planner(vehicle, PlannerConfig(
            tracker="feedforward",
            tau_preview=params["tau_preview"], min_preview=params["min_preview"],
            k_cross=params["k_cross"], k_heading=params["k_heading"],
            lift_band=params["lift_band"], brake_gain=params["brake_gain"],
            kp=params["kp"], ki=params["ki"],
            curvature_cap=params["curvature_cap"],
        ))
        planner.load_path(line, profile)

        result = simulate(
            planner, vehicle, grid, start=start,
            distance_m=line.length * laps, max_time_s=40.0 * laps, sensor=sensor,
            perturbation=Perturbation(
                seed=int(rng.randint(1 << 30)),
                scan_noise_m=SCAN_NOISE_M, steer_noise=STEER_NOISE,
                throttle_noise=THROTTLE_NOISE,
                command_delay_steps=int(rng.randint(0, MAX_DELAY_STEPS + 1)),
                start_jitter_m=START_JITTER_M, start_yaw_jitter=START_YAW_JITTER,
                vehicle_scale=float(np.exp(car.normal(0.0, MODEL_SCALE_SD))),
            ),
        )
        collisions.append(result.collisions)
        if result.status != "COMPLETE":
            outcomes.append(float("inf"))
            continue
        # The DQ cliff, applied at the distance that is actually judged.
        # `laps` here is a search-economy choice; the referee runs JUDGED_LAPS
        # plus warmup, so a collision RATE that looks survivable over 3 laps
        # can be a disqualification over 12. Extrapolating the rate is an
        # approximation, but treating 3 laps as the whole run is a much worse
        # one: it prices the eleventh collision at +10 s when the rules price
        # it at the entire 100 points.
        projected = result.collisions * (JUDGED_LAPS + WARMUP_LAPS) / laps
        if projected > COLLISION_LIMIT:
            outcomes.append(float("inf"))
            continue
        # Net time per lap: the 10 s penalty spread over the laps driven.
        outcomes.append(result.elapsed_s / laps + 10.0 * result.collisions / laps)

    return outcomes, collisions


def _summarise(outcomes, attempts=3, draws=4000):
    """Expected best-of-`attempts`, plus the numbers that explain it."""
    finished = [o for o in outcomes if np.isfinite(o)]
    rate = len(finished) / len(outcomes)

    rng = np.random.RandomState(0)
    picks = rng.randint(0, len(outcomes), size=(draws, attempts))
    values = np.array(outcomes)[picks].min(axis=1)
    expected = float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("inf")

    # expected_best_of_3 averages only the draws where something finished, so
    # it is silent about the draws where all three runs failed - the one
    # outcome that scores zero. Expected POINTS counts them. The rules score
    # time as a ratio to the leader, so points ~ REF_LAP_S / lap, and zero for
    # a draw with no scoring run. `effective_lap` turns that back into
    # seconds so it reads like a lap time, but it is what candidates are
    # ranked on: it moves with speed AND with the chance of scoring nothing.
    finite = np.isfinite(values)
    points = np.zeros(len(values))
    points[finite] = 100.0 * REF_LAP_S / values[finite]
    expected_points = float(np.mean(points))

    return {
        "rollouts": len(outcomes),
        "completion": rate,
        "expected_best_of_3": expected if rate > 0 else float("inf"),
        "expected_points": expected_points,
        "effective_lap": 100.0 * REF_LAP_S / expected_points if expected_points > 0 else float("inf"),
        "p_all_three_fail": float((1.0 - rate) ** attempts),
        "median": float(np.median(finished)) if finished else None,
        "best": float(min(finished)) if finished else None,
    }


def summarise_events(outcomes, attempts=3):
    """Score outcomes that come in blocks of `attempts` runs of ONE car.

    _summarise resamples every rollout independently, model error included.
    But the judges run one real vehicle three times. Best-of-three rescues a
    run spoiled by NOISE - a bad start, a latency spike - and does nothing at
    all about MODEL error: if the real grip is below what we planned for, it
    is below it for all three runs, and all three fail together.

    Treating them as independent made a config finishing 88% of its runs
    look like it scores nothing only 0.2% of the time (0.12 cubed). With the
    failures correlated it is closer to 12%, and the search was converging on
    exactly that kind of config. So each block here is one event, the best of
    its runs, and an event with no finished run scores zero points.
    """
    arr = np.asarray(outcomes, dtype=float)
    events = arr[: len(arr) // attempts * attempts].reshape(-1, attempts).min(axis=1)
    finite = np.isfinite(events)
    points = np.zeros(len(events))
    points[finite] = 100.0 * REF_LAP_S / events[finite]
    expected_points = float(np.mean(points)) if len(points) else 0.0
    finished = arr[np.isfinite(arr)]
    return {
        "rollouts": len(arr),
        "events": len(events),
        "completion": float(np.mean(np.isfinite(arr))) if len(arr) else 0.0,
        "p_event_zero": float(np.mean(~finite)) if len(events) else 1.0,
        "expected_best_of_3": float(np.mean(events[finite])) if finite.any() else float("inf"),
        "expected_points": expected_points,
        "effective_lap": 100.0 * REF_LAP_S / expected_points if expected_points > 0 else float("inf"),
        "median": float(np.median(finished)) if len(finished) else None,
        "best": float(finished.min()) if len(finished) else None,
    }


def sample(rng):
    params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()}
    params["margin"] = float(MARGINS[rng.randint(len(MARGINS))])
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=os.path.join(REPO, "lab", "shortlist"))
    args = parser.parse_args(argv)
    if args.quick:
        args.candidates, args.rollouts = 24, 3

    vehicle = Vehicle.load(os.path.join(
        REPO, "race_ws", "src", "team_driver", "config", "vehicle.json"))
    print(f"vehicle: a_lat_max {vehicle.a_lat_max:.2f}  a_accel {vehicle.a_accel_max:.2f}  "
          f"a_brake {vehicle.a_brake_max:.2f}  measured={vehicle.measured}")
    if not vehicle.measured:
        print("WARNING: vehicle.json is UNMEASURED. The search will faithfully\n"
              "         optimise for a car that may not exist. Run sysid first.\n")

    rng = np.random.RandomState(args.seed)
    jobs = [(i, sample(rng), args.rollouts) for i in range(args.candidates)]

    workers = max(mp.cpu_count() - 1, 1)
    print(f"{args.candidates} candidates x {args.rollouts} rollouts x {args.laps} laps "
          f"on {workers} workers")
    started = time.perf_counter()
    with mp.Pool(workers, initializer=_init,
                 initargs=(vehicle.to_dict(), args.laps)) as pool:
        rows = []
        for n, row in enumerate(pool.imap_unordered(_evaluate, jobs), 1):
            rows.append(row)
            if n % 25 == 0:
                print(f"  {n}/{args.candidates}  ({time.perf_counter()-started:.0f}s)")

    viable = [r for r in rows if np.isfinite(r["effective_lap"])]
    viable.sort(key=lambda r: r["effective_lap"])
    print(f"\n{len(viable)}/{len(rows)} candidates completed at least one rollout, "
          f"{time.perf_counter()-started:.0f}s total\n")

    head = f"{'rank':>4} {'E[best3]':>9} {'median':>7} {'best':>7} {'compl':>6} {'DQ3':>6}  config"
    print(head)
    for rank, r in enumerate(viable[: args.top], 1):
        cfg = (f"margin={r['margin']:.3f} grip={r['grip']:.2f} vmax={r['v_max']:.1f} "
               f"tau={r['tau_preview']:.2f} kc={r['k_cross']:.1f} kh={r['k_heading']:.1f} "
               f"lift={r['lift_band']:.2f} bg={r['brake_gain']:.2f}")
        print(f"{rank:>4} {r['expected_best_of_3']:>9.3f} {r['median']:>7.3f} "
              f"{r['best']:>7.3f} {r['completion']*100:>5.0f}% "
              f"{r['p_all_three_fail']*100:>5.1f}%  {cfg}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "montecarlo.json"), "w") as handle:
        json.dump(viable, handle, indent=2)
    if viable:
        _write_params(os.path.join(args.out, "best.yaml"), viable[0])
        print(f"\nBest written to {args.out}/best.yaml")
        print(f"Regenerate its racing line at margin {viable[0]['margin']:.2f}, then confirm:")
        print("  ./scripts/evaluate.sh --team mc_best --runs 3 "
              "--driver-params lab/shortlist/best.yaml")
    return 0


def _write_params(path, r):
    with open(path, "w") as handle:
        handle.write(
            "# Monte Carlo best, ranked by expected best-of-three net lap time\n"
            "# under model error, sensor noise, actuation latency and start jitter.\n"
            f"# E[best of 3] {r['expected_best_of_3']:.3f} s/lap, median "
            f"{r['median']:.3f}, completion {r['completion']*100:.0f}%,\n"
            f"# chance all three judged runs fail: "
            f"{r.get('p_event_zero', r.get('p_all_three_fail', float('nan')))*100:.2f}%\n"
            f"# effective lap (expected points as seconds): {r['effective_lap']:.3f}\n"
            "#\n"
            f"# REQUIRES the racing line regenerated at margin {r['margin']:.3f}:\n"
            "#   python3 -m team_driver.tools.build_raceline --map maps/icra26_compete \\\n"
            "#     --centreline maps/icra26_compete_centerline.csv \\\n"
            f"#     --out maps/icra26_compete_raceline.csv --margin {r['margin']:.3f}\n"
            "driver:\n  ros__parameters:\n"
            f"    line:\n      grip_fraction: {r['grip']:.3f}\n      v_max: {r['v_max']:.2f}\n"
            f"    pursuit:\n      tracker: \"feedforward\"\n"
            f"      tau_preview: {r['tau_preview']:.3f}\n"
            f"      min_preview: {r['min_preview']:.3f}\n"
            f"      k_cross: {r['k_cross']:.3f}\n      k_heading: {r['k_heading']:.3f}\n"
            f"      curvature_cap: {r['curvature_cap']:.3f}\n"
            f"    speed:\n      kp: {r['kp']:.4f}\n      ki: {r['ki']:.4f}\n"
            f"      lift_band: {r['lift_band']:.3f}\n      brake_gain: {r['brake_gain']:.3f}\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
