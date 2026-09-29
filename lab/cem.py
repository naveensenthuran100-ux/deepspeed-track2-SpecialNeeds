#!/usr/bin/env python3
"""Cross-entropy search: sample a population, keep the best, resample around them.

    python3 lab/cem.py                      # ~30 min on 10 cores
    python3 lab/cem.py --generations 2 --population 16 --cars 2 --final-cars 2   # smoke test

montecarlo.py samples once from the whole box and keeps the best. That maps
the space but spends most of its budget far from anything good. This one
iterates: each generation is drawn from a distribution fitted to the previous
generation's elite, so the budget concentrates where the good configurations
are.

Iterating on a noisy objective has one well-known failure, and this search is
built around avoiding it. Keep the winner of a noisy contest and you have kept
the luckiest draw, not the best car; resample around it and the next
generation inherits the luck, until the search converges confidently on
something broken. montecarlo.py's first winner was exactly that - 11.16 s and
"0% DQ" on its own 8 worlds, 9.62 collisions per run on 40 fresh ones. So:

  * COMMON RANDOM NUMBERS. Every candidate in a generation is run on the same
    worlds. A difference in score is then a difference between candidates,
    not between the worlds they happened to draw.
  * FRESH-WORLD VALIDATION. The generation's front-runners are re-run on
    worlds nobody has been selected on before the elite is chosen, and scored
    on everything they have driven. Luck does not survive a second sample.
  * ELITES RE-DRIVE. Elites carried into the next generation are run again on
    that generation's worlds, so a configuration's record keeps growing and a
    lucky start is diluted rather than preserved.

  * ONE CAR, THREE RUNS. Worlds come in groups of three that share a model
    error and differ in noise, because the judges drive one real vehicle
    three times. Best-of-three forgives noise, never model error - see
    montecarlo.summarise_events.

The objective is expected points - best of each group of three, a group with
no finished run scoring zero - with the DQ cliff, reported as an effective lap
time. The final shortlist is then re-run at the distance actually judged
(10 laps plus 2 warmup), so its collision count is measured, not projected.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montecarlo as MC                                               # noqa: E402
from team_driver.core.vehicle import Vehicle                          # noqa: E402

#: The only racing-line lever worth searching. Smoothing was measured to make
#: the line slower at every setting (it relaxes the line towards the
#: centreline), so the line is the minimum-curvature one at a chosen margin.
MARGINS = [round(0.20 + 0.025 * i, 3) for i in range(13)]            # 0.20 .. 0.50

#: The configuration in config/driver_params.yaml. Seeded into generation 0 so
#: the search is measured against what we ship and can never report a result
#: without having seen it.
SHIPPED = dict(margin=0.35, grip=0.70, v_max=12.0, tau_preview=0.16, min_preview=0.25,
               k_cross=2.2, k_heading=0.7, lift_band=0.15, brake_gain=1.0,
               kp=0.05, ki=0.15, curvature_cap=0.8)

KEYS = list(MC.SPACE)
LO = np.array([MC.SPACE[k][0] for k in KEYS])
HI = np.array([MC.SPACE[k][1] for k in KEYS])


def _worlds(base, cars, runs=3):
    """`cars` model-error draws, each driven `runs` times with its own noise.

    Consecutive blocks of `runs` share a car, which is the layout
    MC.summarise_events expects.
    """
    return [(base + 100 * k, base + 100 * k + 1 + j) for k in range(cars) for j in range(runs)]


def _job(job):
    cid, params, seeds = job
    outcomes, collisions = MC.run_rollouts(params, seeds)
    return cid, outcomes, collisions


class Candidate:
    def __init__(self, cid, params, born):
        self.cid, self.params, self.born = cid, params, born
        self.outcomes, self.collisions = [], []

    def add(self, outcomes, collisions):
        self.outcomes += outcomes
        self.collisions += collisions

    @property
    def score(self):
        return MC.summarise_events(self.outcomes)["effective_lap"] if self.outcomes else np.inf

    def row(self):
        s = MC.summarise_events(self.outcomes)
        return {**self.params, **s, "cid": self.cid, "born": self.born,
                "collisions_mean": float(np.mean(self.collisions))}


def _encode(params):
    return (np.array([params[k] for k in KEYS]) - LO) / (HI - LO)


def _decode(u, margin):
    x = LO + np.clip(u, 0.0, 1.0) * (HI - LO)
    return {**{k: float(v) for k, v in zip(KEYS, x)}, "margin": float(margin)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=72, help="new samples per generation")
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--cars", type=int, default=4,
                        help="common model-error draws per generation, each driven 3 times")
    parser.add_argument("--validate", type=int, default=16, help="front-runners re-run on fresh worlds")
    parser.add_argument("--final-top", type=int, default=5)
    parser.add_argument("--final-cars", type=int, default=10)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=os.path.join(MC.REPO, "lab", "shortlist", "cem"))
    args = parser.parse_args(argv)

    vehicle = Vehicle.load(os.path.join(
        MC.REPO, "race_ws", "src", "team_driver", "config", "vehicle.json"))
    rng = np.random.RandomState(args.seed)
    workers = max(mp.cpu_count() - 1, 1)

    mean = np.full(len(KEYS), 0.5)
    std = np.full(len(KEYS), 0.30)
    margin_p = np.full(len(MARGINS), 1.0 / len(MARGINS))
    alpha, std_floor = 0.7, 0.02

    pool_all, elites, history = {}, [], []
    next_id = 0
    started = time.perf_counter()
    print(f"CEM: {args.generations} gens x ({args.population} new + {args.elite} elite) "
          f"x {args.cars} cars x 3 runs, {args.laps} laps, {workers} workers")

    with mp.Pool(workers, initializer=MC._init,
                 initargs=(vehicle.to_dict(), args.laps, MARGINS)) as pool:

        def drive(cands, seeds):
            jobs = [(c.cid, c.params, seeds) for c in cands]
            for cid, outcomes, collisions in pool.imap_unordered(_job, jobs):
                pool_all[cid].add(outcomes, collisions)

        for gen in range(args.generations):
            fresh = []
            if gen == 0:
                fresh.append(Candidate(next_id, dict(SHIPPED), gen)); next_id += 1
            while len(fresh) < args.population:
                u = mean + std * rng.randn(len(KEYS))
                m = MARGINS[rng.choice(len(MARGINS), p=margin_p)]
                fresh.append(Candidate(next_id, _decode(u, m), gen)); next_id += 1
            for c in fresh:
                pool_all[c.cid] = c

            # Everyone - new samples and carried elites - on the same worlds.
            gen_seeds = _worlds(args.seed * 1000 + 1_000 * gen, args.cars)
            contenders = fresh + elites
            drive(contenders, gen_seeds)

            # Front-runners again, on worlds nobody has been selected on.
            contenders.sort(key=lambda c: c.score)
            front = [c for c in contenders[: args.validate] if np.isfinite(c.score)]
            check_seeds = _worlds(args.seed * 1000 + 500_000 + 1_000 * gen, args.cars)
            drive(front, check_seeds)

            contenders.sort(key=lambda c: c.score)
            elites = [c for c in contenders[: args.elite] if np.isfinite(c.score)]
            if not elites:
                print(f"gen {gen}: nothing finished - widening and retrying")
                std = np.minimum(std * 1.5, 0.5)
                continue

            # Refit the sampling distribution to the elite.
            U = np.array([_encode(c.params) for c in elites])
            mean = alpha * U.mean(axis=0) + (1 - alpha) * mean
            std = np.maximum(alpha * U.std(axis=0) + (1 - alpha) * std, std_floor)
            counts = np.array([sum(c.params["margin"] == m for c in elites) for m in MARGINS],
                              dtype=float)
            margin_p = alpha * (counts + 0.25) / (counts + 0.25).sum() + (1 - alpha) * margin_p

            b = elites[0]
            r = b.row()
            history.append({"gen": gen, "best": r, "mean": mean.tolist(), "std": std.tolist(),
                            "margin_p": margin_p.tolist()})
            print(f"gen {gen}  {time.perf_counter()-started:5.0f}s  "
                  f"best eff {r['effective_lap']:.3f} E[b3] {r['expected_best_of_3']:.3f} (n={r['rollouts']}, "
                  f"compl {r['completion']*100:.0f}%, P0 {r['p_event_zero']*100:.0f}%, coll {r['collisions_mean']:.2f})  "
                  f"margin {b.params['margin']:.3f} grip {b.params['grip']:.3f} "
                  f"| spread {std.mean():.3f}")
            sys.stdout.flush()

    # ---- final: the shortlist at the distance actually judged ---------------
    ranked = sorted((c for c in pool_all.values() if len(c.outcomes) >= 6 * args.cars),
                    key=lambda c: c.score)
    short = ranked[: args.final_top]
    shipped = pool_all[0]
    if shipped not in short:
        short.append(shipped)
    judged = MC.JUDGED_LAPS + MC.WARMUP_LAPS
    print(f"\nfinal: {len(short)} configs x {args.final_cars} fresh cars x 3 runs x {judged} laps")

    finals = {}
    with mp.Pool(workers, initializer=MC._init,
                 initargs=(vehicle.to_dict(), judged, MARGINS)) as pool:
        seeds = _worlds(args.seed * 1000 + 900_000, args.final_cars)
        jobs = [(c.cid, c.params, seeds) for c in short]
        for cid, outcomes, collisions in pool.imap_unordered(_job, jobs):
            finals[cid] = (outcomes, collisions)

    print(f"\n{'config':>8} {'eff lap':>8} {'E[best3]':>9} {'median':>8} {'best':>8} {'compl':>6} "
          f"{'P(zero)':>8} {'coll/run':>9} {'max coll':>9}  search eff")
    results = []
    for c in short:
        outcomes, collisions = finals[c.cid]
        s = MC.summarise_events(outcomes)
        tag = "SHIPPED" if c is shipped else f"#{c.cid}"
        results.append({**c.params, **s, "cid": c.cid, "tag": tag,
                        "collisions_mean": float(np.mean(collisions)),
                        "collisions_max": int(np.max(collisions)),
                        "search_score": c.score, "judged_laps": judged})
    results.sort(key=lambda r: r["effective_lap"])
    for r in results:
        med = r["median"] if r["median"] is not None else float("nan")
        best = r["best"] if r["best"] is not None else float("nan")
        print(f"{r['tag']:>8} {r['effective_lap']:>8.3f} {r['expected_best_of_3']:>9.3f} {med:>8.3f} {best:>8.3f} "
              f"{r['completion']*100:>5.0f}% {r['p_event_zero']*100:>7.1f}% {r['collisions_mean']:>9.2f} "
              f"{r['collisions_max']:>9d}  {r['search_score']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cem.json"), "w") as handle:
        json.dump({"args": vars(args), "history": history, "final": results,
                   "top_search": [c.row() for c in ranked[:20]]}, handle, indent=2)
    winner = next((r for r in results if np.isfinite(r["effective_lap"])), None)
    if winner:
        MC._write_params(os.path.join(args.out, "best.yaml"), winner)
        print(f"\nBest written to {args.out}/best.yaml (margin {winner['margin']:.3f})")
    print(f"total {time.perf_counter()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
