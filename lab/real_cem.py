#!/usr/bin/env python3
"""Cross-entropy search run directly on the REAL AutoDRIVE simulator.

    python3 lab/real_cem.py --hours 7

Every candidate is driven on the simulator through lab/rs.sh (one persistent
bridge, lab/realsim.py as the bench), so there is no model to be wrong. The
price is time - about a minute per candidate - so the search is small and
careful rather than wide:

  * each candidate drives 1 standing + `--laps - 1` flying laps;
  * score = mean flying lap + 10 s per collision spread over the flying laps,
    and a projected disqualification (collision RATE over the 12 judged laps
    above 10) or any non-COMPLETE status scores as a failure;
  * a run flagged INVALID_TIMING (the simulator stuttered) is driven again,
    never scored - it says nothing about the driver;
  * elites are re-driven every generation and scored on everything they have
    driven, so one lucky run cannot carry a configuration;
  * the distribution is seeded around the best configuration found by hand.

Writes results/realsim/real_cem.jsonl (every run) and
lab/shortlist/real_best.json (best so far, rewritten as it improves).
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "results", "realsim", "real_cem.jsonl")
BEST = os.path.join(REPO, "lab", "shortlist", "real_best.json")

FIXED = {
    "vehicle_config": "/hackathon/results/vehicle_map246.json",
    "line.v_max": 18.0,
    "line.friction_circle": True,
    "mode.block_enter_frames": 1000000,      # no LiDAR "blocked path" on a solo time trial
}

#: name -> (low, high, seed). Seeds are the best hand-found configuration.
SPACE = {
    "line.grip_fraction":       (0.85, 1.50, 1.10),
    "speed.preview_s":          (0.00, 0.30, 0.10),
    "speed.full_throttle_band": (0.05, 1.20, 0.30),
    "speed.brake_gain":         (0.60, 3.00, 1.43),
    "speed.lift_band":          (0.02, 0.50, 0.20),
    "speed.kp":                 (0.02, 0.25, 0.104),
    "speed.ki":                 (0.05, 0.60, 0.324),
    "pursuit.tau_preview":      (0.06, 0.40, 0.189),
    "pursuit.min_preview":      (0.15, 0.90, 0.525),
    "pursuit.k_heading":        (0.30, 2.00, 0.842),
    "pursuit.k_cross":          (0.50, 4.00, 1.958),
    "pursuit.curvature_cap":    (0.50, 1.40, 1.011),
    # Full throttle only below this lateral load (m/s^2). 12 is effectively
    # off for every candidate driven before it existed.
    "speed.full_throttle_max_lat": (2.0, 12.0, 12.0),
}
LINES = ["/hackathon/lab/lines/mt_k1.2.csv",            # best clean real run: 11.75 mean
         "/hackathon/lab/shortlist/cem/raceline_m0300.csv",
         "/hackathon/lab/lines/mt_k1.3.csv",
         # More room at the two corner exits where EVERY collision happened
         # (s=32.5 and s=37.3, car ~0.3 m wide on exit): 0.35/0.30 m margin.
         "/hackathon/lab/lines/mt_k1.2_m0.35.csv",
         "/hackathon/lab/lines/mt_k1.3_m0.30.csv"]
FAIL = 30.0          # s/lap charged to a failed or disqualified run


def drive(tag, params, laps):
    args = ["bash", os.path.join(REPO, "lab", "rs.sh"), tag, str(laps)]
    args += [f"{k}={v}" for k, v in {**FIXED, **params}.items()]
    for attempt in range(3):
        subprocess.run(args, capture_output=True, text=True)
        path = os.path.join(REPO, "results", "realsim", f"{tag}.result.json")
        if not os.path.exists(path):
            continue
        result = json.load(open(path))
        if result["status"] == "INVALID_TIMING" or result["status"].endswith("BAD_TIMING") \
                or result["status"] in ("SIM_IGNORING_COMMANDS", "DRIVER_DID_NOT_START"):
            time.sleep(5)
            continue
        return result
    return result if os.path.exists(path) else {"status": "NO_RESULT", "lap_times": []}


def score_run(result, laps):
    flying = result.get("lap_times", [])[1:]
    coll = result.get("collisions") or 0
    if result.get("status") != "COMPLETE" or len(flying) < laps - 1:
        return FAIL
    # Every collision counts towards the DQ limit (rule 16), projected to the
    # 12 laps actually driven; only those in timed laps cost time (rule 18).
    if coll * 12.0 / laps > 10.0:
        return FAIL
    timed = sum(result.get("lap_collisions", [])[1:])
    return float(np.mean(flying)) + 10.0 * timed / len(flying)


class Cand:
    def __init__(self, cid, params):
        self.cid, self.params, self.scores = cid, params, []

    @property
    def score(self):
        # Mean, not min: the judged run is ten laps, not one lucky one.
        return float(np.mean(self.scores)) if self.scores else float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=7.0)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--laps", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-gaps", type=int, default=3,
                   help="odometry gaps >100 ms a run may have and still count")
    p.add_argument("--restart-bridge-every", type=int, default=0,
                   help="restart the bridge after this many runs (0 = never); it slows over hours")
    p.add_argument("--resume", action="store_true",
                   help="replay results/realsim/real_cem.jsonl and carry on from its elites")
    a = p.parse_args()
    os.environ["RS_MAX_GAPS"] = str(a.max_gaps)
    runs_since_restart = 0

    rng = np.random.RandomState(a.seed)
    keys = list(SPACE)
    lo = np.array([SPACE[k][0] for k in keys]); hi = np.array([SPACE[k][1] for k in keys])
    mean = (np.array([SPACE[k][2] for k in keys]) - lo) / (hi - lo)
    std = np.full(len(keys), 0.12)
    line_p = np.full(len(LINES), 1.0 / len(LINES))
    deadline = time.time() + a.hours * 3600
    elites, cid, gen, best_ever = [], 0, 0, None
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    def decode(u, line):
        x = lo + np.clip(u, 0, 1) * (hi - lo)
        out = {k: round(float(v), 4) for k, v in zip(keys, x)}
        out["line.raceline_csv"] = line
        return out

    if a.resume and os.path.exists(LOG):
        # Every candidate driven so far, with every score it earned.
        seen = {}
        for row in map(json.loads, open(LOG)):
            c = seen.setdefault(row["cid"], Cand(row["cid"], row["params"]))
            c.scores.append(row["score"])
            gen = max(gen, row["gen"] + 1)
        cid = max(seen) + 1
        elites = [c for c in sorted(seen.values(), key=lambda c: c.score)[: a.elite]
                  if c.score < FAIL]
        if elites:
            U = np.array([(np.array([c.params.get(k, SPACE[k][2]) for k in keys]) - lo) / (hi - lo)
                          for c in elites])
            mean = U.mean(axis=0)
            std = np.maximum(U.std(axis=0), 0.05)
            counts = np.array([sum(c.params["line.raceline_csv"] == l for c in elites) for l in LINES], float)
            line_p = 0.7 * (counts + 0.3) / (counts + 0.3).sum() + 0.3 * line_p
        print(f"resumed: {len(seen)} candidates, generation {gen}, elites "
              + ", ".join(f"c{c.cid} {c.score:.3f} (n={len(c.scores)})" for c in elites), flush=True)

    while time.time() < deadline:
        fresh = []
        if gen == 0:
            fresh.append(Cand(cid, decode(mean, LINES[0]))); cid += 1
            fresh.append(Cand(cid, decode(mean, LINES[1]))); cid += 1
        while len(fresh) < a.population - len(elites):
            line = LINES[rng.choice(len(LINES), p=line_p)]
            fresh.append(Cand(cid, decode(mean + std * rng.randn(len(keys)), line))); cid += 1
        for c in fresh + elites:
            if time.time() > deadline:
                break
            if a.restart_bridge_every and runs_since_restart >= a.restart_bridge_every:
                out = subprocess.run(["bash", os.path.join(REPO, "lab", "restart_bridge.sh")],
                                     capture_output=True, text=True)
                print(f"[{time.strftime('%H:%M')}] bridge restart: {out.stdout.strip().splitlines()[-1:]}",
                      flush=True)
                runs_since_restart = 0
            runs_since_restart += 1
            tag = f"rc_g{gen}_c{c.cid}_{len(c.scores)}"
            result = drive(tag, c.params, a.laps)
            s = score_run(result, a.laps)
            c.scores.append(s)
            with open(LOG, "a") as handle:
                handle.write(json.dumps({"gen": gen, "cid": c.cid, "tag": tag, "score": s,
                                         "status": result.get("status"),
                                         "laps": result.get("lap_times"),
                                         "collisions": result.get("collisions"),
                                         "params": c.params, "time": time.strftime("%H:%M:%S")}) + "\n")
            print(f"[{time.strftime('%H:%M')}] gen {gen} c{c.cid} n={len(c.scores)} "
                  f"{result.get('status'):<20} score {s:6.3f} mean {c.score:6.3f} laps {result.get('lap_times')}",
                  flush=True)
        pool = sorted(fresh + elites, key=lambda c: c.score)
        elites = [c for c in pool[: a.elite] if c.score < FAIL]
        if elites:
            U = np.array([(np.array([c.params.get(k, SPACE[k][2]) for k in keys]) - lo) / (hi - lo)
                          for c in elites])
            mean = 0.6 * U.mean(axis=0) + 0.4 * mean
            std = np.maximum(0.6 * U.std(axis=0) + 0.4 * std, 0.04)
            counts = np.array([sum(c.params["line.raceline_csv"] == l for c in elites) for l in LINES], float)
            line_p = 0.6 * (counts + 0.3) / (counts + 0.3).sum() + 0.4 * line_p
            top = elites[0]
            if best_ever is None or top.score < best_ever["score"] or top.cid == best_ever["cid"]:
                best_ever = {"cid": top.cid, "score": top.score, "runs": len(top.scores),
                             "scores": top.scores, "params": {**FIXED, **top.params},
                             "time": time.strftime("%H:%M:%S")}
                json.dump(best_ever, open(BEST, "w"), indent=2)
            print(f"== gen {gen} done: best c{top.cid} mean {top.score:.3f} over {len(top.scores)} runs; "
                  f"spread {std.mean():.3f}", flush=True)
        else:
            std = np.minimum(std * 1.3, 0.3)
            print(f"== gen {gen}: no candidate finished cleanly; widening", flush=True)
        gen += 1


if __name__ == "__main__":
    main()
