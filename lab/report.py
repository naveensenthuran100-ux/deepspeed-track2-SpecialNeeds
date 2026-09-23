#!/usr/bin/env python3
"""Aggregate real simulator results into the numbers that decide a config.

    python3 lab/report.py                       # everything in results/
    python3 lab/report.py --dir results/submitted
    python3 lab/report.py --surrogate lab/shortlist/sweep.json

The simulator is not deterministic and the judges take the best of three, so
a single run is not evidence. This reports the distribution and, first among
them, the completion rate: a configuration that completes 60% of the time
delivers a scoring run 94% of the time and is worth more than its mean run
suggests.

With --surrogate it also reports how well the surrogate predicted the real
lap times. That correlation is the number that decides how far the surrogate
may be trusted, and it is meant to be looked at, not assumed.
"""

import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "race_ws", "src", "team_driver"))

from team_driver.tools.scoring import summarise_runs   # noqa: E402


def load(directory):
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path) as handle:
                result = json.load(handle)
        except (ValueError, OSError):
            print(f"  skipping unreadable {os.path.basename(path)}")
            continue
        if "team" in result and "status" in result:
            runs[result["team"]].append(result)
    return runs


def _fmt(value, spec="6.2f"):
    return format(value, spec) if value is not None else "     -"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.path.join(REPO, "results"))
    parser.add_argument("--surrogate", default=None,
                        help="lab/shortlist/sweep.json, to check the model")
    args = parser.parse_args(argv)

    runs = load(args.dir)
    if not runs:
        print(f"No result files in {args.dir}.")
        print("Produce some with: ./scripts/evaluate.sh --team <name> --runs 3")
        return 0

    print(f"{'team':<22} {'runs':>4} {'ok':>4} {'compl':>6} {'best3':>6} "
          f"{'lap mean':>9} {'lap sd':>7} {'lap best':>9} {'total':>8} {'coll':>5}")
    for team in sorted(runs):
        s = summarise_runs(runs[team])
        print(f"{team:<22} {s['runs']:>4} {s['scored_runs']:>4} "
              f"{s['completion_rate'] * 100:>5.0f}% "
              f"{s['expected_scoring_run_in_three'] * 100:>5.0f}% "
              f"{_fmt(s['best_lap_mean'], '9.2f')} {_fmt(s['best_lap_sd'], '7.2f')} "
              f"{_fmt(s['best_lap_best'], '9.2f')} {_fmt(s['total_mean'], '8.2f')} "
              f"{_fmt(s['collisions_mean'], '5.1f')}")

    print("\ncompl = fraction of runs that scored.  best3 = chance of at least one")
    print("scoring run in the judges' three attempts.  All times after penalties.")

    if args.surrogate:
        _compare(args.surrogate, runs)
    return 0


def _compare(path, runs):
    """How well did the surrogate predict reality? The honesty check."""
    try:
        with open(path) as handle:
            predictions = json.load(handle)
    except (ValueError, OSError):
        print(f"\nCould not read {path}")
        return

    by_rank = [p for p in predictions if p.get("net_s") is not None]
    by_rank.sort(key=lambda p: p["net_s"])

    print("\nSurrogate vs simulator")
    print(f"{'candidate':<22} {'surrogate':>10} {'simulator':>10} {'error':>8}")
    pairs = []
    for rank, prediction in enumerate(by_rank, 1):
        team = f"candidate_{rank}"
        if team not in runs:
            continue
        actual = summarise_runs(runs[team])["best_lap_best"]
        if actual is None:
            continue
        pairs.append((prediction["lap_s"], actual))
        print(f"{team:<22} {prediction['lap_s']:>10.2f} {actual:>10.2f} "
              f"{actual - prediction['lap_s']:>+8.2f}")

    if len(pairs) < 3:
        print("  Fewer than three paired runs: not enough to judge the model yet.")
        return

    predicted = [p for p, _ in pairs]
    observed = [a for _, a in pairs]
    print(f"  bias {sum(o - p for p, o in pairs) / len(pairs):+.2f} s, "
          f"correlation {_pearson(predicted, observed):.3f}")
    print("  Trust the surrogate only over the range these pairs cover.")


def _pearson(xs, ys):
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    var_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
