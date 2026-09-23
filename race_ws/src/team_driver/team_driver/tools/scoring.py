"""Scoring, exactly as docs/06-rules.md defines it.

Every sweep ranks against this rather than against raw lap time. The reason
is that the penalty already prices contact correctly: a collision is 10 s on
the lap it happened on, and both halves of the score are ratios against the
best team, so minimising expected net time balances speed against collisions
without a separate hand-chosen safety weight.

Around leader pace (about 20 s a lap, 220 s for ten), the marginal rates are
worth keeping in mind: a second a lap is roughly 4.8 points, one collision is
roughly 2.3 - about the same as being half a second a lap slower everywhere -
and the eleventh collision is the whole 100, because a disqualified run
scores zero for both halves.
"""

import math
from typing import Dict, List, Optional

COLLISION_PENALTY_S = 10.0
#: The judges run each submission three times and take the best attempt
#: (rule 23), so a configuration only has to work once.
JUDGED_ATTEMPTS = 3


def net_lap_time(lap_s: float, collisions: int,
                 penalty_s: float = COLLISION_PENALTY_S) -> float:
    """Lap time after penalties. This is what is scored, not the raw time."""
    return float(lap_s) + float(collisions) * float(penalty_s)


def leaderboard_points(
    your_best_lap: Optional[float],
    field_best_lap: Optional[float],
    your_total: Optional[float],
    field_best_total: Optional[float],
) -> float:
    """Score out of 100. A run with no scored time gets nothing, as rule 22 says."""
    if not your_best_lap or not your_total:
        return 0.0
    if not field_best_lap or not field_best_total:
        return 0.0
    return (
        50.0 * (float(field_best_lap) / float(your_best_lap))
        + 50.0 * (float(field_best_total) / float(your_total))
    )


def summarise_runs(runs: List[Dict]) -> Dict:
    """Aggregate a set of result files into the numbers that decide a config.

    `completion_rate` is a first-class output, not a footnote: the judges take
    the best of three, so a configuration that completes 60% of the time
    delivers a scoring run 94% of the time and is worth more than its average
    run suggests.
    """
    scored = [r for r in runs if r.get("scored") and r.get("best_lap_time")]
    laps = [float(r["best_lap_time"]) for r in scored]
    totals = [float(r["total_time"]) for r in scored if r.get("total_time")]
    collisions = [float(r.get("collisions", 0)) for r in runs]

    rate = (len(scored) / len(runs)) if runs else 0.0

    return {
        "runs": len(runs),
        "scored_runs": len(scored),
        "completion_rate": rate,
        "expected_scoring_run_in_three": 1.0 - (1.0 - rate) ** JUDGED_ATTEMPTS,
        "best_lap_mean": _mean(laps),
        "best_lap_sd": _sd(laps),
        "best_lap_best": min(laps) if laps else None,
        "total_mean": _mean(totals),
        "total_sd": _sd(totals),
        "total_best": min(totals) if totals else None,
        "collisions_mean": _mean(collisions),
    }


def _mean(values) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _sd(values) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    average = sum(values) / len(values)
    return float(math.sqrt(sum((v - average) ** 2 for v in values) / (len(values) - 1)))
