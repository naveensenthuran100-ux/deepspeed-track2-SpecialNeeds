# lab/ — the experiment harness

Development tooling. Not part of the submission's driving code, and not in
any protected directory. The driver itself is `race_ws/src/team_driver/`.

## Why this exists

A real 12-lap run on AutoDRIVE costs about six minutes of wall clock, is
locked to real time, and is not perfectly repeatable. On one laptop that caps
honest experimentation at roughly 25 configurations a day — nowhere near
enough to search by brute force.

So the search happens against a surrogate simulator
(`team_driver/tools/surrogate.py`), which runs a lap in about a quarter of a
second by sphere-tracing the LiDAR against a distance transform of the map.
Only the shortlist ever touches the real thing.

**Nothing here is a result.** The surrogate is a model and it can lie. Every
configuration is a hypothesis until three full runs on the real simulator
agree with it.

## Workflow

```bash
# 1. Search. ~50 configurations in about a minute.
python3 lab/sweep.py --laps 2 --top 8

# 2. Confirm the shortlist on the real simulator, three runs each.
./scripts/evaluate.sh --team candidate_1 --runs 3 \
    --driver-params lab/shortlist/candidate_1.yaml

# 3. Read the distribution, not a single run.
python3 lab/report.py --surrogate lab/shortlist/sweep.json
```

Step 3 with `--surrogate` reports how well the model predicted reality. That
correlation is what decides how far the surrogate may be trusted, and it is
meant to be looked at rather than assumed. If it drifts, fix the model before
running another sweep.

## What is ranked

Net lap time — raw time plus 10 s per collision — because that is what the
organisers score, and the penalty already prices contact correctly. Around
leader pace a second a lap is about 4.8 points, one collision about 2.3, and
the eleventh collision is the whole 100.

`report.py` leads with **completion rate** rather than lap time. The judges
take the best of three attempts, so a configuration that completes 60% of the
time delivers a scoring run 94% of the time and is worth more than its mean
run suggests. A fast configuration that fails two runs in three is still a
reasonable bet; one that fails all three is a zero.

## Before you believe any number here

`config/vehicle.json` currently has `measured: false`. Its grip and braking
limits are derived from the published vehicle specification, not measured
from the car. Every lap time the surrogate produces is relative to those
assumptions. Run system identification first:

```bash
ros2 run team_driver sysid_drive          # inside the container, ~2 minutes
python3 -m team_driver.tools.sysid --log results/sysid.csv \
    --out race_ws/src/team_driver/config/vehicle.json
```

## Files

| File | What it does |
|---|---|
| `sweep.py` | Runs a configuration grid against the surrogate, ranks by net lap time, writes `shortlist/candidate_N.yaml` ready for `evaluate.sh --driver-params` |
| `report.py` | Aggregates `results/*.json` into mean/sd/completion rate per team, and compares against surrogate predictions |
| `shortlist/` | Generated. Candidate parameter files plus `sweep.json` |
