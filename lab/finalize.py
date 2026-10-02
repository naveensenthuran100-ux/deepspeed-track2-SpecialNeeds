#!/usr/bin/env python3
"""Ship a configuration found on the real simulator.

    python3 lab/finalize.py --cid 17            # a real_cem candidate
    python3 lab/finalize.py --cid 17 --dry-run  # show what would change

Writes, in place:
  * race_ws/src/team_driver/config/driver_params.yaml - each value edited on
    its own line, so the file's comments (its documentation) survive;
  * race_ws/src/team_driver/config/vehicle.json - the throttle map the
    candidate was tuned against (its vehicle_config);
  * maps/icra26_compete_raceline.csv - the candidate's racing line.

The shipped yaml then needs no --driver-params: `ros2 run team_driver driver`
with the installed config drives exactly what was measured.
"""
import argparse
import json
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML = os.path.join(REPO, "race_ws/src/team_driver/config/driver_params.yaml")
VEHICLE = os.path.join(REPO, "race_ws/src/team_driver/config/vehicle.json")
LINE = os.path.join(REPO, "maps/icra26_compete_raceline.csv")
LOG = os.path.join(REPO, "results/realsim/real_cem.jsonl")

FIXED = {"vehicle_config": "/hackathon/results/vehicle_map246.json", "line.v_max": 18.0,
         "line.friction_circle": True, "mode.block_enter_frames": 1000000}


def host(path):
    return path.replace("/hackathon", REPO)


def edit_yaml(text, dotted, value):
    """Replace `leaf: old` under group `group:` (4-space groups, 6-space keys)."""
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif isinstance(value, float):
        # ALWAYS with a decimal point: ROS 2 reads `18` as INTEGER, the driver
        # declares a DOUBLE, and the node refuses to start - a DNF.
        literal = f"{value:.6g}"
        if not any(c in literal for c in ".en"):
            literal += ".0"
    else:
        literal = str(value)
    lines = text.split("\n")
    *groups, leaf = dotted.split(".")
    want = groups[0] if groups else None
    group = None
    for i, line in enumerate(lines):
        g = re.match(r"^    (\w+):\s*$", line)
        if g:
            group = g.group(1)
            continue
        indent = "      " if want else "    "
        m = re.match(rf"^{indent}{re.escape(leaf)}:\s*(.*?)\s*(#.*)?$", line)
        if m and (group == want or (want is None and not line.startswith("      "))):
            old = m.group(1)
            lines[i] = f"{indent}{leaf}: {literal}" + (f"  {m.group(2)}" if m.group(2) else "")
            return "\n".join(lines), old
    raise KeyError(f"{dotted} not found in {YAML}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cid", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    rows = [json.loads(l) for l in open(LOG) if json.loads(l)["cid"] == a.cid]
    if not rows:
        raise SystemExit(f"cid {a.cid} not in {LOG}")
    params = {**FIXED, **rows[0]["params"]}
    scores = [r["score"] for r in rows]
    print(f"c{a.cid}: {len(rows)} real runs, scores {[round(s, 3) for s in scores]}")

    line_src = host(params.pop("line.raceline_csv"))
    vehicle_src = host(params.pop("vehicle_config"))

    text = open(YAML).read()
    for key, value in params.items():
        text, old = edit_yaml(text, key, value)
        if str(old) != str(value):
            print(f"  {key}: {old} -> {value}")
    print(f"  line: {os.path.relpath(line_src, REPO)} -> {os.path.relpath(LINE, REPO)}")
    print(f"  vehicle: {os.path.relpath(vehicle_src, REPO)} -> {os.path.relpath(VEHICLE, REPO)}")
    if a.dry_run:
        return
    open(YAML, "w").write(text)
    shutil.copyfile(line_src, LINE)
    vehicle = json.load(open(vehicle_src))
    json.dump(vehicle, open(VEHICLE, "w"), indent=2, sort_keys=True)
    open(VEHICLE, "a").write("\n")
    print("written.")


if __name__ == "__main__":
    main()
