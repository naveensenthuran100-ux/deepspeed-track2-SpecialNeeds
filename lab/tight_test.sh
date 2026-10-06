#!/usr/bin/env bash
# Follow-up: does lower grip in the tightest corners stop the corner-3 crashes?
# 73% of all collisions in the 18-run upgrade test were at s 34-39.
set -uo pipefail
cd "$(dirname "$0")/.."
export RS_BASE=/hackathon/race_ws/src/team_driver/config/driver_params.yaml RS_MAX_GAPS=8
BRAKE=vehicle_config=/hackathon/results/vehicle_brake85.json
for rep in 1 2 3; do
  bash lab/rs.sh tt_B_$rep 12 $BRAKE
  bash lab/rs.sh tt_G_$rep 12 $BRAKE line.grip_fraction_tight=0.95 line.tight_radius=1.0
  bash lab/rs.sh tt_H_$rep 12 $BRAKE line.grip_fraction_tight=0.90 line.tight_radius=1.0
done
/Users/naveen/Deepspeed/.venv/bin/python lab/weave_real.py tt_B_1 tt_B_2 tt_B_3 tt_G_1 tt_G_2 tt_G_3 tt_H_1 tt_H_2 tt_H_3
echo "TIGHT_TEST_DONE $(date +%H:%M)"
