#!/usr/bin/env bash
# Real-simulator test of the 2026-10-06 upgrades, started unattended.
#
#   bash lab/upgrade_test.sh
#
# Waits until the simulator is connected and streaming (someone has clicked
# Connect + Autonomous), then drives six configurations x 3 runs of 12 laps
# (the judged format), interleaved so drifting simulator conditions cannot
# favour one, and prints the summary.
set -uo pipefail
cd "$(dirname "$0")/.."
export RS_BASE=/hackathon/race_ws/src/team_driver/config/driver_params.yaml RS_MAX_GAPS=8
PY=/Users/naveen/Deepspeed/.venv/bin/python

echo "waiting for the simulator (Connect + Autonomous)..."
until grep -q "Connected" results/bridge.log 2>/dev/null; do sleep 5; done
while true; do
  n=$(docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc \
      "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/results/.probe.py" \
      2>&1 | tail -1 | awk '{print $5}' | tr -d ,)
  [ "${n:-0}" -gt 100 ] && break
  sleep 10
done
echo "streaming - starting at $(date +%H:%M)"

BRAKE=vehicle_config=/hackathon/results/vehicle_brake85.json
GRIP=line.grip_fraction_fast=1.5
MPC=pursuit.tracker=mpc
for rep in 1 2 3; do
  bash lab/rs.sh up_A_$rep 12
  bash lab/rs.sh up_B_$rep 12 $BRAKE
  bash lab/rs.sh up_C_$rep 12 $GRIP
  bash lab/rs.sh up_D_$rep 12 $BRAKE $GRIP
  bash lab/rs.sh up_E_$rep 12 $MPC
  bash lab/rs.sh up_F_$rep 12 $MPC $BRAKE $GRIP
done
echo
$PY lab/weave_real.py up_A_1 up_A_2 up_A_3 up_B_1 up_B_2 up_B_3 up_C_1 up_C_2 up_C_3 \
                      up_D_1 up_D_2 up_D_3 up_E_1 up_E_2 up_E_3 up_F_1 up_F_2 up_F_3
echo "UPGRADE_TEST_DONE $(date +%H:%M)"
