#!/usr/bin/env bash
# Real-simulator A/B of the corner-exit full-throttle gate, started unattended.
#
#   bash lab/exit_gate_test.sh
#
# speed.full_throttle_max_lat (3.74 m/s^2) was tuned by the real-sim CEM while
# the hairpin-exit collisions dominated (config c83). Measured on the shipped
# config afterwards: 0.7 s a lap is spent > 0.42 m/s under target with the
# full-throttle band suppressed, mostly on the C2/C3 and C1 exits, at a lateral
# load of ~4.1-4.5 m/s^2 and ~0.25 throttle. Since grip_fraction_tight (config
# H) the hairpin exits are clean, so the gate may be holding back for nothing.
#
# Waits until the simulator streams (Connect + Autonomous clicked), then drives
# four gate settings x 2 runs of 12 laps (judged format), interleaved.
set -uo pipefail
cd "$(dirname "$0")/.."
export RS_BASE=/hackathon/race_ws/src/team_driver/config/driver_params.yaml RS_MAX_GAPS=8

echo "waiting for the simulator to stream (Connect + Autonomous)..."
while true; do
  n=$(docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc \
      "cd /hackathon/race_ws && source install/local_setup.bash && timeout 8 python3 /hackathon/results/.probe.py" \
      2>&1 | tail -1 | awk '{print $5}' | tr -d ,)
  [ "${n:-0}" -gt 10 ] && break
  sleep 3
done
echo "streaming - starting at $(date +%H:%M)"

# A run the bench flags INVALID_TIMING (simulator stutter) says nothing about
# the driver: drive it again, up to twice.
run() {
  local tag="$1"; shift
  for attempt in 1 2 3; do
    bash lab/rs.sh "${tag}" 12 "$@"
    while grep -q '"SIM_IGNORING_COMMANDS"' "results/realsim/${tag}.result.json" 2>/dev/null; do
      echo "  ${tag}: simulator ignoring commands (set it to Autonomous) - retrying in 20 s"
      sleep 20
      bash lab/rs.sh "${tag}" 12 "$@"
    done
    grep -q 'TIMING' "results/realsim/${tag}.result.json" 2>/dev/null || return 0
    echo "  ${tag}: stutter, driving it again"
  done
}
for rep in 1 2; do
  run eg_A_$rep
  run eg_B_$rep speed.full_throttle_max_lat=5.0
  run eg_C_$rep speed.full_throttle_max_lat=6.5
  run eg_D_$rep speed.full_throttle_max_lat=12.0
done
echo "EXIT_GATE_TEST_DONE $(date +%H:%M)"
