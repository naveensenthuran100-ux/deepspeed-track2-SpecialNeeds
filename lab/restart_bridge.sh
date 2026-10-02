#!/usr/bin/env bash
# Restart the AutoDRIVE bridge in the container and wait for the simulator to
# reconnect. The bridge's CPU climbs over hours of back-to-back runs and the
# simulation stutters with it; a restart restores it. Exit 1 if the simulator
# does not reconnect within 5 minutes.
set -uo pipefail
log="$(cd "$(dirname "$0")/.." && pwd)/results/bridge.log"
docker exec roboracer_track2 bash -lc "pkill -INT -f '[s]imulator.launch.py'; sleep 3; pkill -f '[s]im_bridge'; true"
sleep 2
docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc \
  "cd /hackathon/race_ws && source install/local_setup.bash && nohup ros2 launch roboracer_referee simulator.launch.py simulator:=false rviz:=false > /hackathon/results/bridge.log 2>&1 &"
for i in $(seq 1 150); do
  if grep -q "Connected" "$log" 2>/dev/null; then sleep 3; echo "reconnected after ~$((i*2))s"; exit 0; fi
  sleep 2
done
echo "simulator did NOT reconnect"; exit 1
