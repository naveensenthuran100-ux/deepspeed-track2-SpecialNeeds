#!/usr/bin/env bash
#
# Bring up the whole stack on macOS (Apple Silicon or Intel), start to finish.
#
#   ./lab/run_on_mac.sh              # bridge + simulator + driver
#   ./lab/run_on_mac.sh --evaluate   # ...then a scored 3-lap run
#
# On macOS the Linux simulator binary cannot run, so the native .app runs on
# the host and the devkit runs in the container. They meet on port 4567, which
# docker-compose publishes.
#
# TWO THINGS THAT WILL CATCH YOU OUT
#
# 1. scripts/run_simulator.sh does not work on macOS. Line 89 is
#       exec "${prefix[@]}" "${args[@]}"
#    under `set -u`, and expanding an empty array that way is fatal on bash
#    3.2 - which is the bash macOS ships. It dies with "prefix[@]: unbound
#    variable" before launching anything. scripts/ is protected (rule 30), so
#    this launches the binary directly with the same arguments instead.
#    Worth reporting to the organisers: rule 49 credits a genuine bug report.
#
# 2. THE SIMULATOR BOOTS IN MANUAL MODE AND STREAMS NOTHING UNTIL YOU SET IT
#    TO AUTONOMOUS. You have to click that yourself, in the Menu Panel. Until
#    you do, you get an ESTABLISHED socket, the devkit prints "Connected!",
#    every ROS topic exists, and not one message is ever published on any of
#    them. It looks exactly like a broken bridge and is not.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONTAINER=roboracer_track2
SIM="$PWD/simulator/autodrive_simulator/AutoDRIVE Simulator.app/Contents/MacOS/AutoDRIVE Simulator"
EVALUATE=0
[ "${1:-}" = "--evaluate" ] && EVALUATE=1

in_container() { docker exec "$CONTAINER" /usr/local/bin/entrypoint.sh bash -lc "$1"; }

echo "==> container"
docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" \
  || { ./install/macos/run.sh >/dev/null 2>&1 & sleep 1; }
until docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; do sleep 2; done

echo "==> workspace"
in_container "cd /hackathon/race_ws && colcon build --symlink-install 2>&1 | tail -3"

echo "==> bridge (must start before the simulator: it is the SERVER)"
in_container "pkill -f sim_bridge 2>/dev/null; true"
in_container "cd /hackathon/race_ws && source install/local_setup.bash && \
  nohup ros2 launch roboracer_referee simulator.launch.py simulator:=false rviz:=false \
  > /hackathon/results/bridge.log 2>&1 &"
until docker exec "$CONTAINER" bash -lc "netstat -tln 2>/dev/null | grep -q 4567"; do sleep 2; done
echo "    listening on 4567"

echo "==> simulator (native, on your Mac)"
pkill -f "AutoDRIVE Simulator" 2>/dev/null || true
sleep 1
nohup "$SIM" -ip 127.0.0.1 -port 4567 > /tmp/autodrive_sim.log 2>&1 &
echo "    launched; Unity takes ~2-3 min to compile shaders on a first run"

echo "==> driver"
in_container "pkill -f 'team_driver driver' 2>/dev/null; true"
in_container "cd /hackathon/race_ws && source install/local_setup.bash && \
  nohup ros2 run team_driver driver --ros-args \
  --params-file /hackathon/race_ws/src/team_driver/config/driver_params.yaml \
  > /hackathon/results/driver.log 2>&1 &"

cat <<'MSG'

  ------------------------------------------------------------------
  NOW GO TO THE SIMULATOR WINDOW AND SET THE VEHICLE TO **AUTONOMOUS**
  (Menu Panel). Nothing publishes until you do.
  ------------------------------------------------------------------

  Check it took:    tail -f results/driver.log
  Bridge:           tail -f results/bridge.log
MSG

if [ "$EVALUATE" = "1" ]; then
  echo
  echo "==> waiting for telemetry before scoring..."
  until docker exec "$CONTAINER" bash -lc \
    "grep -q 'LIDAR Range Array' /hackathon/results/bridge.log 2>/dev/null" \
    || in_container "timeout 3 ros2 topic echo /autodrive/roboracer_1/lidar --once >/dev/null 2>&1"; do
    sleep 5
  done
  echo "==> scoring 3 laps"
  ./scripts/evaluate.sh --team first_run --laps 3 --no-simulator
fi
