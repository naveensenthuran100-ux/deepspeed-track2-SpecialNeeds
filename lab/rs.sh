#!/usr/bin/env bash
# One real-simulator evaluation through lab/realsim.py, from the host.
#
#   lab/rs.sh <tag> <laps> [KEY=VALUE ...]
#
# Base params: $RS_BASE (container path), default #480's full yaml. Prints a
# one-line summary; the full result and telemetry land in results/realsim/.
set -euo pipefail
tag="$1"; laps="$2"; shift 2
base="${RS_BASE:-/hackathon/lab/shortlist/cem/driver_params_480.yaml}"
sets=()
for kv in "$@"; do sets+=(--set "$kv"); done
# Keep the simulator frontmost: macOS throttles a background Unity window and
# the odometry then arrives in 100+ ms bursts (see realsim.py, timing quality).
sim_pid="$(pgrep -f 'AutoDRIVE Simulator.app/Contents/MacOS' | head -1 || true)"
if [ -n "$sim_pid" ]; then
  osascript -e "tell application \"System Events\" to set frontmost of (first process whose unix id is $sim_pid) to true" >/dev/null 2>&1 || true
fi
docker exec roboracer_track2 /usr/local/bin/entrypoint.sh bash -lc \
  "cd /hackathon/race_ws && source install/local_setup.bash && python3 /hackathon/lab/realsim.py --params $base --laps $laps --tag $tag $(printf '%q ' "${sets[@]}")" \
  > /dev/null 2>&1 || true
f="$(cd "$(dirname "$0")/.." && pwd)/results/realsim/$tag.result.json"
if [ -f "$f" ]; then
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['tag']:<22} {d['status']:<20} laps {d['lap_times']}  coll {d.get('collisions')}  best {d.get('best')}  mean {d.get('mean_flying')}  odom {d.get('odom_median_ms')}/{d.get('odom_p95_ms')}ms gaps {d.get('odom_gaps_100ms')}\")" "$f"
else
  echo "$tag: NO RESULT"
fi
