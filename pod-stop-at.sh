#!/usr/bin/env bash
# Stop the rented teacher pod at a set time so it can't bill past the run.   ./pod-stop-at.sh HH:MM [pod-id]
cd "$(dirname "$0")"
. ./runpod-env.sh; . ./h100-env.sh
POD_ID="${2:-$POD_ID}"
until [ "$(date +%H%M)" -ge "${1/:/}" ]; do sleep 60; done
curl -s -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
echo "$(date '+%F %T') pod $POD_ID stop requested" >> results/supervise.log
