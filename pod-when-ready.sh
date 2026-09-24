#!/usr/bin/env bash
# Wait for a pod's vLLM to answer, then start a teacher on it.
#   ./pod-when-ready.sh <pod-id> <out-name> <seed> <parallel>
cd "$(dirname "$0")"
. ./h100-env.sh
for _ in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 15 -H "Authorization: Bearer $H100_KEY" "https://$1-8000.proxy.runpod.net/v1/models")" = 200 ] && break
  sleep 30
done
echo "$(date '+%F %T') pod $1 ready; teacher started ($4 streams -> $2)" >> results/supervise.log
exec ./pod-teacher.sh 6000 "$4" "$1" "$2" "$3"
