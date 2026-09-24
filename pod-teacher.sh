#!/usr/bin/env bash
# Teacher on the rented Runpod vLLM pod (h100-env.sh, git-ignored, holds POD_ID and the vLLM key).
#   ./pod-teacher.sh <docs> <parallel> [pod-id] [out-name] [seed]   -> data/teacher/<out-name>.jsonl
# The pod has a 32k context, so the planning pass gets 24k tokens. Same family mix as every teacher,
# probability included (its documents parse now that fractions are accepted).
cd "$(dirname "$0")"
. ./h100-env.sh
pod="${3:-$POD_ID}"; out="${4:-train-pod}"; seed="${5:-21}"
# A pod can specialise: data/teacher/<out-name>.families (one comma-separated line) overrides the mix.
families=temporal_numeric,temporal_numeric,temporal_numeric,long_policy,long_policy,long_policy,multi_hop,multi_hop,judge,judge,ambiguous,ambiguous,probability,probability,trap,adversarial,tradeoff,routing,extraction,rubric
[ -f "data/teacher/$out.families" ] && families=$(tr -d "\r\n " < "data/teacher/$out.families")
# ... and data/teacher/<out-name>.lengths (one line, ";"-separated) its document lengths.
[ -f "data/teacher/$out.lengths" ] && export TEACHER_LENGTHS="$(tr -d "\r\n" < "data/teacher/$out.lengths")"
export PYTHONUTF8=1 TEACHER_URL="https://$pod-8000.proxy.runpod.net" TEACHER_KEY="$H100_KEY" \
  TEACHER_MODEL=qwen3.8-27b TEACHER_HEADERS='{"User-Agent": "jevforge-teacher/0.1"}' TEACHER_AUTHOR_TOKENS=24000
# Wait (up to 30 min) until the pod's vLLM answers: a pod that is still loading, or broken, would
# otherwise fail every document in seconds and burn through the whole run.
for _ in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 15 -H "Authorization: Bearer $H100_KEY" "$TEACHER_URL/v1/models")" = 200 ] && break
  sleep 30
done
exec uv run --python 3.12 --no-project python training/teacher.py --out "data/teacher/$out.jsonl" \
  --docs "${1:-40}" --parallel "${2:-16}" --seed "$seed" --url "$TEACHER_URL" \
  --families "$families" \
  >> "data/teacher/$out.log" 2>&1
