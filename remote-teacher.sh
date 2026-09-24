#!/usr/bin/env bash
# Teacher on a shared remote server (teacher-env.sh, git-ignored, holds the URL and key).
#   ./remote-teacher.sh [parallel]      -> data/teacher/train.jsonl
cd "$(dirname "$0")"
. ./teacher-env.sh
exec uv run --python 3.12 --no-project python training/teacher.py --out data/teacher/train.jsonl \
  --docs 6000 --parallel "${1:-1}" --seed 0 --url "$TEACHER_URL" \
  --families temporal_numeric,temporal_numeric,temporal_numeric,long_policy,long_policy,long_policy,multi_hop,multi_hop,judge,judge,ambiguous,ambiguous,probability,probability,trap,adversarial,tradeoff,routing,extraction,rubric \
  >> data/teacher/train.log 2>&1
