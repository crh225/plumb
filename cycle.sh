#!/usr/bin/env bash
# One training cycle: pause the local teacher, build a gate from all teacher data so far, train
# from JevK5, calibrate and score on the public items. Restart the teacher afterwards.
#   ./cycle.sh <name> [epochs] [lr]        e.g. ./cycle.sh v2 2 2e-5
set -euo pipefail
cd "$(dirname "$0")"
name="$1"; epochs="${2:-2}"; lr="${3:-2e-5}"

echo "== pausing the local teacher"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'teacher\.py' -and \$_.CommandLine -match 'localhost:8090' -and (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'uv.exe') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
docker stop jevy-teacher >/dev/null 2>&1 || true

kept=$(node -e 'const fs=require("fs");let k=0;for(const f of fs.readdirSync("data/teacher"))if(/^(train.*|smoke)\.jsonl$/.test(f))for(const l of fs.readFileSync("data/teacher/"+f,"utf8").split("\n"))if(l&&JSON.parse(l).agree)k++;console.log(k)')
echo "== building data/gate-$name from $kept kept teacher questions (+ as much plain replay)"
./box.sh python training/build_gate.py --out "data/gate-$name" \
  --teacher data/teacher/train*.jsonl data/teacher/smoke.jsonl \
  --teacher-test data/teacher/test.jsonl data/teacher/test-local.jsonl \
  --replay "data/decisions/train.jsonl:$kept" \
  --dev data/decisions/dev.jsonl:600 --dev data/decisions/hard_dev.jsonl:all

echo "== overlap scan"
./box.sh python training/scan_overlap.py "data/gate-$name/train.jsonl" --jevbench /jevbench/datasets/public --drop

# round.conf (optional, read fresh every round) can set EPOCHS, LR, RANK, KEEP_BEST=1, MAXLEN, and MINE=1 for hard mining
# (UNSURE, KEEP_EASY tune it); see training/mine_hard.py.
[ -f round.conf ] && . ./round.conf
epochs="${EPOCHS:-$epochs}"; lr="${LR:-$lr}"
if [ "${MINE:-0}" = 1 ]; then
  echo "== hard mining"
  ./box.sh python training/mine_hard.py "data/gate-$name" --unsure "${UNSURE:-0.8}" --keep-easy "${KEEP_EASY:-0.25}" --max-len "${MAXLEN:-4096}" 2>&1     | grep -E "^(mined|hard ones|train )" || echo "hard mining failed; training on everything instead"
fi

echo "== training $name from JevK5 ($epochs epochs, lr $lr)"
./box.sh python training/lora.py "data/gate-$name" "models/jevy-$name" --base alibiserikbay/JevK5 \
  --epochs "$epochs" --lr "$lr" --warmup 20 --max-len "${MAXLEN:-4096}" --token-budget 8192 --eval-before \
  --rank "${RANK:-16}" ${KEEP_BEST:+--keep-best} \
  > "results/$name-train.log" 2>&1
tr '\r' '\n' < "results/$name-train.log" | grep -E "^(before|after|  [a-z_]+ +acc)" | cut -c1-120 || true

echo "== calibrating"
./box.sh python training/fit_temperature.py "models/jevy-$name" "data/gate-$name/dev.jsonl" --min-fit 100 2>&1 \
  | grep -E "^temperature|^  " || true

echo "== public benchmark"
./run-bench.sh "/work/models/jevy-$name" "jevy-$name" > "results/jevy-$name.log" 2>&1
uv run --python 3.12 --no-project python training/score.py results/jevk5-bf16 "results/jevy-$name" \
  | tee "results/jevy-$name/SCORES.txt"

echo "== done; restart the local teacher with ./serve-teacher.sh 2 32768"
