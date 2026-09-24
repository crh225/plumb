#!/usr/bin/env bash
# Stop every rented pod (and its teacher) once N teacher questions are kept.   ./pods-stop-at-kept.sh N
# Writes results/pods-stopped so the supervisor stops restarting pod teachers.
cd "$(dirname "$0")"
. ./runpod-env.sh; . ./h100-env.sh
kept() {
  node -e 'const fs=require("fs");let k=0;for(const f of fs.readdirSync("data/teacher"))if(/^(train.*|smoke)\.jsonl$/.test(f))for(const l of fs.readFileSync("data/teacher/"+f,"utf8").split("\n"))if(l&&JSON.parse(l).agree)k++;console.log(k)'
}
until [ "$(kept)" -ge "$1" ]; do sleep 60; done
touch results/pods-stopped
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'teacher\.py.*train-pod' -and (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'uv.exe') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
for v in $(compgen -v | grep -E '^POD[0-9]*_ID$'); do
  curl -s -X POST "https://rest.runpod.io/v1/pods/${!v}/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
done
echo "$(date '+%F %T') reached $1 kept teacher questions ($(kept)): pod teachers and all pods stopped" >> results/supervise.log
