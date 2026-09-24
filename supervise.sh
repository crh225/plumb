#!/usr/bin/env bash
# Run unattended until a deadline: keep the teachers writing, train a round when enough teacher
# questions are in, and one last round when only a round's worth of time is left; push the
# scores; keep collecting until the deadline, then stop every teacher.
#   ./supervise.sh <HH:MM deadline> [kept-target]         progress: results/supervise.log
# ROUND_EVERY=N trains again after N more kept questions (default 300).
# LOCAL_TEACHER=0 leaves this PC's GPU alone between rounds. POD_PARALLEL=N keeps a teacher with
# N streams on each rented Runpod pod named in h100-env.sh (POD_ID, POD2_ID ... POD9_ID).
# REMOTE_TEACHER=0 leaves the shared remote teacher off. results/pods-stopped (written by
# pods-stop-at-kept.sh) stops the pod watchdog once the pods have been shut down.
set -uo pipefail
cd "$(dirname "$0")"
until_hm="$1"; target="${2:-400}"
deadline=$(date -d "today $until_hm" +%s)
local_teacher="${LOCAL_TEACHER:-1}"; pod_parallel="${POD_PARALLEL:-0}"; round_every="${ROUND_EVERY:-300}"
round_cost=$((40 * 60))   # build, train, calibrate, benchmark: v2 took 21 minutes
log=results/supervise.log
families=temporal_numeric,temporal_numeric,temporal_numeric,long_policy,long_policy,long_policy,multi_hop,multi_hop,judge,judge,ambiguous,ambiguous,probability,probability,trap,adversarial,tradeoff,routing,extraction,rubric
. ./h100-env.sh 2>/dev/null
pods=()   # "pod-id out-name seed"
[ -n "${POD_ID:-}" ] && pods+=("$POD_ID train-pod 21")
for n in 2 3 4 5 6 7 8 9; do v="POD${n}_ID"; [ -n "${!v:-}" ] && pods+=("${!v} train-pod$n 2$n"); done

say() { echo "$(date '+%F %T') $*" >> "$log"; }
left() { echo $(( deadline - $(date +%s) )); }
kept() {
  node -e 'const fs=require("fs");let k=0;for(const f of fs.readdirSync("data/teacher"))if(/^(train.*|smoke)\.jsonl$/.test(f))for(const l of fs.readFileSync("data/teacher/"+f,"utf8").split("\n"))if(l&&JSON.parse(l).agree)k++;console.log(k)'
}
# Teachers are native Windows processes, found and stopped by the file they write.
teachers() {  # train|train-local|train-pod|train-pod2 -> count of running python teachers
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'teacher\.py.*--out data/teacher/$1\.jsonl' }).Count" | tr -d '\r'
}
stop_teacher() {  # train|train-local|train-pod|train-pod2|all
  local pat="--out data/teacher/$1\.jsonl"; [ "$1" = all ] && pat="teacher\.py"
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'teacher\.py' -and \$_.CommandLine -match '$pat' -and (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'uv.exe') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
}
start_local() {
  ./serve-teacher.sh 2 32768 >/dev/null
  for _ in $(seq 1 90); do curl -s -m 3 localhost:8090/health | grep -q '"ok"' && break; sleep 5; done
  PYTHONUTF8=1 TEACHER_AUTHOR_THINKING=0 TEACHER_KEY=none TEACHER_MODEL=qwen3.8-27b-local TEACHER_HEADERS='{}' \
    nohup uv run --python 3.12 --no-project python training/teacher.py --out data/teacher/train-local.jsonl \
    --docs 6000 --parallel 2 --seed 11 --url http://localhost:8090 --families "$families" \
    >> data/teacher/train-local.log 2>&1 &
  say "local teacher started"
}
start_remote() {
  nohup ./remote-teacher.sh "${REMOTE_PARALLEL:-4}" >/dev/null 2>&1 &
  say "remote teacher started (${REMOTE_PARALLEL:-4} streams)"
}
start_pod() {  # pod-id out-name seed
  nohup ./pod-teacher.sh 6000 "$pod_parallel" "$1" "$2" "$3" >/dev/null 2>&1 &
  say "pod teacher started on $1 ($pod_parallel streams -> $2)"
}
push() {
  git add cycle.sh supervise.sh "results/jevy-$1" >/dev/null 2>&1
  git commit -q -m "$2" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed: $2"
}
next_round() {
  local r=2; while [ -d "results/jevy-v$r" ]; do r=$((r + 1)); done; echo "v$r"
}

# Keep the PC from idle-sleeping for exactly this run (a closed lid still sleeps). Settings untouched.
powershell -NoProfile -Command "Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'; [W.P]::SetThreadExecutionState(0x80000001) | Out-Null; Start-Sleep -Seconds $(( $(left) + 300 ))" &
awake=$!
say "start: until $until_hm, round at $target kept teacher questions or $((round_cost / 60 + 10)) min before the end ($(kept) now); then every $round_every more; local teacher $local_teacher, pod streams $pod_parallel x ${#pods[@]}"
final_done=0
while [ "$(left)" -gt 0 ]; do
  if [ "${REMOTE_TEACHER:-1}" = 1 ]; then
    [ "$(teachers train)" -ge 1 ] || { say "remote teacher not running; restarting"; start_remote; }
  fi
  if [ "$pod_parallel" -gt 0 ] && [ ! -f results/pods-stopped ]; then
    for p in "${pods[@]}"; do
      set -- $p
      [ "$(teachers "$2")" -ge 1 ] || { say "pod teacher $2 not running; restarting"; start_pod "$1" "$2" "$3"; }
    done
  fi
  if [ "$local_teacher" = 1 ]; then
    docker ps --format '{{.Names}}' | grep -q jevy-teacher && [ "$(teachers train-local)" -ge 1 ] \
      || { say "local teacher not running; restarting"; stop_teacher train-local; start_local; }
  fi
  k=$(kept)
  by_count=$([ "$k" -ge "$target" ] && echo 1 || echo 0)
  by_time=$([ "$final_done" = 0 ] && [ "$(left)" -le $((round_cost + 600)) ] && echo 1 || echo 0)
  if [ "$(left)" -gt "$round_cost" ] && { [ "$by_count" = 1 ] || [ "$by_time" = 1 ]; }; then
    name=$(next_round)
    say "round $name: training on $k kept teacher questions"
    if ./cycle.sh "$name" 2 2e-5 >> "$log" 2>&1; then
      hard=$(grep -E "jevy-$name +hard" "results/jevy-$name/SCORES.txt" | awk '{print "hard acc " $4 ", ECE " $5 " (JevK5 0.739, 0.077)"}')
      push "$name" "$name: $k teacher questions; $hard"
    else
      say "round $name failed; see results/$name-train.log"
    fi
    target=$((k + round_every))
    [ "$(left)" -le $((round_cost + 600)) ] && final_done=1
    [ "$local_teacher" = 1 ] && start_local
  fi
  s=$(left); [ "$s" -gt 300 ] && s=300; [ "$s" -gt 0 ] && sleep "$s"
done

stop_teacher all
docker stop jevy-teacher >/dev/null 2>&1
say "deadline reached: teachers stopped ($(kept) kept teacher questions in total)"
kill "$awake" 2>/dev/null
