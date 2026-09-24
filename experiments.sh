#!/usr/bin/env bash
# Recipe experiments, one after another and unattended: each is a cycle.sh round with its own
# round.conf, all on the same teacher data. Models are compared on the held-out check only.
#   ./experiments.sh        progress: results/supervise.log (and the dashboard)   summary: results/experiments.txt
cd "$(dirname "$0")"
log=results/supervise.log
say() { echo "$(date '+%F %T') $*" >> "$log"; }
cp round.conf round.conf.before-experiments

# Keep the PC from idle-sleeping for up to 12 hours while this runs (a closed lid still sleeps).
powershell -NoProfile -Command "Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'; [W.P]::SetThreadExecutionState(0x80000001) | Out-Null; Start-Sleep -Seconds 43200" &
awake=$!
say "start: until $(date -d '+7 hours' +%H:%M), round at 99999 kept teacher questions; recipe experiments v6a-v6c on the v5 data"

run() {  # name, description, round.conf lines...
  local name="$1" what="$2"; shift 2
  printf '%s\n' "# $name: $what" "$@" > round.conf
  say "round $name: training on 5026 kept teacher questions ($what)"
  if ./cycle.sh "$name" 3 3e-5 >> "$log" 2>&1; then
    uv run --python 3.12 --no-project python bench/stats.py "results/jevy-$name" > /dev/null 2>&1
    say "done $name: $(grep -E "jevy-$name +hard" results/jevy-$name/SCORES.txt | awk '{print "hard acc " $4 ", ECE " $5}')"
  else
    say "round $name failed; see results/$name-train.log"
  fi
}

base=(MINE=1 EPOCHS=3 LR=3e-5 MAXLEN=6144 KEEP_BEST=1)
run v6a "v5 recipe, keep the best of 3 passes" "${base[@]}"
run v6b "keep best + LoRA rank 32" "${base[@]}" RANK=32
run v6c "keep best + only wrong/unsure questions (no easy ones)" "${base[@]}" KEEP_EASY=0

cp round.conf.before-experiments round.conf && rm -f round.conf.before-experiments
{
  echo "Recipe experiments, $(date '+%F %T'). Pick by the held-out check (after), not the public items."
  echo
  printf '%-5s %-9s %-7s %-6s %-13s %-9s %-9s %-9s\n' run held-out kept T "hard (pub)" hard-ECE std easy
  for name in v5 v6a v6b v6c; do
    tl="results/$name-train.log"; [ -f "$tl" ] || continue
    after=$(tr '\r' '\n' < "$tl" | grep -m1 '^after' | sed 's/.*"acc": \([0-9.]*\).*/\1/')
    kept=$(tr '\r' '\n' < "$tl" | grep -m1 '^keeping pass' | awk '{print $3}')
    temp=$(node -e 'try{console.log(require("./models/jevy-'"$name"'/jevk5_config.json").temperature)}catch{console.log("-")}')
    sc="results/jevy-$name/SCORES.txt"
    hard=$(grep -E "jevy-$name +hard" "$sc" 2>/dev/null | awk '{print $4}'); hece=$(grep -E "jevy-$name +hard" "$sc" 2>/dev/null | awk '{print $5}')
    std=$(grep -E "jevy-$name +standard" "$sc" 2>/dev/null | awk '{print $4}'); easy=$(grep -E "jevy-$name +easy" "$sc" 2>/dev/null | awk '{print $4}')
    printf '%-5s %-9s %-7s %-6s %-13s %-9s %-9s %-9s\n' "$name" "${after:--}" "${kept:-3}" "$temp" "${hard:--}" "${hece:--}" "${std:--}" "${easy:--}"
  done
  echo
  echo "Per-pass held-out scores:"
  for name in v6a v6b v6c; do tr '\r' '\n' < "results/$name-train.log" 2>/dev/null | grep '^pass ' | sed "s/^/  $name /" | cut -c1-120; done
} > results/experiments.txt
say "ALL EXPERIMENTS DONE"
kill "$awake" 2>/dev/null
