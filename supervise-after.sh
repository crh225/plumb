#!/usr/bin/env bash
# Start another supervisor once the current one has exited (so a running round is never cut off).
#   ./supervise-after.sh <HH:MM deadline> [kept-target]     (environment is passed through)
cd "$(dirname "$0")"
while powershell -NoProfile -Command "exit [int](@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -match 'supervise\.sh [0-9]' }).Count -eq 0)" ; do sleep 60; done
echo "$(date '+%F %T') previous supervisor finished; starting the next one (until $1)" >> results/supervise.log
exec ./supervise.sh "$@"
