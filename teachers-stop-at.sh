#!/usr/bin/env bash
# Stop every teacher process at a set time (for runs outside supervise.sh).   ./teachers-stop-at.sh HH:MM
cd "$(dirname "$0")"
until [ "$(date +%H%M)" -ge "${1/:/}" ]; do sleep 60; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'teacher\.py' -and (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'uv.exe') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "$(date '+%F %T') teachers stopped at $1" >> results/supervise.log
