#!/usr/bin/env bash
# Run a command in the jevy container with the repo at /work and JevBench at /jevbench.
MSYS_NO_PATHCONV=1 exec docker run --rm --gpus all --shm-size 8g \
  -v "C:/Users/Chris/Development/jevy-1:/work" -v "C:/Users/Chris/Development/jevbench:/jevbench" \
  -e PYTHONPATH=/work:/work/training:/jevbench -e PYTHONUNBUFFERED=1 jevy:latest "$@"
