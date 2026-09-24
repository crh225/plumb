#!/usr/bin/env bash
# Score a model on JevBench's 231 public items with JevBench's own runner, in the jevy container.
#   ./run-bench.sh <model: hub id or /work/models/... folder> <results folder name>
set -euo pipefail
model="$1"; out="$2"
mkdir -p "results/$out"
for split in easy original hard; do
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Chris/Development/jevy-1:/work" -v "C:/Users/Chris/Development/jevbench:/jevbench" \
    -e PYTHONPATH=/work:/work/training:/jevbench jevy:latest \
    python -m jevbench.cli run --tasks "/jevbench/datasets/public/$split.jsonl" \
      --adapter jevk5_direct --endpoint "$model" \
      --results "/work/results/$out/$split.jsonl" --ledger "/work/results/$out/ledger.jsonl" \
      --raw-dir "/work/results/$out/raw-$split" --run-label "$out-$split"
done
