"""Accuracy and calibration per JevBench public tier, from run-bench.sh result files.

    python training/score.py results/jevk5-bf16 results/jevy-v1

ECE uses ten equal-width confidence bins on the probability of the predicted label, the same
measure JevK5 reports for its hard tier. Failed or invalid answers count as wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPLITS = {"easy": "easy", "original": "standard", "hard": "hard"}


def score(rows: list[dict]) -> dict:
    n = len(rows)
    correct = [bool(r.get("correct")) for r in rows]
    conf = [max((r.get("probs") or {"x": 0.0}).values()) for r in rows]
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
        if idx:
            acc_b = sum(correct[i] for i in idx) / len(idx)
            conf_b = sum(conf[i] for i in idx) / len(idx)
            ece += abs(acc_b - conf_b) * len(idx) / n
    latency = sorted(r.get("latency_s") or 0 for r in rows)
    return {
        "n": n,
        "acc": sum(correct) / n if n else 0.0,
        "ece": ece,
        "p50_ms": 1000 * latency[len(latency) // 2] if latency else 0,
        "invalid": sum(not r.get("valid") for r in rows),
    }


def main() -> int:
    table = []
    for folder in map(Path, sys.argv[1:]):
        for split, name in SPLITS.items():
            path = folder / f"{split}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            table.append((folder.name, name, score(rows)))
    print(f"{'run':<22}{'tier':<10}{'n':>5}{'acc':>8}{'ece':>8}{'p50 ms':>9}{'invalid':>9}")
    for run, tier, s in table:
        print(
            f"{run:<22}{tier:<10}{s['n']:>5}{s['acc']:>8.3f}{s['ece']:>8.3f}"
            f"{s['p50_ms']:>9.0f}{s['invalid']:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
