"""Assemble one training folder (train.jsonl + dev.jsonl) from teacher, replay and hard replay.

    python training/build_gate.py --out data/gate1 \
        --teacher data/teacher/train.jsonl --teacher-test data/teacher/test.jsonl \
        --replay data/decisions/train.jsonl:5000 --replay data/hard/train.jsonl:all \
        --dev data/decisions/dev.jsonl:600 --dev data/hard/dev.jsonl:all --dev data/decisions/hard_dev.jsonl:all

Teacher questions go through build_train.teacher_items (kept questions only, option keys rebuilt
from option text). dev.jsonl holds only held-out material: teacher questions from test domains,
dev splits of the replay sources, and the hand-written hard set. Nothing from JevBench.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_train import teacher_items


def take(spec: str, rng: random.Random) -> list[dict]:
    path, _, n = spec.partition(":")
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    rng.shuffle(rows)
    return rows if n in ("", "all") else rows[: int(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, nargs="*", default=[])
    parser.add_argument("--teacher-test", type=Path, nargs="*", default=[])
    parser.add_argument("--replay", action="append", default=[], help="path[:count|all]")
    parser.add_argument("--dev", action="append", default=[], help="path[:count|all]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    teacher = [i for p in args.teacher if p.exists() for i in teacher_items(p)]
    train = teacher + [r for spec in args.replay for r in take(spec, rng)]
    rng.shuffle(train)
    dev = [i for p in args.teacher_test if p.exists() for i in teacher_items(p)]
    dev += [r for spec in args.dev for r in take(spec, rng)]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in train:
        counts[r["source"].split("_")[0] if r["source"].startswith("teacher") else r["source"]] = (
            counts.get(r["source"].split("_")[0] if r["source"].startswith("teacher") else r["source"], 0) + 1
        )
    print(f"train {len(train)} ({len(teacher)} teacher), dev {len(dev)}; by source {counts}")


if __name__ == "__main__":
    main()
