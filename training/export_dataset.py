"""Export the kept teacher questions as a Hugging Face dataset folder (train and held-out test).

    python training/export_dataset.py out/plumb-decisions

train: every kept question from data/teacher/train*.jsonl and smoke.jsonl (both independent solves
matched the author's intended answer). test: kept questions from the held-out domains
(data/teacher/test*.jsonl), which training never saw and which set the calibration temperature.
Internal bookkeeping (token counts, raw solves, specs beyond family/domain/length) is dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rows(paths):
    seen = set()
    for path in paths:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("agree") or r["id"] in seen:
                continue
            seen.add(r["id"])
            spec = r.get("spec") or {}
            yield {
                "id": r["id"],
                "family": spec.get("family"),
                "domain": spec.get("domain"),
                "length": spec.get("length"),
                "state": r["state"],
                "question": r["question"],
                "expected": r["expected"],
                "explanation": r.get("explanation"),
                "teacher_probs": r.get("teacher_probs"),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    teacher = Path("data/teacher")
    splits = {
        "train": sorted(teacher.glob("train*.jsonl")) + [teacher / "smoke.jsonl"],
        "test": sorted(teacher.glob("test*.jsonl")),
    }
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    for split, paths in splits.items():
        items = list(rows(p for p in paths if p.exists()))
        with (args.out / "data" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        fam = Counter(it["family"] for it in items)
        types = Counter(it["question"]["type"] for it in items)
        long = sum(1 for it in items if it["length"] and "very long" in it["length"])
        print(f"{split}: {len(items)} questions, {long} very long; types {dict(types)}; families {dict(fam.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
