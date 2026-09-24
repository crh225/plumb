"""Keep the teacher questions the starting model gets wrong or is unsure of (hard mining).

    python training/mine_hard.py data/gate-v4 --base alibiserikbay/JevK5 --unsure 0.8 --keep-easy 0.25

Scores every teacher question in <gate>/train.jsonl with the starting model at its own calibration
temperature, then rewrites train.jsonl with: every question it answers wrong, every question where
its probability on the right answer is below --unsure, a random --keep-easy share of the rest (so it
doesn't drift on what it already does well), and as many plain replay items as teacher questions
kept. Only our own teacher and replay data is touched; JevBench items are never read. The original
file is kept as train-all.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from lora import encode_items, evaluate

from jevk5.runtime import JevK5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gate", type=Path)
    parser.add_argument("--base", default="alibiserikbay/JevK5")
    parser.add_argument("--temperature", type=float, default=1.532)
    parser.add_argument("--unsure", type=float, default=0.8)
    parser.add_argument("--keep-easy", type=float, default=0.25)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    path = args.gate / "train.jsonl"
    items = [json.loads(x) for x in path.open(encoding="utf-8")]
    teacher = [it for it in items if str(it.get("source", "")).startswith("teacher")]
    replay = [it for it in items if not str(it.get("source", "")).startswith("teacher")]
    for i, it in enumerate(teacher):
        it["_mine"] = i

    model = JevK5(args.base, graphs=False, temperature=1.0)
    rows = encode_items(model, [{**it, "id": it["_mine"]} for it in teacher], args.max_len)
    probs = evaluate(model, rows, token_budget=8192)["probs"]

    verdict: dict[int, str] = {}
    for row, p in zip(rows, probs, strict=True):
        z = np.log(np.clip(p, 1e-12, 1.0)) / args.temperature
        q = np.exp(z - z.max())
        q /= q.sum()
        gold = int(row["target"].argmax())
        if q.argmax() != gold:
            verdict[row["id"]] = "wrong"
        elif q[gold] < args.unsure:
            verdict[row["id"]] = "unsure"
        else:
            verdict[row["id"]] = "easy"

    kept, counts, fam = [], {"wrong": 0, "unsure": 0, "easy": 0, "easy kept": 0}, {}
    for it in teacher:
        v = verdict.get(it["_mine"])
        if v is None:
            continue
        counts[v] += 1
        if v == "easy":
            if rng.random() >= args.keep_easy:
                continue
            counts["easy kept"] += 1
        else:
            f = it["source"].removeprefix("teacher_")
            fam[f] = fam.get(f, 0) + 1
        it.pop("_mine", None)
        kept.append(it)
    rng.shuffle(replay)
    train = kept + replay[: len(kept)]
    rng.shuffle(train)

    path.rename(args.gate / "train-all.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"mined {len(teacher)} teacher questions: {counts['wrong']} wrong, {counts['unsure']} unsure, "
          f"{counts['easy']} easy ({counts['easy kept']} kept)")
    print(f"hard ones by family: {dict(sorted(fam.items(), key=lambda x: -x[1]))}")
    print(f"train {len(train)} = {len(kept)} teacher + {len(train) - len(kept)} replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
