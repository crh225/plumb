"""Human-labelled replay items in the decision format, so easy decisions don't regress.

    python training/build_replay.py --out data/decisions --train 9000 --dev 1200

Writes train.jsonl (from each dataset's train split) and dev.jsonl (from its validation/test
split). Nothing here comes from JevBench. Each item is
{id, source, state, question: {type, instructions, criteria}, expected}; choice option keys are
built from the option text (as build_train.neutral_keys does), so a key never names the answer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

NLI = {
    "entailment": "The hypothesis must be true if the premise is true.",
    "neutral": "The hypothesis might be true or false; the premise does not settle it.",
    "contradiction": "The hypothesis cannot be true if the premise is true.",
}


def keyed(options: list[str]) -> dict[str, str]:
    out, used = {}, set()
    for text in options:
        words = re.findall(r"[a-z0-9]+", str(text).lower())[:6] or ["option"]
        key = "_".join(words)
        while key in used:
            key += "_x"
        used.add(key)
        out[key] = str(text)
    return out


def nli(rows, source, label_names):
    for i, r in enumerate(rows):
        label = r["label"] if "label" in r else r["gold"]  # WANLI calls it "gold"
        name = label_names[label] if isinstance(label, int) else label
        if name not in NLI:
            continue
        yield {
            "id": f"{source}-{i}",
            "source": source,
            "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
            "question": {
                "type": "choice",
                "instructions": "How does the hypothesis relate to the premise?",
                "criteria": NLI,
            },
            "expected": name,
        }


def boolq(rows):
    for i, r in enumerate(rows):
        yield {
            "id": f"boolq-{i}",
            "source": "boolq",
            "state": {"title": r.get("title", ""), "passage": r["passage"]},
            "question": {"type": "noul", "instructions": r["question"].rstrip("?") + "?"},
            "expected": "true" if r["answer"] else "false",
        }


def banking77(rows, names, rng):
    for i, r in enumerate(rows):
        gold = r["label_text"]
        others = rng.sample([n for n in names if n != gold], rng.randint(5, 11))
        options = others + [gold]
        rng.shuffle(options)
        yield {
            "id": f"banking77-{i}",
            "source": "banking77",
            "state": {"customer_message": r["text"]},
            "question": {
                "type": "choice",
                "instructions": "Which intent best describes the customer's message?",
                "criteria": {o: o.replace("_", " ") for o in options},
            },
            "expected": gold,
        }


def multiple_choice(rows, source, get):
    for i, r in enumerate(rows):
        question, options, gold_index = get(r)
        if not options or gold_index is None or len(options) > 16:
            continue
        criteria = keyed(options)
        yield {
            "id": f"{source}-{i}",
            "source": source,
            "state": {"question": question},
            "question": {
                "type": "choice",
                "instructions": "Which option answers the question correctly?",
                "criteria": criteria,
            },
            "expected": list(criteria)[gold_index],
        }


def arc_get(r):
    labels = r["choices"]["label"]
    return r["question"], r["choices"]["text"], (
        labels.index(r["answerKey"]) if r["answerKey"] in labels else None
    )


def csqa_get(r):
    labels = r["choices"]["label"]
    return r["question"], r["choices"]["text"], (
        labels.index(r["answerKey"]) if r["answerKey"] in labels else None
    )


def mmlu_pro_get(r):
    return r["question"], r["options"], r["answer_index"]


def collect(split: str, rng: random.Random) -> dict[str, list[dict]]:
    """Every source's items for 'train' or 'dev' (validation/test splits)."""
    pick = lambda train, dev: train if split == "train" else dev  # noqa: E731
    out = {}
    mnli = load_dataset("nyu-mll/multi_nli", split=pick("train", "validation_matched"))
    out["mnli"] = list(nli(mnli, "mnli", mnli.features["label"].names))
    wanli = load_dataset("alisawuffles/WANLI", split=pick("train", "test"))
    out["wanli"] = list(nli(wanli, "wanli", None))
    out["boolq"] = list(boolq(load_dataset("google/boolq", split=pick("train", "validation"))))
    # PolyAI/banking77 needs a loader script datasets no longer runs; mteb's copy is plain files
    b77 = load_dataset("mteb/banking77", split=pick("train", "test"))
    out["banking77"] = list(banking77(b77, sorted(set(b77["label_text"])), rng))
    for config in ("ARC-Challenge", "ARC-Easy"):
        arc = load_dataset("allenai/ai2_arc", config, split=pick("train", "test"))
        out.setdefault("arc", []).extend(multiple_choice(arc, "arc", arc_get))
    csqa = load_dataset("tau/commonsense_qa", split=pick("train", "validation"))
    out["csqa"] = list(multiple_choice(csqa, "csqa", csqa_get))
    # MMLU-Pro has only test and validation; its test split is our training pool, validation is dev
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split=pick("test", "validation"))
    out["mmlu_pro"] = list(multiple_choice(mmlu, "mmlu_pro", mmlu_pro_get))
    return out


def balanced(pools: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    """n items spread evenly over sources (a small source gives all it has)."""
    for items in pools.values():
        rng.shuffle(items)
    chosen, share = [], n // len(pools)
    for name, items in pools.items():
        chosen += items[:share]
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=9000)
    parser.add_argument("--dev", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.train), ("dev", args.dev)):
        rng = random.Random(args.seed + (split == "dev"))
        pools = collect(split, rng)
        rows = balanced(pools, n, rng)
        with (args.out / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split, len(rows), {k: len(v) for k, v in pools.items()})


if __name__ == "__main__":
    main()
