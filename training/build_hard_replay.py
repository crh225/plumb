"""Hard human-labelled decisions from public datasets, aimed at JevK5's weak families.

    python training/build_hard_replay.py --out data/hard --per-source 1500 --dev-per-source 150

No teacher needed: every label comes from the dataset's own annotators. Sources and the JevBench
families they exercise (families only; nothing here comes from JevBench):
  contract_nli  real NDAs + a hypothesis: entailed / contradicted / not mentioned  (long policy,
                abstention)
  anli          adversarially written NLI, rounds 1-3                              (traps)
  tabfact       a table + a statement: supported or refuted                        (multi-hop
                lookup, numbers)
  quality       a long article + a hard multiple-choice question                   (long reading)
  aqua_rat      arithmetic word problems with five options                         (numeric)
  date          BIG-Bench Hard date understanding                                  (temporal)
Documents longer than --max-words are dropped, never cut: a truncated contract can change its
answer. Choice keys come from the option text, as in build_replay.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from build_replay import NLI, keyed


def contract_nli(split: str):
    ds = load_dataset("reuben256/contract-nli", split={"train": "train", "dev": "validation"}[split])
    labels = {
        "entailment": ("entailed", "The contract states or clearly implies the hypothesis."),
        "contradiction": ("contradicted", "The contract states the opposite of the hypothesis."),
        "notmentioned": ("not_mentioned", "The contract does not address the hypothesis either way."),
    }
    criteria = {key: text for key, text in labels.values()}
    for i, r in enumerate(ds):
        key = labels[r["label"].lower()][0]
        yield {
            "id": f"contract_nli-{split}-{i}",
            "source": "hard_contract_nli",
            "state": {"contract": r["text"], "hypothesis": r["hypothesis"]},
            "question": {
                "type": "choice",
                "instructions": "What does the contract say about the hypothesis?",
                "criteria": criteria,
            },
            "expected": key,
        }


def anli(split: str):
    names = ["entailment", "neutral", "contradiction"]
    for rnd in ("r1", "r2", "r3"):
        ds = load_dataset("facebook/anli", split=f"{'train' if split == 'train' else 'dev'}_{rnd}")
        for i, r in enumerate(ds):
            yield {
                "id": f"anli-{rnd}-{split}-{i}",
                "source": "hard_anli",
                "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                "question": {
                    "type": "choice",
                    "instructions": "How does the hypothesis relate to the premise?",
                    "criteria": NLI,
                },
                "expected": names[int(r["label"])],
            }


def tabfact(split: str):
    ds = load_dataset("table-benchmark/tabfact", split={"train": "train", "dev": "validation"}[split])
    for i, r in enumerate(ds):
        rows = [" | ".join(cell.strip() for cell in line.split("#")) for line in r["table"].split("\n")]
        yield {
            "id": f"tabfact-{split}-{i}",
            "source": "hard_tabfact",
            "state": {"table_title": r["table_title"], "table": "\n".join(rows)},
            "question": {
                "type": "noul",
                "instructions": f"Does the table support this statement? Statement: {r['question']}",
                "criteria": {
                    "true": "The table supports the statement.",
                    "false": "The table contradicts the statement.",
                },
            },
            "expected": "true" if r["answer"].lower().startswith("entail") else "false",
        }


def quality(split: str):
    ds = load_dataset("emozilla/quality", split={"train": "train", "dev": "validation"}[split])
    for i, r in enumerate(ds):
        options = r["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        criteria = keyed(options)
        yield {
            "id": f"quality-{split}-{i}",
            "source": "hard_quality",
            "state": {"article": r["article"]},
            "question": {
                "type": "choice",
                "instructions": r["question"].strip(),
                "criteria": criteria,
            },
            "expected": list(criteria)[int(r["answer"])],
            "_hard": str(r.get("hard")).lower() == "true",
        }


def aqua_rat(split: str):
    ds = load_dataset("deepmind/aqua_rat", "raw", split={"train": "train", "dev": "validation"}[split])
    for i, r in enumerate(ds):
        options = r["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        texts = [re.sub(r"^[A-E]\)\s*", "", o).strip() for o in options]
        letters = [o[0] for o in options]
        if r["correct"] not in letters or len(set(texts)) != len(texts):
            continue
        criteria = keyed(texts)
        yield {
            "id": f"aqua-{split}-{i}",
            "source": "hard_aqua_rat",
            "state": {"problem": r["question"]},
            "question": {
                "type": "choice",
                "instructions": "Which option is the correct answer to the problem?",
                "criteria": criteria,
            },
            "expected": list(criteria)[letters.index(r["correct"])],
        }


def date_understanding(split: str):
    ds = list(load_dataset("lukaemon/bbh", "date_understanding", split="test"))
    # only a test split exists: its first 200 items train, the last 50 are dev
    chosen = ds[:200] if split == "train" else ds[200:]
    for i, r in enumerate(chosen):
        question, _, opts = r["input"].partition("\nOptions:\n")
        found = re.findall(r"\(([A-F])\)\s*(.+)", opts)
        letters, texts = [f[0] for f in found], [f[1].strip() for f in found]
        gold = r["target"].strip("() ")
        if gold not in letters or len(set(texts)) != len(texts):
            continue
        criteria = keyed(texts)
        yield {
            "id": f"date-{split}-{i}",
            "source": "hard_date",
            "state": {"question": question.strip()},
            "question": {
                "type": "choice",
                "instructions": "Which option answers the date question correctly?",
                "criteria": criteria,
            },
            "expected": list(criteria)[letters.index(gold)],
        }


SOURCES = {
    "contract_nli": contract_nli,
    "anli": anli,
    "tabfact": tabfact,
    "quality": quality,
    "aqua_rat": aqua_rat,
    "date": date_understanding,
}


def word_count(item: dict) -> int:
    return len(json.dumps(item["state"], ensure_ascii=False).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=1500)
    parser.add_argument("--dev-per-source", type=int, default=150)
    parser.add_argument("--max-words", type=int, default=2600, help="~3.5k tokens")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.per_source), ("dev", args.dev_per_source)):
        rng = random.Random(args.seed + (split == "dev"))
        rows, counts = [], {}
        for name, make in SOURCES.items():
            items = [x for x in make(split) if word_count(x) <= args.max_words]
            if name == "quality":  # prefer the questions annotators marked hard
                items.sort(key=lambda x: not x.pop("_hard"))
            else:
                rng.shuffle(items)
            chosen = items[:n] if name == "quality" else items[:n]
            rng.shuffle(chosen)
            counts[name] = (len(chosen), len(items))
            rows += chosen
        rng.shuffle(rows)
        for r in rows:
            r.pop("_hard", None)
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split, len(rows), {k: f"{a} of {b} that fit" for k, (a, b) in counts.items()})


if __name__ == "__main__":
    main()
