"""Exact-text audit: does any training item contain a public JevBench state or instruction?

    python training/audit_exact.py --jevbench ../jevbench/datasets/public \
        data/teacher/train*.jsonl data/decisions/train.jsonl

Normalises text (lowercase, letters and digits only, single spaces) and reports, over every
training record: states or instructions identical to a public item's, and public states or
instructions (of 8+ words) that appear verbatim inside a training state or instruction. This is
the "exact normalized-text audit" JevBench reports for submissions; the 8-gram scan
(scan_overlap.py) is the stricter, fuzzier companion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--jevbench", type=Path, required=True)
    args = parser.parse_args()
    public = []
    for path in sorted(args.jevbench.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            q = item.get("question") or {}
            public.append((item["id"], norm(item.get("state", "")), norm(q.get("instructions", ""))))
    pub_texts = {t for _, s, i in public for t in (s, i) if len(t.split()) >= 8}
    records = identical = contained = 0
    hits = []
    for path in args.data:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") or ("agree" in r and not r["agree"]):
                continue
            records += 1
            q = r.get("question") or {}
            mine = [norm(r.get("state", "")), norm(q.get("instructions", ""))]
            if any(m in pub_texts for m in mine):
                identical += 1
                hits.append((path.name, r.get("id"), "identical"))
                continue
            joined = " || ".join(mine)
            if any(len(t) > 40 and t in joined for t in pub_texts):
                contained += 1
                hits.append((path.name, r.get("id"), "contains a public text"))
    print(f"{len(public)} public items, {len(pub_texts)} public states/instructions of 8+ words")
    print(f"{records} training records checked: {identical} identical, {contained} containing a public text")
    for h in hits[:20]:
        print("  ", *h)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
