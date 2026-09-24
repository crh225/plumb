"""Check training data for text shared with JevBench's public items, before any training run.

    python training/scan_overlap.py data/gate1/train.jsonl --jevbench ../jevbench/datasets/public

Any 8-word sequence an item shares with a public JevBench item (state, instruction or option text)
is reported. JevBench's sealed items are private, so only the public 231 can be scanned; the
teacher never sees JevBench at all, so hits should be generic phrases, not leaked items. Exits 1
if any item shares more than --max-shared sequences, so a pipeline can stop on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

N = 8


def words(value) -> list[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(item: dict) -> set[tuple[str, ...]]:
    q = item.get("question") or {}
    w = words(item.get("state", "")) + words(q.get("instructions", "")) + words(q.get("criteria", ""))
    return {tuple(w[i : i + N]) for i in range(len(w) - N + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--jevbench", type=Path, required=True, help="datasets/public folder")
    parser.add_argument("--max-shared", type=int, default=2)
    parser.add_argument("--drop", action="store_true",
                        help="rewrite the data without items over --max-shared instead of failing")
    args = parser.parse_args()
    public: dict[tuple[str, ...], str] = {}
    for path in sorted(args.jevbench.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            for g in grams(item):
                public.setdefault(g, item["id"])
    flagged, total = [], 0
    for path in args.data:
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            total += 1
            shared = [g for g in grams(item) if g in public]
            if shared:
                flagged.append((item.get("id"), len(shared), public[shared[0]], " ".join(shared[0])))
    worst = sorted(flagged, key=lambda f: -f[1])
    print(f"{total} items scanned, {len(flagged)} share an {N}-word sequence with a public item")
    for item_id, n, public_id, example in worst[:15]:
        print(f"  {item_id}: {n} shared (e.g. with {public_id}: '{example}')")
    over = {f[0] for f in flagged if f[1] > args.max_shared}
    if over and args.drop:
        for path in args.data:
            kept = [line for line in path.open(encoding="utf-8") if json.loads(line).get("id") not in over]
            path.write_text("".join(kept), encoding="utf-8")
        print(f"dropped {len(over)} items sharing more than {args.max_shared} sequences")
        return 0
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
