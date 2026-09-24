"""Numbers for the system card: a model's public-item results against JevK5, from the runner's files.

    python bench/stats.py results/jevy-v4 [--baseline results/jevk5-bf16]

Per split: accuracy, ECE (10 bins), p50/p95 latency, mean input tokens, invalid answers. On the
hard tier: items fixed and broken against the baseline, an exact McNemar p-value, and per family.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(folder: Path, split: str) -> dict[str, dict]:
    path = folder / f"{split}.jsonl"
    if not path.exists():
        return {}
    return {r["task_id"]: r for r in map(json.loads, path.open(encoding="utf-8")) if r.get("task_id")}


def ece(rows: list[dict]) -> float:
    pts = []
    for r in rows:
        probs = r.get("probs") or {}
        if probs:
            conf = max(probs.values())
            pts.append((conf, 1.0 if r.get("correct") else 0.0))
    if not pts:
        return float("nan")
    total = 0.0
    for b in range(10):
        inb = [p for p in pts if min(int(p[0] * 10), 9) == b]
        if inb:
            total += abs(sum(c for c, _ in inb) / len(inb) - sum(h for _, h in inb) / len(inb)) * len(inb) / len(pts)
    return total


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b fixes and c breaks."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("results/jevk5-bf16"))
    args = parser.parse_args()
    out = {"splits": {}}
    print(f"{'split':<10}{'n':>4}{'acc':>8}{'ECE':>8}{'p50 ms':>8}{'p95 ms':>8}{'in tok':>8}{'invalid':>8}")
    for split, label in (("easy", "easy"), ("original", "standard"), ("hard", "hard")):
        rows = list(load(args.model, split).values())
        if not rows:
            continue
        acc = sum(1 for r in rows if r.get("correct")) / len(rows)
        lat = [1000 * r["latency_s"] for r in rows if r.get("latency_s") is not None]
        toks = [((r.get("usage") or {}).get("input_tokens") or (r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows]
        invalid = sum(1 for r in rows if not r.get("valid", True))
        s = {"n": len(rows), "acc": acc, "ece": ece(rows), "p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
             "input_tokens": sum(toks) / len(toks), "invalid": invalid}
        out["splits"][label] = s
        print(f"{label:<10}{s['n']:>4}{acc:>8.3f}{s['ece']:>8.3f}{s['p50']:>8.1f}{s['p95']:>8.1f}"
              f"{s['input_tokens']:>8.0f}{invalid:>8}")
    base, mine = load(args.baseline, "hard"), load(args.model, "hard")
    if base and mine:
        fixes = sum(1 for i in base if i in mine and not base[i].get("correct") and mine[i].get("correct"))
        breaks = sum(1 for i in base if i in mine and base[i].get("correct") and not mine[i].get("correct"))
        out["hard_vs_baseline"] = {"fixes": fixes, "breaks": breaks, "mcnemar_p": mcnemar(fixes, breaks)}
        print(f"\nhard tier vs {args.baseline.name}: fixes {fixes}, breaks {breaks}, "
              f"exact McNemar p = {mcnemar(fixes, breaks):.3f}")
        fam: dict[str, list[int]] = {}
        for i, r in base.items():
            f = fam.setdefault(r.get("family", "?"), [0, 0, 0])
            f[0] += 1
            f[1] += bool(r.get("correct"))
            f[2] += bool(mine.get(i, {}).get("correct"))
        out["families"] = fam
        for f, (n, b, m) in sorted(fam.items()):
            print(f"  {f:<18}{n:>3}  {args.baseline.name} {b:>2}  {args.model.name} {m:>2}")
    (args.model / "card-stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
