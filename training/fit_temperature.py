"""Fit a trained model's one calibration temperature on held-out data and save it with the weights.

    python training/fit_temperature.py models/jevy-v1 data/gate1/dev.jsonl

Like JevK5's 1.532: one scalar T minimising negative log-likelihood on the held-out-domain teacher
questions (sources starting 'teacher_'), falling back to all dev items if there are too few.
Never fitted on JevBench items. Writes jevk5_config.json {"temperature": T} next to the weights,
which the runtime and the JevBench adapter read, and prints ECE before and after per dev group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lora import encode_items, evaluate

from jevk5.runtime import JevK5


def scaled(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def ece(probs: list[np.ndarray], gold: list[int]) -> float:
    conf = np.array([p.max() for p in probs])
    hit = np.array([p.argmax() == g for p, g in zip(probs, gold, strict=True)], dtype=float)
    bins = np.minimum((conf * 10).astype(int), 9)
    return float(sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(10) if (bins == b).any()))


def nll(probs: list[np.ndarray], gold: list[int]) -> float:
    return float(-np.mean([np.log(p[g] + 1e-12) for p, g in zip(probs, gold, strict=True)]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("dev", type=Path)
    parser.add_argument("--min-fit", type=int, default=150)
    args = parser.parse_args()
    model = JevK5(str(args.model), graphs=False, temperature=1.0)
    rows = encode_items(model, [json.loads(x) for x in args.dev.open(encoding="utf-8")], 4096)
    probs = evaluate(model, rows, token_budget=8192)["probs"]
    gold = [int(r["target"].argmax()) for r in rows]
    fit = [i for i, r in enumerate(rows) if r["source"].startswith("teacher_")]
    basis = "held-out teacher questions"
    if len(fit) < args.min_fit:
        fit, basis = list(range(len(rows))), "all dev items (too few held-out teacher questions)"
    grid = np.round(np.arange(0.5, 3.001, 0.01), 2)
    losses = [nll([scaled(probs[i], t) for i in fit], [gold[i] for i in fit]) for t in grid]
    t_best = float(grid[int(np.argmin(losses))])
    (args.model / "jevk5_config.json").write_text(json.dumps({"temperature": t_best}) + "\n")
    print(f"temperature {t_best} fitted on {len(fit)} {basis}")
    groups = {
        "teacher held-out": lambda s: s.startswith("teacher_"),
        "hand-written hard": lambda s: s.startswith("hard_dev_"),
        "replay dev": lambda s: not s.startswith(("teacher_", "hard_dev_")),
    }
    for name, keep in groups.items():
        idx = [i for i, r in enumerate(rows) if keep(r["source"])]
        if not idx:
            continue
        g = [gold[i] for i in idx]
        raw = [probs[i] for i in idx]
        cal = [scaled(probs[i], t_best) for i in idx]
        acc = np.mean([p.argmax() == y for p, y in zip(raw, g, strict=True)])
        print(f"  {name:<18} n {len(idx):>4}  acc {acc:.3f}  ECE {ece(raw, g):.3f} -> {ece(cal, g):.3f}"
              f"  NLL {nll(raw, g):.3f} -> {nll(cal, g):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
