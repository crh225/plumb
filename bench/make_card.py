"""Write the submission documents for one trained round: system card, Hugging Face model card, and
the JevBench issue text, all filled in from that round's own files.

    python bench/make_card.py v4 --hf-repo <user>/jevy-4b [--revision <sha>] [--name "Jevy 4B"]

Reads results/jevy-<round>/card-stats.json (run bench/stats.py first), results/<round>-train.log,
models/jevy-<round>/jevk5_config.json and data/gate-<round>/. Writes bench/out/<round>/SUBMISSION.md,
MODEL_CARD.md (the Hugging Face README) and ISSUE.md. Nothing is uploaded or posted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def lines(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("round")
    parser.add_argument("--hf-repo", default="<hf-user>/jevy-4b")
    parser.add_argument("--revision", default="<hub revision sha>")
    parser.add_argument("--name", default="Jevy 4B")
    parser.add_argument("--repo", default="<public code repo>")
    args = parser.parse_args()
    r = args.round
    stats = json.loads(Path(f"results/jevy-{r}/card-stats.json").read_text(encoding="utf-8"))
    base = json.loads(Path("results/jevk5-bf16/card-stats.json").read_text(encoding="utf-8")) \
        if Path("results/jevk5-bf16/card-stats.json").exists() else {"splits": {}}
    temp = json.loads(Path(f"models/jevy-{r}/jevk5_config.json").read_text())["temperature"]
    log = Path(f"results/{r}-train.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    before = json.loads(re.search(r"^before (\{.*\})", log, re.M).group(1))
    after = json.loads(re.search(r"^after (\{.*\})", log, re.M).group(1))
    gate = Path(f"data/gate-{r}")
    train_all, train = lines(gate / "train-all.jsonl"), lines(gate / "train.jsonl")
    teacher_used = sum(1 for x in (gate / "train.jsonl").open(encoding="utf-8")
                       if json.loads(x).get("source", "").startswith("teacher"))
    sup = Path("results/supervise.log").read_text(encoding="utf-8", errors="replace")
    mined = re.findall(r"^mined (\d+) teacher questions: (\d+) wrong, (\d+) unsure", sup, re.M)
    epochs = re.search(r"== training " + r + r" from JevK5 \((\d+) epochs, lr ([\d.e-]+)\)", sup)
    s, h = stats["splits"], stats.get("hard_vs_baseline", {})
    fam = stats.get("families", {})

    def row(label):
        m, b = s.get(label, {}), base["splits"].get(label, {})
        return (f"| {label} | {m.get('n', '')} | **{m.get('acc', 0):.3f}** | {b.get('acc', float('nan')):.3f} | "
                f"{m.get('ece', 0):.3f} | {b.get('ece', float('nan')):.3f} | {m.get('p50', 0):.0f} ms | {m.get('p95', 0):.0f} ms |")

    table = "\n".join(["| Split | n | " + args.name + " | JevK5 v0.2 | ECE | JevK5 ECE | p50 | p95 |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|",
                       row("easy"), row("standard"), row("hard")])
    fam_lines = "\n".join(f"| {f} | {n} | {b} | {m} |" for f, (n, b, m) in sorted(fam.items()))
    mined_text = (f"Hard mining: of {mined[-1][0]} kept teacher questions, JevK5 answered {mined[-1][1]} wrong and was "
                  f"unsure (<0.8 on the right answer) on {mined[-1][2]}; those, plus a random quarter of the rest, "
                  "form the teacher part of the training set.") if mined else ""
    tokens = " / ".join(f"{s[k]['input_tokens']:.0f}" for k in ("easy", "standard", "hard") if k in s)

    card = f"""# {args.name}: system card

An open one-pass typed-decision model: Qwen3.5-4B, continued from JevK5 v0.2 with a LoRA trained on
decisions written and double-checked by Qwen3.8-27B, read out exactly like JevK5 (restricted softmax
over the declared options' answer-letter logits at the last position, divided by one temperature).
`native` probabilities, zero generated tokens.

## System details
- **Weights:** `{args.hf_repo}`, bf16, hub revision `{args.revision}` (merged; same architecture and
  files as `alibiserikbay/JevK5`, so JevK5's runtime and JevBench adapter load it unchanged)
- **Code:** {args.repo} (training pipeline, audits, per-item results)
- **Readout:** JevK5's `jevk5_direct` adapter, identical option mapping; temperature T = {temp}
  (`jevk5_config.json`)
- **License:** Apache-2.0. Derived from JevK5 v0.2 weights (Apache-2.0) and Qwen3.5-4B (Apache-2.0);
  training scripts from JevK5 (Apache-2.0); readout and prompt from SemIf (MIT). Attributions in NOTICE.
- **Limits:** as JevK5: at most 16 options; inputs over 16,384 tokens are refused, never truncated
- **Output tokens:** 0 per decision

## Training
- **Start:** JevK5 v0.2 (`alibiserikbay/JevK5`), LoRA r16 on the attention projections, merged.
- **Data:** decisions written by Qwen3.8-27B (FP8, vLLM) with thinking, each question solved twice
  independently and kept only when both solutions match the intended answer; typed like JevBench
  (true/false, choice, ordinal score), 2-6 options, families weighted toward JevK5's weak spots, and
  documents up to 2,800 words. Plus human-labelled public datasets in the same format (MNLI, WANLI,
  BoolQ, banking77, ARC, CommonsenseQA, MMLU-Pro train splits) so easy decisions don't regress.
  {mined_text}
- **This round:** {train} training items ({teacher_used} teacher questions, the rest replay) out of
  {train_all or train} assembled; {epochs.group(1) if epochs else '?'} epochs at lr {epochs.group(2) if epochs else '?'}.
- **Selection and calibration:** only held-out data: teacher questions from three domains training
  never sees, held-out splits of the replay datasets, and a hand-written hard set (65 items).
  Held-out accuracy {before['acc']:.3f} -> {after['acc']:.3f}, ECE {before['ece']:.3f} -> {after['ece']:.3f}.
- **No JevBench item, public or held out, and no output of Jev was used for training, tuning or
  selection.** Every training set passed an 8-word-sequence scan against the 231 public items (any
  item sharing more than two sequences is dropped; the hits are generic phrases such as "the action
  of the highest ranked applicable rule") and an exact normalised-text audit (no training state or
  instruction equals or contains a public one).

## Reference local run (public items)
JevBench's own runner, 231 public items, one RTX 4080 Super, in-process, batch 1, no CUDA graphs.

{table}

On the hard tier against JevK5 v0.2 (same runner and GPU): {h.get('fixes', '?')} items fixed,
{h.get('breaks', '?')} broken (exact McNemar p = {h.get('mcnemar_p', float('nan')):.3f}).

| Hard family | n | JevK5 v0.2 | {args.name} |
|---|---:|---:|---:|
{fam_lines}

- **Latency:** RTX 4080 Super, in-process, batch 1, without CUDA graphs (JevK5 reports ~13-30 ms p50 on an
  H100 with graphs; the forward pass is identical)
- **Cost basis:** same as JevK5 (Qwen3.5-4B class, deepinfra $0.03/M input tokens), measured input
  tokens {tokens} per decision (easy / standard / hard), 0 output tokens
"""
    model_card = f"""---
license: apache-2.0
base_model: alibiserikbay/JevK5
tags: [decision-model, classification, calibration, jevbench, qwen3.5]
---

# {args.name}

A 4B decision model: give it evidence, a question and 2-16 options, and it returns a calibrated
probability for every option from one forward pass, with no generated tokens. Continued from
[JevK5 v0.2](https://huggingface.co/alibiserikbay/JevK5) on hard decisions written and
double-checked by Qwen3.8-27B. Runs with JevK5's runtime unchanged (temperature {temp} in
`jevk5_config.json`).

{table}

Public JevBench items, JevBench's own runner, RTX 4080 Super. No JevBench item was used for
training, tuning or selection. Details, audits and per-item results: {args.repo}.

License Apache-2.0 (derived from JevK5 v0.2 and Qwen3.5-4B, both Apache-2.0; readout from SemIf, MIT).
"""
    issue = f"""**Title:** [bench request]: Add {args.name} (Qwen3.5-4B, continued from JevK5 v0.2, native option-letter readout, in-process adapter)

Hi, I would like to submit **{args.name}** to JevBench. It has JevK5 v0.2's exact format, so it runs
through JevK5's existing `jevk5_direct` adapter with only the weights changed; the held-out items stay
on your machine.

```bash
git clone https://github.com/allebee/jevk5 && cd jevk5 && git checkout v0.2.0 && pip install -e ".[fast]"
cp bench/jevk5_direct.py <jevbench>/jevbench/adapters/
( cd <jevbench> && git apply <jevk5>/bench/jevbench-registration.patch )
cd <jevbench> && JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jevk5_direct \\
    --endpoint {args.hf_repo} --revision {args.revision} --model {args.hf_repo.split('/')[-1]} ...
```

The system card, training data description, overlap audits and per-item public results follow.

""" + card.split("\n", 2)[2]
    out = Path("bench/out") / r
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUBMISSION.md").write_text(card, encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (out / "ISSUE.md").write_text(issue, encoding="utf-8")
    print(f"wrote {out}/SUBMISSION.md, MODEL_CARD.md, ISSUE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
