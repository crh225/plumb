**Title:** [bench request]: Add Plumb-4B (Qwen3.5-4B decision model, one-pass option readout, TypeSafe wire format, self-hosted)

Hi, I would like to submit **Plumb-4B**: an open 4B decision model that returns a probability for
every option from one forward pass, with zero generated tokens. On the public items it matches its
starting checkpoint (JevK5 v0.2) on easy and standard, scores **+6.3 points on hard (89
vs 82 of 111)**, and runs at the same speed. It is self-hosted on your GPU and speaks the
TypeSafe `/v1/systemone` wire format, so JevBench's standard `typesafe` adapter measures it unchanged.

## Reproduce

```bash
# 1. The runtime (jevk5 v0.2.0, Apache-2.0) and the pinned weights
pip install "jevk5[fast] @ https://github.com/allebee/jevk5/archive/refs/tags/v0.2.0.tar.gz"
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir plumb-4b

# 2. Serve it on your GPU (TypeSafe /v1/systemone)
jevk5-serve --model ./plumb-4b --port 8090 &
for i in $(seq 60); do curl -sf -o /dev/null localhost:8090/v1/systemone -d '{"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}' && break; sleep 5; done  # it loads in about a minute

# 3. JevBench's standard typesafe adapter, unchanged (the local server ignores the key; the adapter
#    requires one to be set, and results must be written outside the jevbench checkout)
mkdir -p runs && cd jevbench
TYPESAFE_API_KEY=unused python -m jevbench.cli run --adapter typesafe \
    --endpoint http://127.0.0.1:8090 --model plumb-4b \
    --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
    --results ../runs/plumb-4b.jsonl --run-label plumb-4b
```

We ran exactly these steps on a fresh, unmodified clone of jevbench `main` (RTX 4080 Super): 231/231
answered, 0 failed; hard 89/111 in one run and 90/111 in another (one borderline item flips). If you
still have the `jevk5_direct` adapter from JevK5's run, `--adapter jevk5_direct --endpoint crh225/plumb-4b
--revision 55de037801a8a9b9de3db5c0e16cef86210c2186` loads it too (JevK5's registration patch no longer applies to current `main`).

## Why it belongs here

A native one-pass decision model like the Jev-class entries already on the board: open weights
(Apache-2.0), open training data ([crh225/plumb-decisions](https://huggingface.co/datasets/crh225/plumb-decisions),
5,014 decisions plus a 131-item held-out set), and the full pipeline and audits at https://github.com/crh225/plumb.
Provenance: fine-tuned from JevK5 v0.2 on our own data, with its own weights, its own temperature and
different answers (8 hard items fixed and 1 broken against JevK5).

## Public items (our run, same runner and GPU for both)

| Tier | n | Plumb-4B | JevK5 v0.2 | ECE (Plumb / JevK5) | p50 (Plumb / JevK5) |
|---|---:|---:|---:|---:|---:|
| easy | 48 | **1.000** | 1.000 | 0.056 / 0.039 | 28 / 27 ms |
| standard | 72 | **0.972** | 0.972 | 0.152 / 0.137 | 28 / 28 ms |
| hard | 111 | **0.802** | 0.739 | 0.094 / 0.077 | 78 / 79 ms |

Hard tier against JevK5 v0.2: 8 fixed, 1 broken, exact McNemar
p = 0.039. Calibration is one temperature (T = 2.07) fitted on held-out
teacher questions; on these public items Plumb-4B's ECE is slightly higher than JevK5's on all three tiers.

## Use of public JevBench items during development

Two separate things, both true:

- **Not used as data or for selection.** No JevBench item, public or held out, was trained or tuned on.
  Every checkpoint choice and the temperature came only from our own held-out sets: teacher questions
  from three domains training never uses, held-out replay splits and a 65-item hand-written hard set.
- **Aggregate public results did shape the recipe.** We scored the 231 public items after each of five
  training rounds and looked at per-family accuracy. That feedback (long policies and dates not moving)
  is why later rounds added hard mining and long documents, and the long-document length (1,500-2,800
  words) was set after measuring how long the public hard documents are. This is development against
  the public distribution, and we state it plainly; no item's content was used.
- **Audits:** training items sharing more than two 8-word sequences with a public item were dropped (9,
  all generic policy phrases), and an exact normalised-text check finds no training state or
  instruction equal to, or containing, a public one.

## Technical details

- **Weights:** `crh225/plumb-4b` at `55de037801a8a9b9de3db5c0e16cef86210c2186`, bf16, Qwen3.5-4B architecture, Apache-2.0
  (`NOTICE` credits JevK5, Qwen and SemIf).
- **Readout:** softmax over the declared options' answer-letter logits at the last position, divided
  by T = 2.07; up to 16 options; inputs over 16,384 tokens are refused, never truncated; 0 output tokens.
- **Training:** LoRA r16 (attention projections) from JevK5 v0.2, merged. Teacher: Qwen3.8-27B writes
  each document and question and solves it twice with thinking; a question is kept only when both
  solutions match. Hard mining keeps the 2612 questions the starting model got wrong or was unsure
  of, plus a quarter of the rest, with as many human-labelled replay items (MNLI, WANLI, BoolQ, banking77,
  ARC, CommonsenseQA, MMLU-Pro train splits); 3 epochs at lr 3e-5. Held-out accuracy
  0.818 -> 0.846.
- **Cost basis:** Qwen3.5-4B class (deepinfra $0.03/M input tokens); measured input tokens
  164 / 168 / 1274 per decision (easy / standard / hard), 0 output tokens.
- **Full system card**, per-family table and per-item results: https://github.com/crh225/plumb/blob/main/bench/out/v5/SUBMISSION.md
