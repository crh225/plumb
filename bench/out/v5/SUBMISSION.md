# Plumb-4B: system card

An open one-pass typed-decision model: Qwen3.5-4B with a LoRA trained on decisions written and
double-checked by Qwen3.8-27B, read out as a restricted softmax over the declared options'
answer-letter logits at the last position, divided by one temperature. `native` probabilities, zero
generated tokens.

## System details
- **Weights:** `crh225/plumb-4b`, bf16, hub revision `55de037801a8a9b9de3db5c0e16cef86210c2186` (merged; same architecture and
  files as `alibiserikbay/JevK5`, so JevK5's runtime and JevBench adapter load it unchanged)
- **Code:** https://github.com/crh225/plumb (training pipeline, audits, per-item results)
- **Serving:** `jevk5-serve` (TypeSafe `/v1/systemone`, JevBench's `typesafe` adapter) or the
  in-process `jevk5_direct` adapter; one temperature T = 2.07 (`jevk5_config.json`)
- **License:** Apache-2.0. Derived from JevK5 v0.2 weights (Apache-2.0) and Qwen3.5-4B (Apache-2.0);
  training scripts from JevK5 (Apache-2.0); readout and prompt from SemIf (MIT). Attributions in NOTICE.
- **Limits:** at most 16 options; inputs over 16,384 tokens are refused, never truncated
- **Output tokens:** 0 per decision

## Training
- **Start:** JevK5 v0.2 (`alibiserikbay/JevK5`), LoRA r16 on the attention projections, merged.
- **Data:** decisions written by Qwen3.8-27B (FP8, vLLM) with thinking, each question solved twice
  independently and kept only when both solutions match the intended answer; typed like JevBench
  (true/false, choice, ordinal score), 2-6 options, families weighted toward the starting model's weak spots, and
  documents up to 2,800 words. Plus human-labelled public datasets in the same format (MNLI, WANLI,
  BoolQ, banking77, ARC, CommonsenseQA, MMLU-Pro train splits) so easy decisions don't regress.
  Hard mining: of 5017 kept teacher questions, the starting model answered 1279 wrong and was unsure (<0.8 on the right answer) on 1333; those, plus a random quarter of the rest, form the teacher part of the training set.
- **This round:** 6408 training items (3204 teacher questions, the rest replay) out of
  10043 assembled; 3 epochs at lr 3e-5.
- **Selection and calibration:** only held-out data: teacher questions from three domains training
  never sees, held-out splits of the replay datasets, and a hand-written hard set (65 items).
  Held-out accuracy 0.818 -> 0.846, ECE 0.043 -> 0.078.
- **No JevBench item, public or held out, and no output of Jev was used for training, tuning or
  selection.** Every training set passed an 8-word-sequence scan against the 231 public items (any
  item sharing more than two sequences is dropped; the hits are generic phrases such as "the action
  of the highest ranked applicable rule") and an exact normalised-text audit (no training state or
  instruction equals or contains a public one).

## Reference local run (public items)
JevBench's own runner, 231 public items, one RTX 4080 Super, in-process, batch 1, no CUDA graphs.

| Split | n | Plumb-4B | JevK5 v0.2 | ECE | JevK5 ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 48 | **1.000** | 1.000 | 0.056 | 0.039 | 28 ms | 30 ms |
| standard | 72 | **0.972** | 0.972 | 0.152 | 0.137 | 28 ms | 29 ms |
| hard | 111 | **0.802** | 0.739 | 0.094 | 0.077 | 78 ms | 505 ms |

On the hard tier against JevK5 v0.2 (same runner and GPU): 8 items fixed,
1 broken (exact McNemar p = 0.039).

| Hard family | n | JevK5 v0.2 | Plumb-4B |
|---|---:|---:|---:|
| adversarial | 6 | 5 | 5 |
| ambiguous | 7 | 6 | 6 |
| judge_hard | 17 | 13 | 13 |
| long_policy | 19 | 11 | 14 |
| multi_hop | 18 | 15 | 15 |
| probability | 10 | 7 | 10 |
| routing_hard | 5 | 5 | 5 |
| temporal_numeric | 15 | 7 | 7 |
| tradeoff | 6 | 5 | 6 |
| trap | 8 | 8 | 8 |

- **Latency:** RTX 4080 Super, in-process, batch 1, without CUDA graphs (an H100 with CUDA graphs is
  several times faster)
- **Cost basis:** Qwen3.5-4B class (deepinfra $0.03/M input tokens), measured input
  tokens 164 / 168 / 1274 per decision (easy / standard / hard), 0 output tokens
