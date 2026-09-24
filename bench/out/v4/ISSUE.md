**Title:** [bench request]: Add Plumb-4B (Qwen3.5-4B, continued from JevK5 v0.2, native option-letter readout, in-process adapter)

Hi, I would like to submit **Plumb-4B** to JevBench. It has JevK5 v0.2's exact format, so it runs
through JevK5's existing `jevk5_direct` adapter with only the weights changed; the held-out items stay
on your machine.

```bash
git clone https://github.com/allebee/jevk5 && cd jevk5 && git checkout v0.2.0 && pip install -e ".[fast]"
cp bench/jevk5_direct.py <jevbench>/jevbench/adapters/
( cd <jevbench> && git apply <jevk5>/bench/jevbench-registration.patch )
cd <jevbench> && JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jevk5_direct \
    --endpoint crh225/plumb-4b --revision <hub revision sha> --model plumb-4b ...
```

The system card, training data description, overlap audits and per-item public results follow.

An open one-pass typed-decision model: Qwen3.5-4B, continued from JevK5 v0.2 with a LoRA trained on
decisions written and double-checked by Qwen3.8-27B, read out exactly like JevK5 (restricted softmax
over the declared options' answer-letter logits at the last position, divided by one temperature).
`native` probabilities, zero generated tokens.

## System details
- **Weights:** `crh225/plumb-4b`, bf16, hub revision `<hub revision sha>` (merged; same architecture and
  files as `alibiserikbay/JevK5`, so JevK5's runtime and JevBench adapter load it unchanged)
- **Code:** https://github.com/crh225/jevy (training pipeline, audits, per-item results)
- **Readout:** JevK5's `jevk5_direct` adapter, identical option mapping; temperature T = 1.95
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
  Hard mining: of 5017 kept teacher questions, JevK5 answered 1279 wrong and was unsure (<0.8 on the right answer) on 1333; those, plus a random quarter of the rest, form the teacher part of the training set.
- **This round:** 3238 training items (1619 teacher questions, the rest replay) out of
  5079 assembled; 3 epochs at lr 3e-5.
- **Selection and calibration:** only held-out data: teacher questions from three domains training
  never sees, held-out splits of the replay datasets, and a hand-written hard set (65 items).
  Held-out accuracy 0.806 -> 0.844, ECE 0.056 -> 0.065.
- **No JevBench item, public or held out, and no output of Jev was used for training, tuning or
  selection.** Every training set passed an 8-word-sequence scan against the 231 public items (any
  item sharing more than two sequences is dropped; the hits are generic phrases such as "the action
  of the highest ranked applicable rule") and an exact normalised-text audit (no training state or
  instruction equals or contains a public one).

## Reference local run (public items)
JevBench's own runner, 231 public items, one RTX 4080 Super, in-process, batch 1, no CUDA graphs.

| Split | n | Plumb-4B | JevK5 v0.2 | ECE | JevK5 ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 48 | **1.000** | 1.000 | 0.044 | 0.039 | 28 ms | 29 ms |
| standard | 72 | **0.972** | 0.972 | 0.139 | 0.137 | 28 ms | 30 ms |
| hard | 111 | **0.775** | 0.739 | 0.051 | 0.077 | 79 ms | 507 ms |

On the hard tier against JevK5 v0.2 (same runner and GPU): 9 items fixed,
5 broken (exact McNemar p = 0.424).

| Hard family | n | JevK5 v0.2 | Plumb-4B |
|---|---:|---:|---:|
| adversarial | 6 | 5 | 5 |
| ambiguous | 7 | 6 | 6 |
| judge_hard | 17 | 13 | 13 |
| long_policy | 19 | 11 | 13 |
| multi_hop | 18 | 15 | 13 |
| probability | 10 | 7 | 9 |
| routing_hard | 5 | 5 | 5 |
| temporal_numeric | 15 | 7 | 8 |
| tradeoff | 6 | 5 | 6 |
| trap | 8 | 8 | 8 |

- **Latency:** RTX 4080 Super, in-process, batch 1, without CUDA graphs (JevK5 reports ~13-30 ms p50 on an
  H100 with graphs; the forward pass is identical)
- **Cost basis:** same as JevK5 (Qwen3.5-4B class, deepinfra $0.03/M input tokens), measured input
  tokens 164 / 168 / 1274 per decision (easy / standard / hard), 0 output tokens
