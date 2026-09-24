# Plumb-4B

An open, one-pass typed-decision model. Give it evidence, a question and 2-16 options; it returns a
calibrated probability for every option from a single forward pass, with no generated tokens.

This repository is the full pipeline behind it: a teacher that writes and double-checks hard
decisions, the training rounds, the audits that keep benchmark items out of training, and every
round's results.

## Results so far

[JevBench](https://github.com/fstandhartinger/jevbench)'s own runner on its 231 public items, one
RTX 4080 Super, in-process, batch 1. The first row is the starting checkpoint, re-run on the same
machine.

| Model | Teacher questions | Easy | Standard | Hard | Hard ECE | Held-out check |
|---|---:|---:|---:|---:|---:|---:|
| starting point (JevK5 v0.2) | | 1.000 | 0.972 | 0.739 | 0.077 | 0.812 |
| v1 | 0 (public datasets only) | 1.000 | 0.972 | 0.694 | 0.095 | |
| v2 | 495 | 1.000 | 0.972 | 0.757 | 0.034 | 0.824 |
| v3 | 909 | 1.000 | 0.972 | 0.739 | 0.070 | 0.834 |

The held-out check is our own: teacher questions from domains training never sees, held-out splits
of the public datasets, and a hand-written hard set. Models are chosen on it, never on JevBench.

So far the held-out check improves steadily, but the public hard tier barely moves: v2 is two items
of 111 ahead of the starting point, and v3 is level with it. Per family, dates and numbers, long
policies, judging and probability haven't changed. The rounds in progress test two likely reasons
(see Log): training too gently on questions the model already answers, and documents much shorter
than the benchmark's.

JevBench's official score also includes a private sealed set that only its maintainer runs.

## Model card

Each scored round gets a system card, a Hugging Face model card and a JevBench submission draft,
generated from its own results into [`bench/out/<round>/`](bench/out/). The first is v4's.

## Rules

- **No JevBench item, public or held out, is trained on, tuned on, or used to pick a model or a
  temperature.** The public items are a scoreboard at the end of a round.
- Every training set is scanned against the 231 public items before training
  (`training/scan_overlap.py`): items sharing more than two 8-word sequences are dropped. The hits
  are generic phrases such as "the action of the highest ranked applicable rule".
- An exact normalised-text audit (`training/audit_exact.py`) finds no training state or instruction
  equal to, or containing, a public one (12,143 records checked).

## How it is trained

1. **Teacher** (`training/teacher.py`): Qwen3.8-27B writes realistic documents with hard typed
   questions (true/false, choice, ordinal score) and answers each twice with thinking; a question is
   kept only when both answers match the intended one. Families follow JevBench's published
   categories, weighted toward the model's weak spots. Served with vLLM (FP8) on rented GPUs; a local
   llama.cpp build (`serve-teacher.sh`) and a shared remote server were used early on.
2. **Replay** (`training/build_replay.py`): human-labelled public datasets (MNLI, WANLI, BoolQ,
   banking77, ARC, CommonsenseQA, MMLU-Pro) in the same format, as many items as teacher questions,
   so easy decisions don't regress.
3. **Round** (`cycle.sh`): assemble the set (`training/build_gate.py`), scan it, optionally keep only
   the teacher questions the starting model gets wrong or is unsure of (`training/mine_hard.py`),
   LoRA-train with cross-entropy on the option-letter logits (`training/lora.py`), fit one
   temperature on held-out teacher questions (`training/fit_temperature.py`), and run the public
   items (`run-bench.sh`). `round.conf` holds the current recipe.
4. **Unattended runs** (`supervise.sh`): keep the teachers writing, start rounds on a question count
   or a clock, push each round's scores. `dashboard/` shows it live.

Training and evaluation run in the `Dockerfile` image (PyTorch 2.14, CUDA 13) on a 16 GB GPU.
Teacher endpoints and keys live in git-ignored `*-env.sh` files.

## Running it

The weights use the same format and readout as JevK5 v0.2, so its runtime and JevBench adapter
(`jevk5_direct`) load them unchanged; only the weights name and the temperature in
`jevk5_config.json` differ.

## Submitting to JevBench

`bench/stats.py results/jevy-<round>` computes the numbers a submission needs (per-tier accuracy,
calibration, latency, input tokens, fixes and breaks against the starting point with an exact
McNemar test). `bench/make_card.py <round>` writes the system card, the Hugging Face model card and
the `[bench request]` issue text from that round's own files, into `bench/out/<round>/`.

## Log

**v1: public datasets alone make it worse.** Trained on public datasets aimed at the weak families
(ContractNLI, ANLI, TabFact, QuALITY, AQuA-RAT, date understanding): hard 0.694. They pull the model
toward their own short styles. Teacher-written documents are what matter.

**Teachers.** A shared remote server gave ~10 kept questions an hour, a local 3-bit build ~50. Rented
RTX PRO 6000 pods running vLLM with 48 streams give ~270-300 each, about $0.006 per kept question.
Two parser fixes mattered: a failed planning pass now retries without thinking, and probability
documents written with fractions (`1/15`) now parse, which took probability from 16 kept questions
to 170.

**v2 and v3: more of the same data doesn't reach the hard items.** Both train for 2 epochs at lr
2e-5. The held-out check rises (0.812, 0.824, 0.834), but the public hard tier stays at 0.739-0.757
and its families don't move.

**Two changes under test.** First, hard mining: the starting model answers about a quarter of our
teacher questions wrong and is unsure of another quarter (most often on dates and numbers,
probability and long policies); v4 trains on those, 3 epochs at lr 3e-5. Second, document length:
JevBench's hard documents reach ~3,700 tokens (a quarter are over 2,000), while ours stopped at
~1,400. Three pods now write 1,500-2,800-word documents, and training accepts 6,144 tokens; v5
includes them.

## Credits and licence

Starting weights, training scripts and runtime from [JevK5](https://github.com/allebee/jevk5)
v0.2 (Apache-2.0; `LICENSE-jevk5`, `NOTICE-jevk5`), whose one-pass readout and prompt come from
[SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT). Base model Qwen3.5-4B and teacher Qwen3.8-27B by
the Qwen team (Apache-2.0). Evaluated with [JevBench](https://github.com/fstandhartinger/jevbench)
(MIT). This repository: Apache-2.0 (`LICENSE`, `NOTICE`).
