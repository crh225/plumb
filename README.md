# jevy

An open, one-pass typed-decision model trained to beat
[JevK5](https://github.com/allebee/jevk5) on [JevBench](https://github.com/fstandhartinger/jevbench).

JevK5 v0.2 is JevBench's best open entrant: #2 of 76 on v1.4 (62.04, against Jev 1.13.0's 63.29).
It is Qwen3.5-4B with a LoRA distilled from Qwen3.6-27B, read out in one forward pass as a softmax
over the answer letters' logits. jevy uses the same shape (so the same speed and cost) and aims
to win on intelligence and calibration with a stronger teacher and more, broader data.

## The target (JevBench v1.2 public items, JevK5 v0.2, its own published run)

| Split | n | JevK5 v0.2 |
|---|---:|---:|
| easy | 48 | 1.000 |
| original (standard) | 72 | 0.958 |
| hard (public half) | 111 | 0.739 |
| hard-tier ECE | | 0.066 |

The official score also needs JevBench's private sealed set, which only its maintainer runs.

## Rules we hold to

- **No JevBench item is trained on, tuned on, or used to pick a checkpoint.** The 231 public
  items are only a scoreboard at the end of a run. Checkpoints and the calibration temperature
  are chosen on our own held-out data (teacher questions from domains training never sees, plus
  held-out splits of the replay datasets).
- **Same readout and size as JevK5**, so speed and cost stay level and any gain is quality.

## Pipeline

1. `training/teacher.py` (from JevK5, Apache-2.0): a teacher writes realistic documents with
   hard typed questions and answers each twice with thinking; only questions both answers agree
   on are kept. Our teacher is Qwen3.8-27B (on a shared remote server and rented GPUs), a newer model than JevK5's
   Qwen3.6-27B. Changes: extra request headers and streamed responses, for a gateway behind a
   proxy that drops silent requests after ~100 s.
2. `training/build_replay.py` (new): human-labelled public datasets (MNLI, WANLI, BoolQ,
   banking77, ARC, CommonsenseQA, MMLU-Pro) in the decision format, so easy decisions don't regress.
3. `training/build_train.py` and `training/lora.py` (from JevK5): assemble the training set and
   LoRA-tune Qwen3.5-4B with cross-entropy on the option-letter logits; merge the adapter.
4. Calibrate one temperature on held-out teacher questions; evaluate on the public 231.

Training and evaluation run in the `Dockerfile` image (PyTorch 2.14, CUDA 13) on an RTX 4080
Super (16 GB). `teacher-env.sh` (git-ignored) points the teacher at its server.

## Log

**Baseline (reproduced here).** JevK5 v0.2 bf16 on the 231 public items, JevBench's own runner,
RTX 4080 Super: easy 1.000 (ECE 0.039), standard 0.972 (0.137), hard 0.739 (0.077), 27-79 ms.
`results/jevk5-bf16/`.

**Teacher capacity.** The teacher server takes 1-2 concurrent requests (8 at once all fail) and
sits behind Cloudflare, whose ~100 s cut-off kills queued requests (fixed by streaming) and whose
bot protection later started refusing the Python and Node clients with 403. We don't disguise
traffic to get past bot protection; the route needs an exception or a direct address from the
server's owner. At 1-2 requests the teacher yields ~25-50 kept questions an hour.

**Teacher-free hard data.** While the teacher is limited, `build_hard_replay.py` turns public,
human-labelled datasets into decisions aimed at JevK5's weakest families: ContractNLI (long
policies, "not mentioned"), ANLI (traps), TabFact (table lookups), QuALITY (long reading),
AQuA-RAT (arithmetic) and BIG-Bench Hard date understanding. The overlap scan finds only generic
legal boilerplate shared with one public item.

**Teacher family mix.** Teacher documents are weighted toward JevK5's weakest hard-tier families
(its own README: dates and numbers 0.47, long policies 0.58, judging 0.76, multi-step 0.78) and
toward probability, which JevK5 barely trained on. Families are JevBench's published categories,
not its items.

**v1: public hard data alone makes it worse.** Continued from JevK5 on 5,000 replay + 6,663
hard-replay items (one epoch, lr 2e-5), temperature 1.41 fitted on dev. Dev accuracy rose 0.716 ->
0.754, but only on the sources it trained on; the hand-written hard set stayed 50/65. JevBench
public: easy 1.000 (ECE 0.027), standard 0.972 (0.144), **hard 0.694** against JevK5's 0.739. The
public datasets pull it toward their own styles (short NLI, tables, arithmetic) and away from long,
trap-laden documents. Discarded; teacher-written documents are what matter.

**Local teacher.** Qwen3.8-27B (the same model as the remote teacher) in Unsloth's UD-Q3_K_XL
quantization, 13.1 GB, fully on the 4080 Super under llama.cpp (`serve-teacher.sh`): 41 tok/s
for one request, 73 tok/s for two, and no proxy in the way.

**Rented teachers.** Two RTX PRO 6000 pods on Runpod ($1.69/h each) serve Qwen3.8-27B-FP8 in
vLLM with 48 streams each: ~250-320 kept questions an hour per pod (about $0.007 per kept
question), against ~9/h from the shared remote teacher and ~50/h from the local Q3 build.
Probability documents are dropped on the pods (exact distributions almost never survive two
solves), and the planning pass gets 24k tokens (date documents ran out of room at 14k).

**v2: first win over JevK5.** Continued from JevK5 on 495 kept teacher questions + as many plain
replay items, 2 epochs, lr 2e-5; temperature 1.47 fitted on 131 held-out teacher questions.
Held-out dev 0.812 -> 0.824 (teacher questions 0.802 -> 0.840; hand-written hard unchanged at
51/65). JevBench public: easy 1.000 (ECE 0.038), standard 0.972 (0.141), **hard 0.757 (ECE
0.034)** against JevK5's 0.739 (0.077). On the hard tier v2 fixes 2 of JevK5's misses and breaks
none (multi-step lookups 15 -> 16, trade-offs 5 -> 6), and its calibration error is less than
half. Two items of 111 is within noise; the calibration gain is the stronger signal. Dates and
numbers (7/15) and long policies (11/19) are unchanged and are what more teacher data has to move.

## Credits

Training scripts and runtime from [JevK5](https://github.com/allebee/jevk5) (Apache-2.0; see
`LICENSE-jevk5` and `NOTICE-jevk5`), whose one-pass readout and prompt come from
[SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT). Base model Qwen3.5-4B by the Qwen team
(Apache-2.0). Evaluated with JevBench (MIT).
