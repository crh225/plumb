---
license: apache-2.0
language: [en]
library_name: transformers
base_model: alibiserikbay/JevK5
datasets: [crh225/plumb-decisions]
tags: [decision-model, decision-making, multiple-choice, classification, calibration, jevbench, qwen3.5]
---

# Plumb-4B

A 4B decision model: give it evidence, a question and 2-16 options, and it returns a probability for
every option from one forward pass, with no generated tokens. Probabilities use a single temperature
(T = 2.07) fitted on held-out decisions. Trained on hard decisions written and double-checked by
Qwen3.8-27B, with the questions it got wrong or was unsure of weighted up.

| Split | n | Accuracy | ECE | p50 latency |
|---|---:|---:|---:|---:|
| easy | 48 | 1.000 | 0.056 | 28 ms |
| standard | 72 | 0.972 | 0.152 | 28 ms |
| hard | 111 | 0.802 | 0.094 | 78 ms |

On JevBench hard, Plumb-4B scores 89/111 (80.2%) versus 82/111 (73.9%) for
JevK5 v0.2: 8 items fixed, 1 regressed. Easy and standard accuracy are unchanged.

Public JevBench items, JevBench's own runner, RTX 4080 Super. No JevBench item was used for training,
tuning, checkpoint selection or calibration. Aggregate results on the public JevBench set did inform
later training-recipe decisions; see https://github.com/crh225/plumb for the full disclosure, audits and per-item results.

Training data: [crh225/plumb-decisions](https://huggingface.co/datasets/crh225/plumb-decisions). GGUF for llama.cpp and Ollama: [crh225/plumb-4b-GGUF](https://huggingface.co/crh225/plumb-4b-GGUF).

## Use it

It runs with the open `jevk5` runtime (Apache-2.0), which reads the temperature from
`jevk5_config.json` in this repo. It needs a CUDA GPU with about 10 GB free (bf16).

```bash
pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0"
```

**From Python**, one decision per call:

```python
from jevk5 import JevK5

model = JevK5("crh225/plumb-4b")

model.decide(
    "Refunds need a receipt and a purchase within 30 days. "
    "The customer bought 12 days ago and has no receipt.",
    {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
)
# {'type': 'noul', 'noul': <probability of true>, 'confidence': ..., 'input_tokens': ...}

model.decide(
    "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
    {"type": "choice", "instructions": "What happened to the parcel?",
     "criteria": ["delivered", "misdelivered", "unknown"]},
)
# {'type': 'choice', 'choice': ..., 'probabilities': {...}, 'confidence': ..., 'input_tokens': ...}
```

Question types: `noul` (true/false), `choice` (2-16 options, a list or a `{key: description}` map)
and `score` (ordinal levels). Every answer is a full probability over the options, from one forward
pass, with no generated tokens.

**As a service**, with a TypeSafe-style request shape (several questions about one state per
request):

```bash
jevk5-serve --model crh225/plumb-4b --port 8090
curl -s localhost:8090/v1/systemone -d '{"state": "I was billed twice, please refund",
  "questions": {"refund": {"type": "noul", "instructions": "Asks for money back?"}}}'
```

## Credits and licence

Apache-2.0. Fine-tuned from [JevK5 v0.2](https://huggingface.co/alibiserikbay/JevK5) (Apache-2.0),
itself built on Qwen3.5-4B (Apache-2.0), with a readout from SemIf (MIT); see `NOTICE`.
