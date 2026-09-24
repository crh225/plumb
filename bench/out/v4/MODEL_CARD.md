---
license: apache-2.0
base_model: alibiserikbay/JevK5
tags: [decision-model, classification, calibration, jevbench, qwen3.5]
---

# Plumb-4B

A 4B decision model: give it evidence, a question and 2-16 options, and it returns a calibrated
probability for every option from one forward pass, with no generated tokens. Continued from
[JevK5 v0.2](https://huggingface.co/alibiserikbay/JevK5) on hard decisions written and
double-checked by Qwen3.8-27B. Runs with JevK5's runtime unchanged (temperature 1.95 in
`jevk5_config.json`).

| Split | n | Plumb-4B | JevK5 v0.2 | ECE | JevK5 ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 48 | **1.000** | 1.000 | 0.044 | 0.039 | 28 ms | 29 ms |
| standard | 72 | **0.972** | 0.972 | 0.139 | 0.137 | 28 ms | 30 ms |
| hard | 111 | **0.775** | 0.739 | 0.051 | 0.077 | 79 ms | 507 ms |

Public JevBench items, JevBench's own runner, RTX 4080 Super. No JevBench item was used for
training, tuning or selection. Details, audits and per-item results: https://github.com/crh225/jevy.

## Use it

Plumb-4B has the same format as JevK5 v0.2, so JevK5's runtime loads it unchanged and reads its
calibration temperature from `jevk5_config.json` in the model repo. It needs a CUDA GPU with about
10 GB free (bf16).

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

License Apache-2.0 (derived from JevK5 v0.2 and Qwen3.5-4B, both Apache-2.0; readout from SemIf, MIT).
