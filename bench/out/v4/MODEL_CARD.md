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

License Apache-2.0 (derived from JevK5 v0.2 and Qwen3.5-4B, both Apache-2.0; readout from SemIf, MIT).
