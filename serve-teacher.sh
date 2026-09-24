#!/usr/bin/env bash
# Serve Qwen3.8-27B (Unsloth UD-Q3_K_XL, 13.1 GB) on the local GPU with llama.cpp, as the teacher.
#   ./serve-teacher.sh [slots] [context]   -> OpenAI-compatible API on http://localhost:8090
# Fully on the GPU (the Q4 build is 18 GB and spills to the CPU at ~12 tok/s). The KV cache is
# 8-bit so two long slots fit beside the weights in 16 GB.
slots="${1:-2}"; ctx="${2:-32768}"
docker rm -f jevy-teacher >/dev/null 2>&1
MSYS_NO_PATHCONV=1 docker run -d --name jevy-teacher --gpus all -p 8090:8080 \
  -v "C:/Users/Chris/models/qwen3.8-27b:/models" ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-UD-Q3_K_XL.gguf --alias qwen3.8-27b-local \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c "$ctx" --parallel "$slots" \
  --jinja --reasoning-format deepseek --host 0.0.0.0 --port 8080
