# Training and evaluation box for jevforge: PyTorch + CUDA on the RTX 4080 Super (driver 591, CUDA 13).
FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
RUN pip install --no-cache-dir --break-system-packages "transformers>=5.17" "accelerate>=1.12" "huggingface-hub>=1.0" \
      peft datasets einops jinja2 numpy "flash-linear-attention>=0.5"
ENV HF_HOME=/work/.hf PYTHONPATH=/work:/work/training
WORKDIR /work
