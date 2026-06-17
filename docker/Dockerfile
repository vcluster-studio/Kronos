# =============================================================================
# Kronos Training Docker Image (Multi-stage build)
# =============================================================================
# Build:
#   cd docker && docker build -t kronos:latest .
#
# Run:
#   docker run --gpus all \
#       -v ../finetune/data:/app/data \
#       -v ../outputs:/app/outputs \
#       -v ../pretrained:/app/pretrained \
#       -v ../logs:/app/logs \
#       kronos:latest train --use-block --model mini --epochs 50
# =============================================================================

# ---- Stage 1: Base ----
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 使用清华 apt 源（加速国内下载）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage 2: Dependencies ----
FROM base AS deps

# 使用清华 pip 源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies (separate layer for caching)
COPY ../requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ---- Stage 3: Application ----
FROM deps AS app

WORKDIR /app

# Copy source code (training/eval only)
COPY model/ ./model/
COPY finetune/predictor/ ./finetune/predictor/

# Entrypoint script
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Create directories for mounted volumes
RUN mkdir -p /app/data /app/models /app/tokenizers /app/pretrained /app/outputs /app/logs

ENV PYTHONPATH=/app

# Health check
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
CMD ["--help"]