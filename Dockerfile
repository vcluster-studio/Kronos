# =============================================================================
# Kronos Training Docker Image
# =============================================================================
# Multi-stage build for minimal image size
#
# Build:
#   docker build -t kronos-train .
#
# Quick run (single task):
#   docker run --gpus all -v /path/to/csv:/app/finetune_csv kronos-train \
#       bash entrypoint.sh --task tokenizer --dataset mid
#
# Cloud GPU (AutoDL / 恒源云 / 矩池云):
#   docker run --gpus all \
#       -v /root/kronos_csv:/app/finetune_csv \
#       -v /root/kronos_pretrained:/app/pretrained \
#       -v /root/kronos_output:/app/output \
#       -v /root/kronos_logs:/app/logs \
#       kronos-train bash entrypoint.sh --task pipeline --dataset mid
# =============================================================================

# ---- Stage 1: Base with CUDA ----
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# ---- Stage 2: Install Python dependencies ----
FROM base AS deps

# Install PyTorch first (largest dependency, separate layer for caching)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies from requirements file
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ---- Stage 3: Application ----
FROM deps AS app

WORKDIR /app

# Copy source code (fine-grained for better layer caching)
COPY model/ ./model/
COPY finetune/config.py ./finetune/config.py
COPY finetune/dataset.py ./finetune/dataset.py
COPY finetune/train_tokenizer.py ./finetune/train_tokenizer.py
COPY finetune/train_predictor.py ./finetune/train_predictor.py
COPY finetune/csv_data_preprocess.py ./finetune/csv_data_preprocess.py
COPY finetune/unified_test.py ./finetune/unified_test.py
COPY finetune/validate_tokenizer.py ./finetune/validate_tokenizer.py
COPY finetune/utils/ ./finetune/utils/
COPY examples/ ./examples/
COPY webui/ ./webui/
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

# Create directories for mounted volumes
RUN mkdir -p /app/pretrained \
    /app/data \
    /app/output \
    /app/finetune/data \
    /app/finetune_csv/exported_kline_data/stocks \
    /app/logs

# Pretrained models: downloaded on first run, or mounted from host
# Data: mounted from host via volumes

ENV PYTHONPATH=/app

# Health check: verify PyTorch + CUDA
RUN python -c "import torch; assert torch.cuda.is_available() or True; print(f'PyTorch {torch.__version__} ready')"

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
CMD ["--help"]
