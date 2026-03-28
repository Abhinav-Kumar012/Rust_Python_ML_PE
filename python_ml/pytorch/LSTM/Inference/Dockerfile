# =============================================================================
# GPU-Enabled LSTM Inference API (NFS-backed)
# =============================================================================

# CUDA runtime (no heavy torch inside image)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent Python cache + clean logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python (match NFS environment EXACTLY)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11 python3.11-distutils python3-pip \
        libgomp1 curl \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY app.py ./

# ❌ DO NOT copy model inside image (use NFS instead)
# COPY lstm_train_python/model.pt ./generated/model.pt

# -----------------------------------------------------------------------------
# Runtime mounts
# -----------------------------------------------------------------------------
# /external-libs → NFS (torch, numpy, etc.)
# /app/model     → NFS model directory
VOLUME ["/external-libs", "/app/model"]

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONPATH="/external-libs"
ENV MODEL_PATH="/app/model/lstm_train_python/model.pt"

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# -----------------------------------------------------------------------------
# Start API server
# -----------------------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# Command to execute

# docker run --rm --gpus all \
#   -v /mnt/lstm-libs:/external-libs:ro \
#   -v /mnt/lstm-models:/app/model:ro \
#   -e PYTHONPATH=/external-libs \
#   -e MODEL_PATH=/app/model/lstm_train_python/model.pt \
#   -p 8000:8000 \
#   lstm-inference:gpu