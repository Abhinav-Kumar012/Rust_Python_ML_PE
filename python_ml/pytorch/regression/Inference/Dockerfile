# =============================================================================
# GPU Dockerfile (Aligned with CPU - regression setup)
# Only difference = CUDA base image
# =============================================================================

# -----------------------------------------------------------------------------
# Base Image (GPU support)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# -----------------------------------------------------------------------------
# Environment (same as CPU)
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# IMPORTANT: Must match CPU exactly (NFS path)
ENV PYTHONPATH=/external-libs/regression_env/lib/python3.12/site-packages

WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python 3.12 (to match CPU)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-distutils \
    python3-pip \
    libgomp1 \
    curl \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Copy application code (same as CPU)
# -----------------------------------------------------------------------------
COPY app.py ./

# -----------------------------------------------------------------------------
# Copy model (same as CPU)
# -----------------------------------------------------------------------------
COPY generated/model.pt ./generated/model.pt

# -----------------------------------------------------------------------------
# Security (recommended)
# -----------------------------------------------------------------------------
RUN useradd --create-home appuser
USER appuser

# -----------------------------------------------------------------------------
# Expose API port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Start FastAPI (same as CPU)
# -----------------------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# ========================== 🐳 BUILD COMMAND ================================
# =============================================================================

# docker build -f Dockerfile.gpu -t regression-inference:gpu .


# =============================================================================
# ========================== 🚀 RUN COMMAND (GPU) =============================
# =============================================================================

# docker run -d \
#   --name regression_container \
#   --gpus all \
#   -v /mnt/regression-libs:/external-libs:ro \
#   -e PYTHONPATH=/external-libs/regression_env/lib/python3.12/site-packages \
#   -p 8000:8000 \
#   regression-inference:gpu


# =============================================================================
# ========================== 🔍 DEBUG GPU ====================================
# =============================================================================

# docker exec -it regression_container python

# Inside Python:
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# =============================================================================
# ========================== ⚠️ REQUIREMENTS =================================
# =============================================================================

# 1. NFS must contain:
#    /mnt/regression-libs/regression_env/lib/python3.12/site-packages
#    → torch (GPU version - cu121)
#    → numpy, sklearn, etc.

# 2. Host must have GPU:
#    nvidia-smi

# 3. Install NVIDIA container toolkit:
#    sudo apt install nvidia-container-toolkit
#    sudo systemctl restart docker


# =============================================================================
# ========================== 🚨 COMMON ISSUES ===============================
# =============================================================================

# ❌ torch.cuda.is_available() == False
# → NFS has CPU torch

# ❌ ModuleNotFoundError
# → Wrong PYTHONPATH or mount

# ❌ GPU not detected
# → Missing --gpus all

# ❌ Model not loading
# → Check path: ./generated/model.pt


# =============================================================================
# ========================== ✅ FINAL NOTES ==================================
# =============================================================================

# ✔ CPU Dockerfile unchanged
# ✔ GPU Dockerfile now perfectly aligned
# ✔ Only difference = CUDA base image
# ✔ Dependencies come ONLY from NFS
# ✔ Model inside container