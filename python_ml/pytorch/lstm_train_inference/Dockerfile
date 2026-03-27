# =============================================================================
# NFS-Backed Inference Dockerfile — lstm_train_inference
# =============================================================================
#
# Architecture (Hybrid NFS + Docker):
#
#   ┌─────────────────────────┐        ┌────────────────────────────────────┐
#   │  NFS Server             │  NFS   │  Docker Host (Inference VM)        │
#   │  /srv/lstm-libs/        │◄──────►│  /mnt/lstm-libs/   (NFS mount)     │
#   │  (torch, numpy, …)      │        │                                    │
#   └─────────────────────────┘        │  docker run                        │
#                                      │    -v /mnt/lstm-libs:/external-libs│
#                                      │    -e PYTHONPATH=/external-libs    │
#                                      └────────────────────────────────────┘
#
# The Docker image contains ONLY:
#   • nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 base (CUDA runtime + cuDNN)
#   • Python 3.11  (installed via apt)
#   • Application source code
#
# Heavy Python libraries (torch-cu121, numpy, …) reside on the NFS server and
# are injected at runtime via a bind-mount + PYTHONPATH.
# Result: image stays lean; rebuilding never requires re-downloading PyTorch.
#
# ─────────────────────────────────────────────────────────────────────────────
# PRE-REQUISITES  (run once on the NFS server):
#
#   pip install \
#     --target /srv/lstm-libs \
#     --index-url https://download.pytorch.org/whl/cu121 \
#     torch numpy
#
# On the Docker host, mount the NFS export before running the container:
#
#   sudo mount -t nfs <NFS_SERVER_IP>:/srv/lstm-libs /mnt/lstm-libs
#
# ─────────────────────────────────────────────────────────────────────────────
# BUILD  (from repository root):
#
#   docker build \
#     -f python_ml/pytorch/lstm_train_inference/Dockerfile \
#     -t lstm-inference:latest \
#     .
#
# RUN  (requires nvidia-container-toolkit on the host):
#
#   docker run --rm --gpus all \
#     -v /mnt/lstm-libs:/external-libs:ro \
#     -v /mnt/lstm-models/lstm_train_python:/app/model/lstm_train_python:ro \
#     -e PYTHONPATH=/external-libs \
#     -e ARTIFACT_DIR=model/lstm_train_python \
#     lstm-inference:latest
#
# =============================================================================

# Single stage — no pip installs of heavy packages.
# IMPORTANT: Python version here MUST match the version used on the NFS server
# when running `pip install --target`.  A mismatch causes ImportError at runtime.
#
# Base image: NVIDIA CUDA 12.1 runtime + cuDNN 8.
# This provides all CUDA shared libraries that torch-cu121 needs at runtime
# (libcuda.so, libcudnn.so, etc.) without baking torch itself into the image.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11.  The CUDA base is Ubuntu 22.04 which ships Python 3.10;
# we add the deadsnakes PPA to get 3.11 and match the NFS library environment.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11 python3.11-distutils python3-pip \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# ── Application source code ───────────────────────────────────────────────────
# Only the inference pipeline source files are baked in — no wheels, no venv.
COPY python_ml/__init__.py                                    python_ml/
COPY python_ml/pytorch/__init__.py                            python_ml/pytorch/
COPY python_ml/pytorch/lstm_train_inference/__init__.py       python_ml/pytorch/lstm_train_inference/
COPY python_ml/pytorch/lstm_train_inference/dataset.py        python_ml/pytorch/lstm_train_inference/
COPY python_ml/pytorch/lstm_train_inference/model.py          python_ml/pytorch/lstm_train_inference/
COPY python_ml/pytorch/lstm_train_inference/training.py       python_ml/pytorch/lstm_train_inference/
COPY python_ml/pytorch/lstm_train_inference/inference.py      python_ml/pytorch/lstm_train_inference/

# ── Runtime volumes ───────────────────────────────────────────────────────────
# /external-libs  — bind-mounted from the NFS-backed path on the Docker host.
#                   Must contain a `pip --target` install of torch, numpy, etc.
# /app/model      — bind-mounted trained model artifacts (model.pt + config.json).
VOLUME ["/external-libs", "/app/model"]

# ── Runtime environment ───────────────────────────────────────────────────────
# PYTHONPATH points to the NFS-mounted site-packages inside the container.
ENV PYTHONPATH="/external-libs"
ENV ARTIFACT_DIR="model/lstm_train_python"

# Run as non-root for security.
RUN useradd --create-home appuser
USER appuser

CMD ["python", "-m", "python_ml.pytorch.lstm_train_inference.inference"]
