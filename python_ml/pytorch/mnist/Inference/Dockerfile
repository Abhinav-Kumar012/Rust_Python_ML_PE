# ============================================================
# Stage 1 — Build Python wheels (compile once)
# ============================================================
FROM python:3.10 AS builder

# Disable .pyc files + enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directory where wheels will be built
WORKDIR /wheels

# Build dependencies (ONLY for compiling wheels)
# Needed for torch / torchvision / numpy native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Upgrade pip and build wheels
# --no-deps ensures dependencies are resolved once
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps -r requirements.txt


# ============================================================
# Stage 2 — Runtime image (LEAN)
# ============================================================
FROM python:3.10-slim

# Runtime environment flags
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Torch CPU tuning (match inference expectations)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Set working directory
WORKDIR /app

# ------------------------------------------------------------
# Install prebuilt wheels ONLY (no compilers here)
# ------------------------------------------------------------
COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# ------------------------------------------------------------
# Application files (small, copied last for cache efficiency)
# ------------------------------------------------------------
COPY app.py model.py model.pt ./

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with a single worker (deterministic latency)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
