# ============================================================
# Stage 1 — build wheels
# ============================================================
FROM python:3.10 AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /wheels

# Build dependencies (for psutil, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps -r requirements.txt


# ============================================================
# Stage 2 — runtime image
# ============================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Torch CPU tuning
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install prebuilt wheels
COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Application files
COPY app.py .
COPY model.py .
COPY model.pt .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
