FROM python:3.10-slim

# ----------------------------
# Environment hygiene
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python dependencies
# ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Application + model
# ----------------------------
COPY app.py .
COPY model.onnx .

EXPOSE 8000

# ----------------------------
# Start inference server
# ----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
