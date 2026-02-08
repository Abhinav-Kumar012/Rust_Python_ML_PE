# Stage 1: Builder
# Use a slim python image to minimize size
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (gcc might be needed for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.docker.txt .

# Install dependencies into ~/.local
# We use --user to make it easy to copy to the next stage
# We specify the PyTorch CUDA index to ensure we get the GPU-enabled version
RUN pip install --user --no-cache-dir -r requirements.docker.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Stage 2: Runtime
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy the python packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the application
# We assume the model file will be present (either copied or mounted context)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
