FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

COPY requirements.docker.txt .

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.docker.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]