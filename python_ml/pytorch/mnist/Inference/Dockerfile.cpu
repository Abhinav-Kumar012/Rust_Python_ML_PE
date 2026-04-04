FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENV PYTHONPATH=/external-libs/ml_env/lib/python3.12/site-packages

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install fastapi==0.110.0 uvicorn==0.29.0 python-multipart==0.0.9

COPY app.py model.py model.pt ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -t MNIST_infer_image .
# docker run   -v /mnt/ml-libs:/external-libs   -p 8000:8000   MNIST_infer_image
