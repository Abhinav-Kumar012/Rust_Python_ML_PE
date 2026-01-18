import time
import os
import psutil
import numpy as np
import onnxruntime as ort
import torch
from torchvision.datasets import MNIST
from fastapi import FastAPI
from torchvision import transforms


# ----------------------------
# App + process setup
# ----------------------------
app = FastAPI()
process = psutil.Process(os.getpid())
start_time = time.time()

# ----------------------------
# Load ONNX model once (cold start)
# ----------------------------
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

# ----------------------------
# Load MNIST (inference only)
# ----------------------------
mnist = MNIST(
    root=".",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# ----------------------------
# Normalization (must match training)
# ----------------------------
def preprocess(image):
    # image: Tensor [1, 28, 28] in range [0,1]
    x = (image - 0.1307) / 0.3081
    return x.unsqueeze(0).numpy().astype(np.float32)
          # [1, 1, 28, 28]

# ----------------------------
# Metrics helper
# ----------------------------
def system_metrics():
    return {
        "rss_mb": process.memory_info().rss / (1024 ** 2),
        "cpu_percent": process.cpu_percent(interval=None),
        "uptime_sec": time.time() - start_time,
    }

# ----------------------------
# Inference endpoint
# ----------------------------
@app.get("/infer")
def infer(index: int = 0):
    """
    Inference on a real MNIST test image.

    Query param:
    - index: MNIST test index (0–9999)
    """

    image, label = mnist[index]

    input_tensor = preprocess(image)

    t0 = time.perf_counter()
    outputs = session.run(None, {"input": input_tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    prediction = int(np.argmax(outputs[0]))

    return {
        "prediction": prediction,
        "ground_truth": int(label),
        "latency_ms": latency_ms,
        **system_metrics()
    }

# ----------------------------
# Warmup endpoint (optional)
# ----------------------------
@app.get("/warmup")
def warmup():
    dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
    session.run(None, {"input": dummy})
    return {"status": "warmed"}
