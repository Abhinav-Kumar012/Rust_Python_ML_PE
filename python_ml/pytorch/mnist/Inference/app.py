import time
import os
import psutil
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# App + process setup
# ----------------------------
app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Normalization (must match training)
# ----------------------------
def preprocess(image_bytes):
    # Open image from bytes
    img = Image.open(BytesIO(image_bytes)).convert('L') # Grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Preprocess
    img_np = np.array(img).astype(np.float32)
    x = img_np / 255.0
    x = (x - 0.1307) / 0.3081
    
    # Add batch and channel dims: [1, 1, 28, 28]
    x = np.expand_dims(x, axis=0) 
    x = np.expand_dims(x, axis=0)
    
    return x.astype(np.float32)

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
# Inference endpoint for File Upload
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    input_tensor = preprocess(contents)

    t0 = time.perf_counter()
    outputs = session.run(None, {"input": input_tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    prediction = int(np.argmax(outputs[0]))

    return {
        "prediction": prediction,
        "meta": {
            "latency_ms": latency_ms,
            "resources": system_metrics(),
            "security_context": {
                "service_version": "0.1.0-pytorch",
                "model_version": "v1-onnx"
            }
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}
