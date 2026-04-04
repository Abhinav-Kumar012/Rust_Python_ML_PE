import time
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO

from model import Model

app = FastAPI()
start_time = time.time()

# ----------------------------
# Device setup (GPU if available)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

print(f"Using device: {device}")

# ----------------------------
# Load model
# ----------------------------
try:
    print("Loading model...")
    checkpoint = torch.load("model.pt", map_location=device)
    print("Checkpoint loaded")

    model = Model(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)

except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ----------------------------
# Preprocess
# ----------------------------
def preprocess(img: Image.Image):
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return x

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "device": str(device)
    }

# ----------------------------
# Inference endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()

        # Read image
        data = await file.read()
        img = Image.open(BytesIO(data))

        # Preprocess + move to device
        x = preprocess(img).to(device)

        # Inference
        with torch.no_grad():
            logits = model(x)

        # (Optional) sync for accurate GPU timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000
        prediction = int(torch.argmax(logits, dim=1))

        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "device": str(device),
            "uptime_sec": time.time() - start_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Using GPU:", torch.cuda.is_available())