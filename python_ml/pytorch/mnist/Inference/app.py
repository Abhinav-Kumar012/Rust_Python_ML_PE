import time
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

from model import Model

app = FastAPI()
start_time = time.time()

device = torch.device("cpu")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ----------------------------
# Load model
# ----------------------------
checkpoint = torch.load("model.pt", map_location=device)

model = Model(**checkpoint["model_args"])
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to(device)

# ----------------------------
# Preprocess (MATCH RUST)
# ----------------------------
def preprocess(img: Image.Image):
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return x

# ----------------------------
# Inference endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    data = await file.read()
    img = Image.open(BytesIO(data))

    x = preprocess(img).to(device)

    with torch.no_grad():
        logits = model(x)

    latency_ms = (time.perf_counter() - t0) * 1000
    prediction = int(torch.argmax(logits, dim=1))

    return {
        "prediction": prediction,
        "latency_ms": latency_ms,
        "uptime_sec": time.time() - start_time,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)