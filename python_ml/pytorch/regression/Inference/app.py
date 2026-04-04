import os
import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn

# ==========================================================
# Constants
# ==========================================================

NUM_FEATURES = 13   # keep as-is (we will pad)

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

# ==========================================================
# Model Definition
# ==========================================================

class RegressionModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.input_layer = nn.Linear(NUM_FEATURES, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# ==========================================================
# FastAPI setup
# ==========================================================

app = FastAPI()
start_time = time.time()

# GPU support (works with NVIDIA docker)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

print(f"Using device: {device}")

# ==========================================================
# Load model
# ==========================================================

model = RegressionModel()

state_dict = torch.load(
    os.path.join(GENERATED_DIR, "model.pt"),
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# ==========================================================
# Request schema (MATCHES LOCUST)
# ==========================================================

class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    MedHouseVal: float

# ==========================================================
# Preprocess (convert dict → tensor)
# ==========================================================

def preprocess(data: HousingFeatures):
    x = np.array([
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude,
        data.MedHouseVal
    ], dtype=np.float32)

    # Pad to 13 features if needed
    if len(x) < NUM_FEATURES:
        x = np.pad(x, (0, NUM_FEATURES - len(x)), mode='constant')

    x = torch.tensor(x).unsqueeze(0)
    return x

# ==========================================================
# Health check
# ==========================================================

@app.get("/")
def health():
    return {"status": "ok", "device": str(device)}

# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
async def predict(data: HousingFeatures):
    try:
        t0 = time.perf_counter()

        x = preprocess(data).to(device)

        with torch.no_grad():
            prediction = model(x)

        # sync for accurate GPU timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "prediction": float(prediction.item()),
            "latency_ms": latency_ms,
            "device": str(device),
            "uptime_sec": time.time() - start_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9050)  
    print("Using GPU:", torch.cuda.is_available())