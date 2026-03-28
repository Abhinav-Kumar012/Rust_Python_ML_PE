import os
import time
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from torch import nn

# ==========================================================
# Constants (must match training)
# ==========================================================

NUM_FEATURES = 13

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

# ==========================================================
# Model Definition (same as training)
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

device = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ==========================================================
# Load model
# ==========================================================

model = RegressionModel()

state_dict = torch.load(os.path.join(GENERATED_DIR, "model.pt"), map_location=device)

model.load_state_dict(state_dict)

model.eval()
model.to(device)


# ==========================================================
# Request schema
# ==========================================================

class HousingFeatures(BaseModel):

    features: list


# ==========================================================
# Preprocess
# ==========================================================

def preprocess(features):

    x = np.array(features, dtype=np.float32)

    if len(x) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features")

    x = torch.tensor(x).unsqueeze(0)

    return x


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
async def predict(data: HousingFeatures):

    t0 = time.perf_counter()

    x = preprocess(data.features).to(device)

    with torch.no_grad():
        prediction = model(x)

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "prediction": float(prediction.item()),
        "latency_ms": latency_ms,
        "uptime_sec": time.time() - start_time
    }


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)