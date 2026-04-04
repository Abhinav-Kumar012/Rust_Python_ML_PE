import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException
from typing import List
import contextlib
import time

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LstmNetwork
from config import TrainingConfig

# ==========================================================
# Globals
# ==========================================================

app = FastAPI()

model = None
device = None
start_time = time.time()

# ==========================================================
# Lifespan (load once)
# ==========================================================

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    artifact_dir = os.environ.get(
        "ARTIFACT_DIR",
        os.path.join("model", "lstm_train_python")
    )

    # Load config
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = TrainingConfig(**config_dict)

    # Load model
    model_path = os.path.join(artifact_dir, "model.pt")

    model = LstmNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    yield

# attach lifespan
app = FastAPI(lifespan=lifespan)

# ==========================================================
# Endpoint
# ==========================================================

@app.post("/predict")
async def predict(sequence: List[float]):
    try:
        t0 = time.perf_counter()

        if len(sequence) == 0:
            raise ValueError("Empty sequence")

        # Convert → tensor (1, seq_len, 1)
        x = torch.tensor(sequence, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            output = model(x, None)

        if device.type == "cuda":
            torch.cuda.synchronize()

        prediction = float(output.squeeze().item())

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
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