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

    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")

    artifact_dir = os.environ.get(
        "ARTIFACT_DIR",
        os.path.join("model", "lstm_train_python")
    )

    print(f"[INIT] Loading artifacts from: {artifact_dir}")

    # Load config
    config_path = os.path.join(artifact_dir, "config.json")
    print(f"[INIT] Loading config: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = TrainingConfig(**config_dict)

    # Load model
    model_path = os.path.join(artifact_dir, "model.pt")
    print(f"[INIT] Loading model: {model_path}")

    model = LstmNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    t_load_start = time.time()
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INIT] Model weights loaded in {(time.time() - t_load_start):.3f}s")

    model.eval()

    print(f"[INIT] Total startup time: {(time.time() - t0):.3f}s")

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
        print("\n🔥 [REQUEST] /predict called")

        if len(sequence) == 0:
            raise ValueError("Empty sequence")

        # Step 1: tensor creation
        t1 = time.perf_counter()
        x = torch.tensor(sequence, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(-1)
        print(f"[STEP] Tensor created: shape={x.shape} in {(time.perf_counter()-t1)*1000:.2f} ms")

        # Step 2: move to device
        t2 = time.perf_counter()
        x = x.to(device)
        print(f"[STEP] Moved to {device} in {(time.perf_counter()-t2)*1000:.2f} ms")

        # Step 3: inference
        t3 = time.perf_counter()
        with torch.no_grad():
            output = model(x, None)

        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"[STEP] Model forward pass took {(time.perf_counter()-t3)*1000:.2f} ms")

        # Step 4: output processing
        t4 = time.perf_counter()
        prediction = float(output.squeeze().item())
        print(f"[STEP] Output processing took {(time.perf_counter()-t4)*1000:.2f} ms")

        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"✅ [DONE] Total latency: {latency_ms:.2f} ms\n")

        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "uptime_sec": time.time() - start_time
        }

    except Exception as e:
        print(f"❌ [ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9050)