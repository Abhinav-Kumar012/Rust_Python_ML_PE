import time
import torch
from fastapi import FastAPI
from torchvision.datasets import MNIST
from torchvision import transforms

from model import Model


# ----------------------------
# App setup
# ----------------------------
app = FastAPI()
start_time = time.time()

device = torch.device("cpu")

# Stabilize latency
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ----------------------------
# Load PyTorch model (.pt)
# ----------------------------
checkpoint = torch.load("model.pt", map_location=device)

model = Model(**checkpoint["model_args"])
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to(device)


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
# Normalization (same as training)
# ----------------------------
def preprocess(image):
    # image: Tensor [1, 28, 28] in range [0,1]
    x = (image - 0.1307) / 0.3081
    return x.unsqueeze(0)   # [1, 1, 28, 28]


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

    x = preprocess(image).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    latency_ms = (time.perf_counter() - t0) * 1000

    prediction = int(torch.argmax(logits, dim=1))

    return {
        "prediction": prediction,
        "ground_truth": int(label),
        "latency_ms": latency_ms,
        "uptime_sec": time.time() - start_time,
    }


# ----------------------------
# Warmup endpoint
# ----------------------------
@app.get("/warmup")
def warmup():
    dummy = torch.zeros(1, 1, 28, 28)
    with torch.no_grad():
        model(dummy)
    return {"status": "warmed"}
