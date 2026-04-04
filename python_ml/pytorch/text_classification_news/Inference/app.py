from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import os
import contextlib
import time

# ==========================================================
# Config
# ==========================================================

class Config:
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    num_classes = 4
    vocab_size = 28996
    max_seq_len = 256
    norm_first = True

# ==========================================================
# Model
# ==========================================================

class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_token = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_pos = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            batch_first=True,
            norm_first=config.norm_first
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model) if config.norm_first else None
        )

        self.output = nn.Linear(config.d_model, config.num_classes)

    def forward(self, tokens, mask_pad=None):
        batch_size, seq_length = tokens.shape
        device = tokens.device

        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)

        embedding = (
            self.embedding_token(tokens) +
            self.embedding_pos(positions)
        ) / 2

        encoded = self.transformer(embedding, src_key_padding_mask=mask_pad)
        output = self.output(encoded)

        return output[:, 0, :]

# ==========================================================
# Schema
# ==========================================================

class PredictionRequest(BaseModel):
    text: str

# ==========================================================
# Globals
# ==========================================================

ml_models = {}
tokenizer = None
device = None

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Technology"
}

# ==========================================================
# Lifespan (load once)
# ==========================================================

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    model = TextClassificationModel(Config())

    model_path = "model/ag_news_model.pth"

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ml_models["text_classifier"] = model

    yield

    ml_models.clear()

# ==========================================================
# App
# ==========================================================

app = FastAPI(lifespan=lifespan)

# ==========================================================
# Endpoint
# ==========================================================

@app.post("/predict")
async def predict(request: PredictionRequest):
    if "text_classifier" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    model = ml_models["text_classifier"]

    encoding = tokenizer(
        [request.text],
        padding="max_length",
        truncation=True,
        max_length=Config.max_seq_len,
        return_tensors="pt"
    )

    tokens = encoding["input_ids"].to(device)
    mask_pad = (encoding["attention_mask"] == 0).to(device)

    with torch.no_grad():
        outputs = model(tokens, mask_pad)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probs, dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    class_id = predicted_class.item()
    class_name = CLASS_NAMES[class_id]

    latency_ms = (time.perf_counter() - t0) * 1000

    # ✅ CRITICAL FIX: match Locust expectation
    return {
        "prediction": class_name,
        "class_id": class_id,
        "confidence": confidence.item(),
        "latency_ms": latency_ms
    }

# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9050)  
    print("Using GPU:", torch.cuda.is_available())