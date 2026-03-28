import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# Add parent (LSTM/) to sys.path so shared modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL,
    SequenceDataset, collate_fn,
)
from model import LstmNetwork
from config import TrainingConfig


# ==========================================================
# Inference  (mirrors inference.rs infer())
# ==========================================================

def infer(artifact_dir: str, device: torch.device) -> None:
    """
    Load a trained LstmNetwork and run inference on a validation dataset.

    Prints a predicted-vs-expected table (first 10 rows), mirroring the
    Polars df! output in Rust's infer().

    Parameters
    ----------
    artifact_dir : str
        Directory containing config.json and model.pt saved by training.py.
    device : torch.device
    """

    # ── Load config ─────────────────────────────────────────
    config_path = os.path.join(artifact_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. Run training first."
        )
    with open(config_path) as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)

    # ── Load model ──────────────────────────────────────────
    model_path = os.path.join(artifact_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

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

    # ── Build validation dataset (20% size — mirrors infer()) ──
    dataset = SequenceDataset(NUM_SEQUENCES // 5, SEQ_LENGTH, NOISE_LEVEL)

    # Put all items in a single batch (mirrors Rust's single batch call)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Run inference ────────────────────────────────────────
    with torch.no_grad():
        sequences, targets = next(iter(loader))
        sequences = sequences.to(device)   # (N, seq_len, 1)
        targets   = targets.to(device)     # (N, 1)

        predicted = model(sequences, None)  # (N, 1)

    predicted_vals = predicted.squeeze(1).cpu().tolist()   # [N]
    expected_vals  = targets.squeeze(1).cpu().tolist()     # [N]

    # ── Print table (mirrors Polars df![] output) ─────────────
    col_w = 14
    header = f"{'predicted':>{col_w}} {'expected':>{col_w}}"
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for pred, exp in zip(predicted_vals[:10], expected_vals[:10]):
        print(f"{pred:>{col_w}.6f} {exp:>{col_w}.6f}")
    print(sep)


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    # Allow the artifact directory to be configured via an environment variable
    # so the Docker container can override it at runtime (e.g. -e ARTIFACT_DIR=…).
    ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("model", "lstm_train_python"))

    infer(
        artifact_dir=ARTIFACT_DIR,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
