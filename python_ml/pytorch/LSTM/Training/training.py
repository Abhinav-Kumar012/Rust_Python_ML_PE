import os
import sys
import json
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
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
# Utilities
# ==========================================================

def create_artifact_dir(artifact_dir: str) -> None:
    """Remove and recreate artifact directory — mirrors create_artifact_dir()."""
    shutil.rmtree(artifact_dir, ignore_errors=True)
    os.makedirs(artifact_dir, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================
# Training loop  (mirrors training.rs train())
# ==========================================================

def train(artifact_dir: str, config: TrainingConfig, device: torch.device) -> None:
    """
    Full training routine.

    Mirrors Rust train():
    - Adam optimizer with gradient clipping norm=1.0
    - MSE loss accumulated × batch_size → averaged over epoch items
    - Validation loop with no_grad
    - Print every 5 epochs  ((epoch+1) % 5 == 0)
    - Save config.json + model.pt
    """
    create_artifact_dir(artifact_dir)

    # Save config
    with open(os.path.join(artifact_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    set_seed(config.seed)

    # ── Datasets ────────────────────────────────────────────
    train_dataset = SequenceDataset(NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL)
    valid_dataset = SequenceDataset(NUM_SEQUENCES // 5, SEQ_LENGTH, NOISE_LEVEL)  # 20%

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    train_num_items = len(train_dataset)
    valid_num_items = len(valid_dataset)

    # ── Model + Optimizer ───────────────────────────────────
    model = LstmNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss(reduction="mean")

    print("Starting training...")

    # ── Epoch loop ──────────────────────────────────────────
    for epoch in range(1, config.num_epochs + 1):

        # ---------- Training ----------
        model.train()
        train_loss_acc = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)   # (B, seq_len, 1)
            targets   = targets.to(device)     # (B, 1)

            optimizer.zero_grad()
            output = model(sequences, None)    # (B, 1)
            loss = mse(output, targets)

            # Accumulate weighted loss (mirrors Rust: loss * batch_targets.dims()[0])
            train_loss_acc += loss.item() * targets.shape[0]

            loss.backward()
            # Gradient clipping norm=1.0  (mirrors GradientClippingConfig::Norm(1.0))
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = train_loss_acc / train_num_items

        # ---------- Validation ----------
        model.eval()
        valid_loss_acc = 0.0

        with torch.no_grad():
            for sequences, targets in valid_loader:
                sequences = sequences.to(device)
                targets   = targets.to(device)
                output = model(sequences, None)
                loss = mse(output, targets)
                valid_loss_acc += loss.item() * targets.shape[0]

        avg_valid_loss = valid_loss_acc / valid_num_items

        # Print every 5 epochs — mirrors (epoch + 1) % 5 == 0
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{config.num_epochs}, "
                f"Avg Loss {avg_train_loss:.4f}, "
                f"Avg Val Loss: {avg_valid_loss:.4f}"
            )

    # ── Save trained model ──────────────────────────────────
    torch.save(model.state_dict(), os.path.join(artifact_dir, "model.pt"))
    print(f"Training complete. Artifacts saved to: {artifact_dir}")


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    ARTIFACT_DIR = os.path.join("model", "lstm_train_python")

    train(
        artifact_dir=ARTIFACT_DIR,
        config=TrainingConfig(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
