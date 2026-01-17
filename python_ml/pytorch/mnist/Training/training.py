# training.py
# Burn-equivalent training loop (no extras)

import os
import json
import time
import platform
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from python_ml.pytorch.mnist.Training.data import (
    SimpleMnistDataset,
    mnist_collate,
)
from python_ml.pytorch.mnist.Training.model import Model

ARTIFACT_DIR = "/tmp/burn_python_mnist"


def run(device):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
        },
    }

    start_time = time.time()

    try:
        # ----------------------------
        # Reproducibility
        # ----------------------------
        torch.manual_seed(42)
        random.seed(42)

        # ----------------------------
        # Dataset (exact Burn split)
        # ----------------------------
        base_dataset = MNIST(".", train=True, download=True)

        total_len = len(base_dataset)
        train_len = int(0.8 * total_len)
        valid_len = int(0.1 * total_len)
        test_len = total_len - train_len - valid_len

        train_raw, valid_raw, test_raw = random_split(
            base_dataset,
            [train_len, valid_len, test_len],
            generator=torch.Generator().manual_seed(42),
        )

        train_ds = SimpleMnistDataset(train_raw)
        valid_ds = SimpleMnistDataset(valid_raw)
        test_ds = SimpleMnistDataset(test_raw)

        train_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            collate_fn=mnist_collate,
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            collate_fn=mnist_collate,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            collate_fn=mnist_collate,
        )

        # ----------------------------
        # Model
        # ----------------------------
        model = Model().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1.0,
        )

        criterion = nn.CrossEntropyLoss()

        # ----------------------------
        # Training
        # ----------------------------
        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches

            # ----------------------------
            # Validation
            # ----------------------------
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, targets in valid_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    output = model(images)
                    loss = criterion(output, targets)

                    val_loss += loss.item()
                    preds = output.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            val_loss /= len(valid_loader)
            acc = correct / total

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"acc={acc:.4f}"
            )

        # ----------------------------
        # Test
        # ----------------------------
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                output = model(images)
                loss = criterion(output, targets)

                test_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_loss /= len(test_loader)
        test_acc = correct / total

        print(
            f"Test Set Evaluation: "
            f"loss={test_loss:.4f}, acc={test_acc:.4f}"
        )

        metrics["test_metrics"] = {
            "accuracy": test_acc,
            "loss": test_loss,
        }

        metrics["status"] = "success"

    except Exception as e:
        metrics["status"] = "failure"
        metrics["error_message"] = str(e)
        raise

    finally:
        metrics["total_duration"] = time.time() - start_time

        with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")
