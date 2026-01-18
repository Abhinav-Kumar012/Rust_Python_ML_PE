# training.py
# PyTorch training pipeline aligned with the Burn implementation

import os
import json
import time
import platform
import sys
import random
import psutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

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
        # Dataset (80/10/10 exactly like Burn)
        # ----------------------------
        base_dataset = MNIST(
            ".",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

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
            lr=1e-4,
        )

        criterion = nn.CrossEntropyLoss()

        # ----------------------------
        # Training
        # ----------------------------
        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

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

                    logits = model(images)
                    loss = criterion(logits, targets)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

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
        # Test evaluation
        # ----------------------------
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        all_logits = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits, targets)

                test_loss += loss.item()

                preds = logits.argmax(dim=1)

                correct += (preds == targets).sum().item()
                total += targets.size(0)

                all_logits.append(logits.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        test_loss /= len(test_loader)
        test_acc = correct / total

        combined_logits = torch.cat(all_logits)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        probs = torch.softmax(combined_logits, dim=1).detach().numpy()
        # ----------------------------
        # Burn-equivalent metrics
        # ----------------------------
        precision = precision_score(
            targets, preds, average="macro"
        )

        recall = recall_score(
            targets, preds, average="macro"
        )

        f1 = f1_score(
            targets, preds, average="macro"
        )

        top5 = top_k_accuracy_score(
            targets.detach().numpy(),
            probs,
            k=5,
        )

        # ----------------------------
        # ONNX export (Track-2)
        # ----------------------------
        onnx_path = os.path.join(ARTIFACT_DIR, "model.onnx")

        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        print(f"ONNX model exported to {onnx_path}")

        # ----------------------------
        # System metrics
        # ----------------------------
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for _, entries in temps.items():
                    if entries:
                        cpu_temp = entries[0].current
                        break
        except Exception:
            cpu_temp = None

        metrics["test_metrics"] = {
            "accuracy": float(test_acc),
            "loss": float(test_loss),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top5_accuracy": float(top5),
        }

        metrics["system_metrics"] = {
            "cpu_percent": psutil.cpu_percent(interval=1.0),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_temperature": cpu_temp,
        }

        metrics["status"] = "success"

    except Exception as e:
        metrics["status"] = "failure"
        metrics["error_message"] = str(e)
        raise

    finally:
        metrics["total_duration_sec"] = time.time() - start_time

        with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")
