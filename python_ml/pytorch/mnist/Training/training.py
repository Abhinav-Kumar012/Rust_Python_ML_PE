# training.py
# Exact translation of training.rs (Python)

import os
import json
import time
import platform
import sys
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import MNIST

from python_ml.pytorch.mnist.Training.data import (
    Transform,
    MnistMapper,
    MappedMnistDataset,
    SampledDataset,
    mnist_collate,
)
from python_ml.pytorch.mnist.Training.model import Model

ARTIFACT_DIR = "/tmp/burn-example-mnist"


def generate_idents(num_samples_base):
    idents = []
    for shear in [None, Transform.Shear]:
        for scale in [None, Transform.Scale]:
            for rot in [None, Transform.Rotation]:
                for tr in [None, Transform.Translate]:
                    current = [t for t in [shear, scale, rot, tr] if t is not None]
                    if not current:
                        ident = []
                    elif len(current) == 4:
                        ident = "ALL"
                    else:
                        ident = current

                    size = None
                    if num_samples_base is not None:
                        size = num_samples_base * max(1, len(current))

                    idents.append((ident, size))
    return idents


def build_dataset(base, idents, train=True):
    datasets = []
    for ident, size in idents:
        if ident == "ALL":
            transforms = [
                Transform.Translate,
                Transform.Shear,
                Transform.Scale,
                Transform.Rotation,
            ]
        else:
            transforms = ident

        mapper = MnistMapper().transform(transforms)
        mapped = MappedMnistDataset(base, mapper)

        if size is not None:
            datasets.append(SampledDataset(mapped, size, with_replacement=train))
        else:
            datasets.append(mapped)

    return ConcatDataset(datasets)


def run(device):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "error_message": None,
        "total_duration": 0,
        "epoch_metrics": [],
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cpu_info": platform.processor(),
        },
        "reproducibility": {"seed": 42},
        "artifact_size": {},
    }

    start_time = time.time()

    try:
        # ----------------------------
        # Reproducibility
        # ----------------------------
        torch.manual_seed(42)
        random.seed(42)

        # ----------------------------
        # Model & optimizer
        # ----------------------------
        model = Model().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0,
            weight_decay=5e-5,
        )

        criterion = nn.CrossEntropyLoss()

        # ----------------------------
        # LR scheduler
        # ----------------------------
        def lr_lambda(step):
            if step < 2000:
                return (1e-8 + step * (1.0 - 1e-8) / 2000)
            elif step < 4000:
                return 0.5 * (1 + math.cos(math.pi * (step - 2000) / 2000))
            else:
                return max(
                    1e-6,
                    1e-2 - (step - 4000) * (1e-2 - 1e-6) / 10000,
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ----------------------------
        # Dataset
        # ----------------------------
        base_train = MNIST(".", train=True, download=True)

        total_size = len(base_train)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_plain, valid_plain, test_plain = torch.utils.data.random_split(
            base_train,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_idents = generate_idents(10_000)
        valid_idents = generate_idents(None)
        test_idents = generate_idents(None)

        train_ds = build_dataset(train_plain, train_idents, train=True)
        valid_ds = build_dataset(valid_plain, valid_idents, train=False)
        test_ds = build_dataset(test_plain, test_idents, train=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=256,
            shuffle=True,
            num_workers=8,
            collate_fn=mnist_collate,
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            collate_fn=mnist_collate,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            collate_fn=mnist_collate,
        )

        # ----------------------------
        # Training loop
        # ----------------------------
        best_loss = float("inf")
        patience = 5
        bad_epochs = 0
        step = 0

        for epoch in range(20):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_batches = 0

            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, targets in valid_loader:
                    images, targets = images.to(device), targets.to(device)
                    output = model(images)
                    loss = criterion(output, targets)

                    val_loss += loss.item()
                    preds = output.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            val_loss /= len(valid_loader)
            acc = correct / total
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"acc={acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

            metrics["epoch_metrics"].append(
                {
                    "epoch": epoch,
                    "time": epoch_time,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": acc,
                }
            )

            if val_loss < best_loss:
                best_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), f"{ARTIFACT_DIR}/model.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping")
                    break

        # ----------------------------
        # Test + ONNX export
        # ----------------------------
        model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/model.pt"))
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                loss = criterion(output, targets)

                test_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_loss /= len(test_loader)
        test_acc = correct / total

        print(
            f"Test Set Evaluation: loss={test_loss:.4f}, acc={test_acc:.4f}"
        )

        metrics["test_metrics"] = {
            "accuracy": test_acc,
            "loss": test_loss,
        }

        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        onnx_path = f"{ARTIFACT_DIR}/model.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        print(f"Model exported to {onnx_path}")
        metrics["status"] = "success"

    except Exception as e:
        metrics["status"] = "failure"
        metrics["error_message"] = str(e)
        raise

    finally:
        end_time = time.time()
        metrics["total_duration"] = end_time - start_time

        try:
            if os.path.exists(f"{ARTIFACT_DIR}/model.pt"):
                metrics["artifact_size"]["model.pt"] = os.path.getsize(
                    f"{ARTIFACT_DIR}/model.pt"
                )
            if os.path.exists(f"{ARTIFACT_DIR}/model.onnx"):
                metrics["artifact_size"]["model.onnx"] = os.path.getsize(
                    f"{ARTIFACT_DIR}/model.onnx"
                )
        except Exception:
            pass

        with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")
