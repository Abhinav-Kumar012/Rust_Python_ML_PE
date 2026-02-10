# training.py — PyTorch .pt export (Burn-aligned)

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
from torch.utils.data import Subset


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

# Store model in this directory
ARTIFACT_DIR = "./artifacts/pytorch_mnist_pt"


def normalize(x):
    return (x - 0.1307) / 0.3081


def run(device):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Configuration matching Rust TrainingConfig
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1.0e-4
    SEED = 42
    NUM_WORKERS = 4 

    # ----------------------------
    # Reproducibility
    # ----------------------------
    # Rust: B::seed(&device, config.seed);
    # Rust: let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ----------------------------
    # Model & optimizer
    # ----------------------------
    # Rust: config.model.init::<B>(&device)
    model = Model(dropout=0.5).to(device)

    # Rust: config.optimizer.init() -> AdamConfig default is Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5, 
    )

    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # Dataset
    # ----------------------------
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    base_train = MNIST(".", train=True, download=True, transform=transform)

    # Rust: items.shuffle(&mut rng); -> shuffle indices
    # Rust: Split 80:10:10
    
    total_len = len(base_train)
    indices = list(range(total_len))
    random.Random(SEED).shuffle(indices) # Match Rust shuffling with seed 42
    
    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len 

    train_indices = indices[:train_len]
    valid_indices = indices[train_len : train_len + valid_len]
    test_indices = indices[train_len + valid_len :]

    train_ds = Subset(base_train, train_indices)
    valid_ds = Subset(base_train, valid_indices)
    test_ds = Subset(base_train, test_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True, # Rust: dataloader_train...shuffle(config.seed)
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=True, # Rust: dataloader_test...shuffle(config.seed) (Used for validation)
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=True, # Rust: dataloader_test_final...shuffle(config.seed)
        num_workers=NUM_WORKERS,
    )

    # ----------------------------
    # Training loop
    # ----------------------------
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
        # Dataset
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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # ----------------------------
        # Training
        # ----------------------------
        for epoch in range(10):
            model.train()
            train_loss = 0.0

            for images, targets in train_loader:
                images = normalize(images.to(device))
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
                    images = normalize(images.to(device))
                    targets = targets.to(device)

                    logits = model(images)
                    loss = criterion(logits, targets)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            acc = correct / total

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss/len(valid_loader):.4f}, "
                f"acc={acc:.4f}"
            )

        # ----------------------------
        # Test evaluation
        # ----------------------------
        model.eval()

        all_logits, all_preds, all_targets = [], [], []

        with torch.no_grad():
            for images, targets in test_loader:
                images = normalize(images.to(device))
                targets = targets.to(device)

                logits = model(images)
                preds = logits.argmax(dim=1)

                all_logits.append(logits.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        logits = torch.cat(all_logits)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        probs = torch.softmax(logits, dim=1).numpy()

        precision = precision_score(targets, preds, average="macro")
        recall = recall_score(targets, preds, average="macro")
        f1 = f1_score(targets, preds, average="macro")
        top5 = top_k_accuracy_score(targets.numpy(), probs, k=5)

        # ----------------------------
        # Save .pt
        # ----------------------------
        pt_path = os.path.join(ARTIFACT_DIR, "model.pt")

        torch.save(
            {
                "model_state": model.state_dict(),
                "model_args": {
                    "num_classes": 10,
                    "hidden_size": 512,
                    "dropout": 0.5,
                },
            },
            pt_path,
        )

        print(f"Model saved to {pt_path}")

        # ----------------------------
        # Metrics
        # ----------------------------
        metrics["test_metrics"] = {
            "accuracy": float((preds == targets).float().mean()),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top5_accuracy": float(top5),
        }

        metrics["status"] = "success"

    finally:
        metrics["total_duration_sec"] = time.time() - start_time

        with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")
