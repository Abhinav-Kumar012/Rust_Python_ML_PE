# training.py
# Exact translation of training.rs (Python)

import os
import json
import time
import platform
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from python_ml.pytorch.mnist.Training.model import Model

ARTIFACT_DIR = "./artifacts"

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
        "epoch_metrics": [],
        "setup_info": {"seed": SEED},
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
        }
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS): # 0 to 9
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
            
            train_loss += loss.item()
            train_batches += 1
            
        train_loss /= train_batches

        # Validation
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
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={acc:.4f}, time={epoch_time:.2f}s")

        metrics["epoch_metrics"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "time": epoch_time
        })

    # ----------------------------
    # Test Evaluation (Matches Rust 'Test Evaluation')
    # ----------------------------
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total_test_batches = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            
            test_loss += loss.item()
            
            preds = output.argmax(dim=1)
            batch_correct = (preds == targets).sum().item()
            batch_acc = batch_correct / targets.size(0)
            
            test_acc += batch_acc
            total_test_batches += 1

    if total_test_batches > 0:
        test_loss /= total_test_batches
        test_acc /= total_test_batches

    print(f"Test Set Evaluation: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # ----------------------------
    # Artifacts & Export
    # ----------------------------
    
    # Save Model state
    torch.save(model.state_dict(), f"{ARTIFACT_DIR}/model.pt")
    
    # Export ONNX
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

    # Save Metrics
    total_duration = time.time() - start_time
    model_size = os.path.getsize(f"{ARTIFACT_DIR}/model.pt") if os.path.exists(f"{ARTIFACT_DIR}/model.pt") else 0

    metrics["status"] = "success"
    metrics["total_duration_sec"] = total_duration
    metrics["avg_epoch_duration_sec"] = total_duration / NUM_EPOCHS
    metrics["artifact_metrics"] = {"model_size_bytes": model_size}
    metrics["test_metrics"] = {
        "accuracy": test_acc,
        "loss": test_loss
    }

    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")
