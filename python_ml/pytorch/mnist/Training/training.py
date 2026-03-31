# training.py — PyTorch .pt export (Burn-aligned)
# Enhanced with comprehensive training metrics & stability reporting

import os
import json
import time
import platform
import sys
import random
import math
import psutil
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
    confusion_matrix,
)

# Add current directory to sys.path so local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (
    SimpleMnistDataset,
    mnist_collate,
)
from model import Model

# Store model in this directory
ARTIFACT_DIR = "./artifacts/pytorch_mnist_pt"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")


def normalize(x):
    return (x - 0.1307) / 0.3081


# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_system_utilization(device):
    """Collect CPU/GPU utilization metrics (reported cautiously)."""
    util = {}

    # CPU metrics
    process = psutil.Process(os.getpid())
    util["cpu_percent"] = psutil.cpu_percent(interval=None)
    util["cpu_mem_mb"] = process.memory_info().rss / (1024 ** 2)

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            util["cpu_temp_c"] = list(temps.values())[0][0].current
    except Exception:
        pass

    # GPU metrics (if available)
    if device.type == "cuda":
        util["gpu_name"] = torch.cuda.get_device_name(device)
        util["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        util["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        util["gpu_max_mem_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        try:
            util["gpu_utilization_percent"] = torch.cuda.utilization(device)
        except Exception:
            pass

    return util


def check_for_nans(tensor, name="tensor"):
    """Check tensor for NaN/Inf values. Returns dict with stability info."""
    has_nan = bool(torch.isnan(tensor).any())
    has_inf = bool(torch.isinf(tensor).any())
    return {
        "name": name,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_stable": not has_nan and not has_inf,
    }


def compute_gradient_stats(model):
    """Compute gradient statistics for stability monitoring."""
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm(2).item())
    if grad_norms:
        return {
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "grad_norm_min": float(np.min(grad_norms)),
            "grad_norm_total": float(np.sqrt(sum(g**2 for g in grad_norms))),
        }
    return {}


# ==========================================================
# Plotting (saved to separate plots/ folder)
# ==========================================================

def save_loss_curves(history, plots_dir):
    """Plot train/validation loss curves."""
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, [h["val_loss"] for h in history], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def save_accuracy_curve(history, plots_dir):
    """Plot validation accuracy curve."""
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [h["val_accuracy"] for h in history], "g-o", label="Val Accuracy", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def save_convergence_plot(history, plots_dir):
    """Plot loss delta (convergence behavior)."""
    epochs = [h["epoch"] for h in history]
    train_deltas = [0.0] + [
        abs(history[i]["train_loss"] - history[i - 1]["train_loss"])
        for i in range(1, len(history))
    ]
    val_deltas = [0.0] + [
        abs(history[i]["val_loss"] - history[i - 1]["val_loss"])
        for i in range(1, len(history))
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_deltas, "b-s", label="Train Loss Δ", markersize=4)
    ax.plot(epochs, val_deltas, "r-s", label="Val Loss Δ", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss Change|")
    ax.set_title("Convergence Behavior (Loss Delta Per Epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "convergence_behavior.png"), dpi=150)
    plt.close(fig)


def save_epoch_timing_plot(history, plots_dir):
    """Plot time per epoch."""
    epochs = [h["epoch"] for h in history]
    times = [h["epoch_time_sec"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(epochs, times, color="steelblue", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Per Epoch")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "epoch_timing.png"), dpi=150)
    plt.close(fig)


def save_gradient_norm_plot(history, plots_dir):
    """Plot gradient norm across epochs."""
    epochs = [h["epoch"] for h in history]
    grad_norms = [h.get("grad_norm_total", 0.0) for h in history]

    if any(g > 0 for g in grad_norms):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, grad_norms, "m-o", label="Total Gradient Norm", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm (L2)")
        ax.set_title("Gradient Norm Across Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "gradient_norms.png"), dpi=150)
        plt.close(fig)


def save_resource_utilization_plot(history, plots_dir):
    """Plot CPU/memory utilization."""
    epochs = [h["epoch"] for h in history]
    cpu_pct = [h.get("cpu_percent", 0) for h in history]
    mem_mb = [h.get("cpu_mem_mb", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(epochs, cpu_pct, "c-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CPU %")
    ax1.set_title("CPU Utilization (approximate)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, mem_mb, "orange", marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RSS Memory (MB)")
    ax2.set_title("Process Memory Usage")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "resource_utilization.png"), dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(targets, preds, plots_dir):
    """Plot confusion matrix for test evaluation."""
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


# ==========================================================
# Main training function
# ==========================================================

def run(device):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Configuration matching Rust TrainingConfig
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1.0e-4
    SEED = 42
    NUM_WORKERS = 4

    # ----------------------------
    # Reproducibility
    # ----------------------------
    set_seed(SEED)

    # ----------------------------
    # Metrics container
    # ----------------------------
    metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "num_workers": NUM_WORKERS,
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if torch.cuda.is_available():
        metrics["environment"]["gpu_name"] = torch.cuda.get_device_name(device)

    start_time = time.time()
    nan_crash_log = []
    epoch_history = []

    try:
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
            generator=torch.Generator().manual_seed(SEED),
        )

        train_ds = SimpleMnistDataset(train_raw)
        valid_ds = SimpleMnistDataset(valid_raw)
        test_ds = SimpleMnistDataset(test_raw)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=mnist_collate,
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=mnist_collate,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=mnist_collate,
        )

        # ----------------------------
        # Model
        # ----------------------------
        model = Model().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Device: {device}")
        print(f"Training samples: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")
        print("=" * 60)

        # ----------------------------
        # Training loop
        # ----------------------------
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()

            # ---- Training phase ----
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_batches = 0
            epoch_nan_detected = False
            last_grad_stats = {}

            for images, targets in train_loader:
                images = normalize(images.to(device))
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)

                # NaN / Inf detection
                loss_check = check_for_nans(loss, name=f"epoch_{epoch}_train_loss")
                if not loss_check["is_stable"]:
                    epoch_nan_detected = True
                    nan_crash_log.append({
                        "epoch": epoch,
                        "phase": "train",
                        "batch": train_batches,
                        "detail": loss_check,
                    })
                    print(f"  ⚠ NaN/Inf detected at epoch {epoch}, batch {train_batches}")

                loss.backward()

                # Gradient clipping (mirrors Rust GradientClippingConfig)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Gradient stats (sampled at last batch per epoch)
                last_grad_stats = compute_gradient_stats(model)

                optimizer.step()

                train_loss += loss.item()
                train_preds = logits.argmax(dim=1)
                train_correct += (train_preds == targets).sum().item()
                train_total += targets.size(0)
                train_batches += 1

            train_loss /= train_batches
            train_acc = train_correct / train_total

            # ---- Validation phase ----
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_batches = 0

            with torch.no_grad():
                for images, targets in valid_loader:
                    images = normalize(images.to(device))
                    targets = targets.to(device)

                    logits = model(images)
                    loss = criterion(logits, targets)

                    # NaN check on validation
                    loss_check = check_for_nans(loss, name=f"epoch_{epoch}_val_loss")
                    if not loss_check["is_stable"]:
                        nan_crash_log.append({
                            "epoch": epoch,
                            "phase": "validation",
                            "batch": val_batches,
                            "detail": loss_check,
                        })

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    val_batches += 1

            val_loss /= val_batches
            acc = correct / total

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start

            # System utilization snapshot
            util = get_system_utilization(device)

            # Iteration speed
            iteration_speed = train_batches / epoch_time if epoch_time > 0 else 0.0

            # Record epoch history
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": acc,
                "epoch_time_sec": round(epoch_time, 3),
                "iteration_speed": round(iteration_speed, 2),
                "nan_detected": epoch_nan_detected,
                **last_grad_stats,
                **util,
            }
            epoch_history.append(epoch_record)

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={acc:.4f} | "
                f"time={epoch_time:.2f}s | "
                f"iter/s={iteration_speed:.1f}"
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
        test_accuracy = float((preds == targets).float().mean())

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
        print(f"\nModel saved to {pt_path}")

        # ----------------------------
        # Compute convergence metrics
        # ----------------------------
        train_losses = [h["train_loss"] for h in epoch_history]
        val_losses = [h["val_loss"] for h in epoch_history]

        convergence = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_val_loss_epoch": int(np.argmin(val_losses) + 1),
            "loss_decreased_monotonically": all(
                train_losses[i] >= train_losses[i + 1]
                for i in range(len(train_losses) - 1)
            ),
            "overfit_detected": val_losses[-1] > min(val_losses) * 1.1,
            "final_loss_delta": abs(train_losses[-1] - train_losses[-2]) if len(train_losses) > 1 else 0.0,
        }

        # ----------------------------
        # Training performance summary
        # ----------------------------
        total_training_time = time.time() - start_time
        epoch_times = [h["epoch_time_sec"] for h in epoch_history]

        performance = {
            "total_training_time_sec": round(total_training_time, 3),
            "avg_time_per_epoch_sec": round(np.mean(epoch_times), 3),
            "min_epoch_time_sec": round(min(epoch_times), 3),
            "max_epoch_time_sec": round(max(epoch_times), 3),
            "avg_iteration_speed": round(
                np.mean([h["iteration_speed"] for h in epoch_history]), 2
            ),
        }

        # ----------------------------
        # Training stability summary
        # ----------------------------
        stability = {
            "total_nan_events": len(nan_crash_log),
            "nan_free_training": len(nan_crash_log) == 0,
            "convergence": convergence,
            "nan_crash_log": nan_crash_log if nan_crash_log else "none",
        }

        # ----------------------------
        # Global summary (aggregated across all epochs)
        # ----------------------------
        grad_norms = [h.get("grad_norm_total", 0.0) for h in epoch_history]
        grad_means = [h.get("grad_norm_mean", 0.0) for h in epoch_history]
        grad_maxes = [h.get("grad_norm_max", 0.0) for h in epoch_history]
        cpu_pcts = [h.get("cpu_percent", 0.0) for h in epoch_history]
        cpu_mems = [h.get("cpu_mem_mb", 0.0) for h in epoch_history]
        iter_speeds = [h["iteration_speed"] for h in epoch_history]
        train_accs = [h["train_accuracy"] for h in epoch_history]
        val_accs = [h["val_accuracy"] for h in epoch_history]

        global_summary = {
            "train_loss": {
                "min": float(np.min(train_losses)),
                "max": float(np.max(train_losses)),
                "mean": float(np.mean(train_losses)),
                "std": float(np.std(train_losses)),
                "first": train_losses[0],
                "last": train_losses[-1],
                "total_reduction": train_losses[0] - train_losses[-1],
                "reduction_pct": round((1 - train_losses[-1] / train_losses[0]) * 100, 2) if train_losses[0] != 0 else 0.0,
            },
            "val_loss": {
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "first": val_losses[0],
                "last": val_losses[-1],
            },
            "train_accuracy": {
                "min": float(np.min(train_accs)),
                "max": float(np.max(train_accs)),
                "mean": float(np.mean(train_accs)),
                "std": float(np.std(train_accs)),
                "first": train_accs[0],
                "last": train_accs[-1],
            },
            "val_accuracy": {
                "min": float(np.min(val_accs)),
                "max": float(np.max(val_accs)),
                "mean": float(np.mean(val_accs)),
                "std": float(np.std(val_accs)),
                "first": val_accs[0],
                "last": val_accs[-1],
            },
            "gradient_norms": {
                "total_norm_min": float(np.min(grad_norms)),
                "total_norm_max": float(np.max(grad_norms)),
                "total_norm_mean": float(np.mean(grad_norms)),
                "total_norm_std": float(np.std(grad_norms)),
                "per_param_mean_min": float(np.min(grad_means)),
                "per_param_mean_max": float(np.max(grad_means)),
                "per_param_max_min": float(np.min(grad_maxes)),
                "per_param_max_max": float(np.max(grad_maxes)),
            },
            "timing": {
                "total_training_time_sec": performance["total_training_time_sec"],
                "epoch_time_min": performance["min_epoch_time_sec"],
                "epoch_time_max": performance["max_epoch_time_sec"],
                "epoch_time_mean": performance["avg_time_per_epoch_sec"],
                "epoch_time_std": round(float(np.std(epoch_times)), 3),
            },
            "system_utilization": {
                "cpu_percent_min": float(np.min(cpu_pcts)),
                "cpu_percent_max": float(np.max(cpu_pcts)),
                "cpu_percent_mean": round(float(np.mean(cpu_pcts)), 2),
                "cpu_mem_mb_min": float(np.min(cpu_mems)),
                "cpu_mem_mb_max": float(np.max(cpu_mems)),
                "cpu_mem_mb_mean": round(float(np.mean(cpu_mems)), 2),
            },
            "iteration_speed": {
                "min": float(np.min(iter_speeds)),
                "max": float(np.max(iter_speeds)),
                "mean": round(float(np.mean(iter_speeds)), 2),
            },
            "epochs_completed": len(epoch_history),
            "nan_epochs": sum(1 for h in epoch_history if h.get("nan_detected", False)),
        }

        # Add GPU global stats if available
        gpu_mems = [h.get("gpu_mem_allocated_mb", None) for h in epoch_history]
        if any(g is not None for g in gpu_mems):
            gpu_mems_valid = [g for g in gpu_mems if g is not None]
            global_summary["system_utilization"]["gpu_mem_allocated_mb_min"] = float(np.min(gpu_mems_valid))
            global_summary["system_utilization"]["gpu_mem_allocated_mb_max"] = float(np.max(gpu_mems_valid))
            global_summary["system_utilization"]["gpu_mem_allocated_mb_mean"] = round(float(np.mean(gpu_mems_valid)), 2)

        # ----------------------------
        # Assemble metrics
        # ----------------------------
        metrics["test_metrics"] = {
            "accuracy": test_accuracy,
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top5_accuracy": float(top5),
        }
        metrics["global_summary"] = global_summary
        metrics["training_performance"] = performance
        metrics["training_stability"] = stability
        metrics["epoch_history"] = epoch_history
        metrics["status"] = "success"

        # ----------------------------
        # Generate plots
        # ----------------------------
        print("\nGenerating training plots...")
        save_loss_curves(epoch_history, PLOTS_DIR)
        save_accuracy_curve(epoch_history, PLOTS_DIR)
        save_convergence_plot(epoch_history, PLOTS_DIR)
        save_epoch_timing_plot(epoch_history, PLOTS_DIR)
        save_gradient_norm_plot(epoch_history, PLOTS_DIR)
        save_resource_utilization_plot(epoch_history, PLOTS_DIR)
        save_confusion_matrix_plot(targets.numpy(), preds.numpy(), PLOTS_DIR)
        print(f"Plots saved to {PLOTS_DIR}/")

        # ----------------------------
        # Print summary
        # ----------------------------
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total time       : {total_training_time:.2f}s")
        print(f"  Avg epoch time   : {performance['avg_time_per_epoch_sec']:.2f}s")
        print(f"  Test accuracy    : {test_accuracy:.4f}")
        print(f"  Test precision   : {precision:.4f}")
        print(f"  Test recall      : {recall:.4f}")
        print(f"  Test F1          : {f1:.4f}")
        print(f"  Top-5 accuracy   : {top5:.4f}")
        print(f"  NaN-free         : {stability['nan_free_training']}")
        print(f"  Converged        : {convergence['loss_decreased_monotonically']}")
        print("=" * 60)

    except Exception as e:
        metrics["status"] = "error"
        metrics["error"] = str(e)
        import traceback
        metrics["traceback"] = traceback.format_exc()
        print(f"\n✗ Training failed: {e}")
        raise

    finally:
        metrics["total_duration_sec"] = time.time() - start_time

        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4, default=str)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(device)
