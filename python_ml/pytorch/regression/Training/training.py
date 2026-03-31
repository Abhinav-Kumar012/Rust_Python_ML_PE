import os
import json
import shutil
import time
import random
import platform
import sys
import urllib.request
import psutil
import torch
import numpy as np
from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Constants
# ==========================================================

NUM_FEATURES = 13

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

DATASET_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz"
RAW_DATA_FILE = os.path.join(OUTPUT_DIR, "boston_housing.npz")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_data.npz")
VALID_FILE = os.path.join(OUTPUT_DIR, "valid_data.npz")


# ==========================================================
# Dataset preparation
# ==========================================================

def prepare_dataset():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_FILE):
        print("Downloading Boston Housing dataset...")
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_FILE)
        print("Download complete.")

    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VALID_FILE):

        data = np.load(RAW_DATA_FILE)

        X = data["x"]
        y = data["y"]

        split = int(0.8 * len(X))

        X_train = X[:split]
        y_train = y[:split]

        X_valid = X[split:]
        y_valid = y[split:]

        np.savez(TRAIN_FILE, x=X_train, y=y_train)
        np.savez(VALID_FILE, x=X_valid, y=y_valid)

        print("Dataset prepared.")


# ==========================================================
# Dataset
# ==========================================================

class HousingDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    @staticmethod
    def train():

        prepare_dataset()

        data = np.load(TRAIN_FILE)

        return HousingDataset(data["x"], data["y"])

    @staticmethod
    def validation():

        prepare_dataset()

        data = np.load(VALID_FILE)

        return HousingDataset(data["x"], data["y"])


# ==========================================================
# Model
# ==========================================================

class RegressionModel(nn.Module):

    def __init__(self, hidden_size=64):

        super().__init__()

        self.input_layer = nn.Linear(NUM_FEATURES, hidden_size, bias=True)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x


# ==========================================================
# Config
# ==========================================================

@dataclass
class ExpConfig:

    num_epochs: int = 100
    num_workers: int = 2
    seed: int = 1337
    batch_size: int = 256
    learning_rate: float = 1e-3


# ==========================================================
# Utilities
# ==========================================================

def create_artifact_dir(artifact_dir):

    shutil.rmtree(artifact_dir, ignore_errors=True)

    os.makedirs(artifact_dir, exist_ok=True)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_system_utilization(device):
    """Collect CPU/GPU utilization metrics (reported cautiously)."""
    util = {}

    process = psutil.Process(os.getpid())
    util["cpu_percent"] = psutil.cpu_percent(interval=None)
    util["cpu_mem_mb"] = round(process.memory_info().rss / (1024 ** 2), 2)

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            util["cpu_temp_c"] = list(temps.values())[0][0].current
    except Exception:
        pass

    if device.type == "cuda":
        util["gpu_name"] = torch.cuda.get_device_name(device)
        util["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2)
        util["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2)
        util["gpu_max_mem_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        try:
            util["gpu_utilization_percent"] = torch.cuda.utilization(device)
        except Exception:
            pass

    return util


def check_for_nans(tensor, name="tensor"):
    """Check tensor for NaN/Inf values."""
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
    ax.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train Loss", markersize=3)
    ax.plot(epochs, [h["valid_loss"] for h in history], "r-o", label="Valid Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def save_convergence_plot(history, plots_dir):
    """Plot loss delta (convergence behavior)."""
    epochs = [h["epoch"] for h in history]
    train_deltas = [0.0] + [
        abs(history[i]["train_loss"] - history[i - 1]["train_loss"])
        for i in range(1, len(history))
    ]
    val_deltas = [0.0] + [
        abs(history[i]["valid_loss"] - history[i - 1]["valid_loss"])
        for i in range(1, len(history))
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_deltas, "b-s", label="Train Loss Δ", markersize=3)
    ax.plot(epochs, val_deltas, "r-s", label="Valid Loss Δ", markersize=3)
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
    ax.bar(epochs, times, color="steelblue", alpha=0.8, width=0.8)
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
        ax.plot(epochs, grad_norms, "m-o", label="Total Gradient Norm", markersize=3)
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

    ax1.plot(epochs, cpu_pct, "c-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CPU %")
    ax1.set_title("CPU Utilization (approximate)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, mem_mb, "orange", marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RSS Memory (MB)")
    ax2.set_title("Process Memory Usage")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "resource_utilization.png"), dpi=150)
    plt.close(fig)


def save_prediction_scatter(model, valid_loader, device, plots_dir):
    """Scatter plot of predicted vs actual values."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.numpy().flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, alpha=0.6, s=20, color="steelblue")
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (Validation Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "prediction_scatter.png"), dpi=150)
    plt.close(fig)


# ==========================================================
# Training step functions
# ==========================================================

def train_epoch(model, dataloader, optimizer, device):

    model.train()

    mse = nn.MSELoss()

    total_loss = 0

    start_time = time.time()

    last_grad_stats = {}
    nan_events = []
    batch_count = 0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = mse(outputs, targets)

        # NaN check
        loss_check = check_for_nans(loss, name=f"train_batch_{batch_count}")
        if not loss_check["is_stable"]:
            nan_events.append({"batch": batch_count, "detail": loss_check})

        loss.backward()

        # Gradient stats (sampled at last batch)
        last_grad_stats = compute_gradient_stats(model)

        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy().flatten())
        all_targets.append(targets.detach().cpu().numpy().flatten())
        batch_count += 1

    end_time = time.time()

    iteration_speed = len(dataloader) / (end_time - start_time) if (end_time - start_time) > 0 else 0

    # Regression metrics
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_targets)
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    mae = float(np.mean(np.abs(preds - tgts)))
    ss_res = np.sum((tgts - preds) ** 2)
    ss_tot = np.sum((tgts - np.mean(tgts)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    util = get_system_utilization(device)

    return {
        "loss": total_loss / len(dataloader),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "iteration_speed": iteration_speed,
        "epoch_time_sec": round(end_time - start_time, 3),
        "nan_events": nan_events,
        **last_grad_stats,
        **util,
    }


def validate_epoch(model, dataloader, device):

    model.eval()

    mse = nn.MSELoss()

    total_loss = 0
    nan_events = []
    batch_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)

            outputs = model(inputs)

            loss = mse(outputs, targets)

            loss_check = check_for_nans(loss, name=f"val_batch_{batch_count}")
            if not loss_check["is_stable"]:
                nan_events.append({"batch": batch_count, "detail": loss_check})

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            batch_count += 1

    # Regression metrics
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_targets)
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    mae = float(np.mean(np.abs(preds - tgts)))
    ss_res = np.sum((tgts - preds) ** 2)
    ss_tot = np.sum((tgts - np.mean(tgts)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    util = get_system_utilization(device)

    return {
        "loss": total_loss / len(dataloader),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nan_events": nan_events,
        **util,
    }


# ==========================================================
# Run (Equivalent to Rust run())
# ==========================================================

def run(artifact_dir, device):

    device = torch.device(device)

    create_artifact_dir(artifact_dir)
    plots_dir = os.path.join(artifact_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    config = ExpConfig()

    set_seed(config.seed)

    # Metrics container
    run_metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "config": config.__dict__,
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if torch.cuda.is_available():
        run_metrics["environment"]["gpu_name"] = torch.cuda.get_device_name(device)

    start_time = time.time()
    nan_crash_log = []

    try:
        train_dataset = HousingDataset.train()
        valid_dataset = HousingDataset.validation()

        print("Train Dataset Size:", len(train_dataset))
        print("Valid Dataset Size:", len(valid_dataset))

        # Model info
        model = RegressionModel().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run_metrics["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Device: {device}")
        print("=" * 60)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        history = []

        for epoch in range(config.num_epochs):

            train_metrics = train_epoch(model, train_loader, optimizer, device)
            valid_metrics = validate_epoch(model, valid_loader, device)

            # Collect NaN events
            for evt in train_metrics.pop("nan_events", []):
                nan_crash_log.append({"epoch": epoch + 1, "phase": "train", **evt})
            for evt in valid_metrics.pop("nan_events", []):
                nan_crash_log.append({"epoch": epoch + 1, "phase": "validation", **evt})

            lr = optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "valid_loss": valid_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_r2": valid_metrics["r2"],
                "iteration_speed": train_metrics["iteration_speed"],
                "epoch_time_sec": train_metrics["epoch_time_sec"],
                "learning_rate": lr,
                "nan_detected": len(nan_crash_log) > 0,
                # Gradient stats
                "grad_norm_mean": train_metrics.get("grad_norm_mean", 0),
                "grad_norm_max": train_metrics.get("grad_norm_max", 0),
                "grad_norm_total": train_metrics.get("grad_norm_total", 0),
                # System utilization
                "cpu_percent": train_metrics.get("cpu_percent", 0),
                "cpu_mem_mb": train_metrics.get("cpu_mem_mb", 0),
                "cpu_temp_c": train_metrics.get("cpu_temp_c", 0),
            }

            # GPU metrics if available
            if "gpu_mem_allocated_mb" in train_metrics:
                metrics["gpu_mem_allocated_mb"] = train_metrics["gpu_mem_allocated_mb"]
                metrics["gpu_mem_reserved_mb"] = train_metrics.get("gpu_mem_reserved_mb", 0)

            history.append(metrics)

            print(
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"Train Loss: {metrics['train_loss']:.6f} | "
                f"Valid Loss: {metrics['valid_loss']:.6f} | "
                f"R²={metrics['valid_r2']:.4f} | "
                f"RMSE={metrics['valid_rmse']:.4f} | "
                f"time={metrics['epoch_time_sec']:.3f}s"
            )

        # Convergence metrics
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["valid_loss"] for h in history]

        convergence = {
            "final_train_loss": train_losses[-1],
            "final_valid_loss": val_losses[-1],
            "best_valid_loss": min(val_losses),
            "best_valid_loss_epoch": int(np.argmin(val_losses) + 1),
            "loss_decreased_monotonically": all(
                train_losses[i] >= train_losses[i + 1]
                for i in range(len(train_losses) - 1)
            ),
            "overfit_detected": val_losses[-1] > min(val_losses) * 1.1,
            "final_loss_delta": abs(train_losses[-1] - train_losses[-2]) if len(train_losses) > 1 else 0.0,
        }

        # Training performance
        total_training_time = time.time() - start_time
        epoch_times = [h["epoch_time_sec"] for h in history]

        performance = {
            "total_training_time_sec": round(total_training_time, 3),
            "avg_time_per_epoch_sec": round(np.mean(epoch_times), 3),
            "min_epoch_time_sec": round(min(epoch_times), 3),
            "max_epoch_time_sec": round(max(epoch_times), 3),
            "avg_iteration_speed": round(
                np.mean([h["iteration_speed"] for h in history]), 2
            ),
        }

        # Stability
        stability = {
            "total_nan_events": len(nan_crash_log),
            "nan_free_training": len(nan_crash_log) == 0,
            "convergence": convergence,
            "nan_crash_log": nan_crash_log if nan_crash_log else "none",
        }

        # Global summary (aggregated across all epochs)
        grad_norms = [h.get("grad_norm_total", 0.0) for h in history]
        grad_means = [h.get("grad_norm_mean", 0.0) for h in history]
        grad_maxes = [h.get("grad_norm_max", 0.0) for h in history]
        cpu_pcts = [h.get("cpu_percent", 0.0) for h in history]
        cpu_mems = [h.get("cpu_mem_mb", 0.0) for h in history]
        iter_speeds = [h["iteration_speed"] for h in history]
        train_rmses = [h["train_rmse"] for h in history]
        train_maes = [h["train_mae"] for h in history]
        train_r2s = [h["train_r2"] for h in history]
        valid_rmses = [h["valid_rmse"] for h in history]
        valid_maes = [h["valid_mae"] for h in history]
        valid_r2s = [h["valid_r2"] for h in history]

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
            "valid_loss": {
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "first": val_losses[0],
                "last": val_losses[-1],
            },
            "train_rmse": {
                "min": float(np.min(train_rmses)),
                "max": float(np.max(train_rmses)),
                "mean": float(np.mean(train_rmses)),
                "first": train_rmses[0],
                "last": train_rmses[-1],
            },
            "valid_rmse": {
                "min": float(np.min(valid_rmses)),
                "max": float(np.max(valid_rmses)),
                "mean": float(np.mean(valid_rmses)),
                "first": valid_rmses[0],
                "last": valid_rmses[-1],
            },
            "train_mae": {
                "min": float(np.min(train_maes)),
                "max": float(np.max(train_maes)),
                "mean": float(np.mean(train_maes)),
                "first": train_maes[0],
                "last": train_maes[-1],
            },
            "valid_mae": {
                "min": float(np.min(valid_maes)),
                "max": float(np.max(valid_maes)),
                "mean": float(np.mean(valid_maes)),
                "first": valid_maes[0],
                "last": valid_maes[-1],
            },
            "train_r2": {
                "min": float(np.min(train_r2s)),
                "max": float(np.max(train_r2s)),
                "mean": float(np.mean(train_r2s)),
                "first": train_r2s[0],
                "last": train_r2s[-1],
            },
            "valid_r2": {
                "min": float(np.min(valid_r2s)),
                "max": float(np.max(valid_r2s)),
                "mean": float(np.mean(valid_r2s)),
                "first": valid_r2s[0],
                "last": valid_r2s[-1],
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
            "epochs_completed": len(history),
            "nan_epochs": sum(1 for h in history if h.get("nan_detected", False)),
        }

        # Add GPU global stats if available
        gpu_mems = [h.get("gpu_mem_allocated_mb", None) for h in history]
        if any(g is not None for g in gpu_mems):
            gpu_mems_valid = [g for g in gpu_mems if g is not None]
            global_summary["system_utilization"]["gpu_mem_allocated_mb_min"] = float(np.min(gpu_mems_valid))
            global_summary["system_utilization"]["gpu_mem_allocated_mb_max"] = float(np.max(gpu_mems_valid))
            global_summary["system_utilization"]["gpu_mem_allocated_mb_mean"] = round(float(np.mean(gpu_mems_valid)), 2)

        run_metrics["global_summary"] = global_summary
        run_metrics["training_performance"] = performance
        run_metrics["training_stability"] = stability
        run_metrics["epoch_history"] = history
        run_metrics["status"] = "success"

        # Save config
        with open(os.path.join(artifact_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=4)

        with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
            json.dump(run_metrics, f, indent=4, default=str)

        model_dir = os.path.join(artifact_dir, "model_pytorch_regression")
        os.makedirs(model_dir, exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(model_dir, "ag_news_model.pth")
        )

        # Generate plots
        print("\nGenerating training plots...")
        save_loss_curves(history, plots_dir)
        save_convergence_plot(history, plots_dir)
        save_epoch_timing_plot(history, plots_dir)
        save_gradient_norm_plot(history, plots_dir)
        save_resource_utilization_plot(history, plots_dir)
        save_prediction_scatter(model, valid_loader, device, plots_dir)
        print(f"Plots saved to {plots_dir}/")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total time       : {total_training_time:.2f}s")
        print(f"  Avg epoch time   : {performance['avg_time_per_epoch_sec']:.3f}s")
        print(f"  Final train loss : {convergence['final_train_loss']:.6f}")
        print(f"  Final valid loss : {convergence['final_valid_loss']:.6f}")
        print(f"  Best valid loss  : {convergence['best_valid_loss']:.6f} (epoch {convergence['best_valid_loss_epoch']})")
        print(f"  NaN-free         : {stability['nan_free_training']}")
        print(f"  Monotonic conv.  : {convergence['loss_decreased_monotonically']}")
        print("=" * 60)

    except Exception as e:
        run_metrics["status"] = "error"
        run_metrics["error"] = str(e)
        import traceback
        run_metrics["traceback"] = traceback.format_exc()
        print(f"\n✗ Training failed: {e}")
        raise

    finally:
        run_metrics["total_duration_sec"] = time.time() - start_time

        metrics_path = os.path.join(artifact_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(run_metrics, f, indent=4, default=str)

        print(f"Metrics saved to {metrics_path}")
        print("Training complete. Artifacts saved to:", artifact_dir)


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    run(
        artifact_dir=os.path.join(OUTPUT_DIR, "artifacts"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )