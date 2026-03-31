import os
import sys
import json
import shutil
import random
import time
import platform
import psutil
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ax.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, [h["val_loss"] for h in history], "r-o", label="Val Loss", markersize=4)
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


# ==========================================================
# Training loop  (mirrors training.rs train())
# ==========================================================

def train(artifact_dir: str, config: TrainingConfig, device: torch.device) -> None:
    """
    Full training routine with comprehensive metrics.

    Mirrors Rust train():
    - Adam optimizer with gradient clipping norm=1.0
    - MSE loss accumulated × batch_size → averaged over epoch items
    - Validation loop with no_grad
    - Print every 5 epochs  ((epoch+1) % 5 == 0)
    - Save config.json + model.pt + metrics + plots
    """
    create_artifact_dir(artifact_dir)
    plots_dir = os.path.join(artifact_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save config
    with open(os.path.join(artifact_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    set_seed(config.seed)

    # ── Metrics container ──────────────────────────────────
    run_metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "config": asdict(config),
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
    epoch_history = []

    try:
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

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run_metrics["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Device: {device}")
        print(f"Training samples: {train_num_items}, Validation: {valid_num_items}")
        print("Starting training...")
        print("=" * 60)

        # ── Epoch loop ──────────────────────────────────────────
        for epoch in range(1, config.num_epochs + 1):
            epoch_start = time.time()

            # ---------- Training ----------
            model.train()
            train_loss_acc = 0.0
            train_batches = 0
            epoch_nan_detected = False
            last_grad_stats = {}
            train_preds_all = []
            train_targets_all = []

            for sequences, targets in train_loader:
                sequences = sequences.to(device)   # (B, seq_len, 1)
                targets   = targets.to(device)     # (B, 1)

                optimizer.zero_grad()
                output = model(sequences, None)    # (B, 1)
                loss = mse(output, targets)

                # NaN / Inf check
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

                # Accumulate weighted loss (mirrors Rust: loss * batch_targets.dims()[0])
                train_loss_acc += loss.item() * targets.shape[0]
                train_preds_all.append(output.detach().cpu())
                train_targets_all.append(targets.detach().cpu())

                loss.backward()

                # Gradient stats (sampled at last batch per epoch)
                last_grad_stats = compute_gradient_stats(model)

                # Gradient clipping norm=1.0  (mirrors GradientClippingConfig::Norm(1.0))
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_batches += 1

            avg_train_loss = train_loss_acc / train_num_items

            # Train regression metrics
            train_p = torch.cat(train_preds_all).numpy().flatten()
            train_t = torch.cat(train_targets_all).numpy().flatten()
            train_rmse = float(np.sqrt(np.mean((train_p - train_t) ** 2)))
            train_mae = float(np.mean(np.abs(train_p - train_t)))
            ss_res = np.sum((train_t - train_p) ** 2)
            ss_tot = np.sum((train_t - np.mean(train_t)) ** 2)
            train_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

            # ---------- Validation ----------
            model.eval()
            valid_loss_acc = 0.0
            val_preds_all = []
            val_targets_all = []

            with torch.no_grad():
                for sequences, targets in valid_loader:
                    sequences = sequences.to(device)
                    targets   = targets.to(device)
                    output = model(sequences, None)
                    loss = mse(output, targets)

                    loss_check = check_for_nans(loss, name=f"epoch_{epoch}_val_loss")
                    if not loss_check["is_stable"]:
                        nan_crash_log.append({
                            "epoch": epoch,
                            "phase": "validation",
                            "detail": loss_check,
                        })

                    valid_loss_acc += loss.item() * targets.shape[0]
                    val_preds_all.append(output.cpu())
                    val_targets_all.append(targets.cpu())

            avg_valid_loss = valid_loss_acc / valid_num_items

            # Validation regression metrics
            val_p = torch.cat(val_preds_all).numpy().flatten()
            val_t = torch.cat(val_targets_all).numpy().flatten()
            val_rmse = float(np.sqrt(np.mean((val_p - val_t) ** 2)))
            val_mae = float(np.mean(np.abs(val_p - val_t)))
            ss_res_v = np.sum((val_t - val_p) ** 2)
            ss_tot_v = np.sum((val_t - np.mean(val_t)) ** 2)
            val_r2 = float(1 - ss_res_v / ss_tot_v) if ss_tot_v != 0 else 0.0

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start

            # System utilization
            util = get_system_utilization(device)

            # Iteration speed
            iteration_speed = train_batches / epoch_time if epoch_time > 0 else 0.0

            # Record
            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_valid_loss,
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "train_r2": train_r2,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "epoch_time_sec": round(epoch_time, 3),
                "iteration_speed": round(iteration_speed, 2),
                "nan_detected": epoch_nan_detected,
                **last_grad_stats,
                **util,
            }
            epoch_history.append(epoch_record)

            # Print every 5 epochs — mirrors (epoch + 1) % 5 == 0
            if (epoch + 1) % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{config.num_epochs} | "
                    f"Loss={avg_train_loss:.6f} | "
                    f"Val Loss={avg_valid_loss:.6f} | "
                    f"R²={val_r2:.4f} | "
                    f"RMSE={val_rmse:.4f} | "
                    f"time={epoch_time:.2f}s | "
                    f"iter/s={iteration_speed:.1f}"
                )

        # ── Save trained model ──────────────────────────────────
        torch.save(model.state_dict(), os.path.join(artifact_dir, "model.pt"))

        # ── Convergence metrics ─────────────────────────────────
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

        # Training performance
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

        # Stability
        stability = {
            "total_nan_events": len(nan_crash_log),
            "nan_free_training": len(nan_crash_log) == 0,
            "convergence": convergence,
            "nan_crash_log": nan_crash_log if nan_crash_log else "none",
        }

        # ── Global summary (aggregated across all epochs) ────────
        train_losses = [h["train_loss"] for h in epoch_history]
        val_losses_all = [h["val_loss"] for h in epoch_history]
        grad_norms = [h.get("grad_norm_total", 0.0) for h in epoch_history]
        grad_means = [h.get("grad_norm_mean", 0.0) for h in epoch_history]
        grad_maxes = [h.get("grad_norm_max", 0.0) for h in epoch_history]
        cpu_pcts = [h.get("cpu_percent", 0.0) for h in epoch_history]
        cpu_mems = [h.get("cpu_mem_mb", 0.0) for h in epoch_history]
        iter_speeds = [h["iteration_speed"] for h in epoch_history]
        train_rmses = [h["train_rmse"] for h in epoch_history]
        train_maes = [h["train_mae"] for h in epoch_history]
        train_r2s = [h["train_r2"] for h in epoch_history]
        val_rmses = [h["val_rmse"] for h in epoch_history]
        val_maes = [h["val_mae"] for h in epoch_history]
        val_r2s = [h["val_r2"] for h in epoch_history]

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
                "min": float(np.min(val_losses_all)),
                "max": float(np.max(val_losses_all)),
                "mean": float(np.mean(val_losses_all)),
                "std": float(np.std(val_losses_all)),
                "first": val_losses_all[0],
                "last": val_losses_all[-1],
            },
            "train_rmse": {
                "min": float(np.min(train_rmses)),
                "max": float(np.max(train_rmses)),
                "mean": float(np.mean(train_rmses)),
                "first": train_rmses[0],
                "last": train_rmses[-1],
            },
            "val_rmse": {
                "min": float(np.min(val_rmses)),
                "max": float(np.max(val_rmses)),
                "mean": float(np.mean(val_rmses)),
                "first": val_rmses[0],
                "last": val_rmses[-1],
            },
            "train_mae": {
                "min": float(np.min(train_maes)),
                "max": float(np.max(train_maes)),
                "mean": float(np.mean(train_maes)),
                "first": train_maes[0],
                "last": train_maes[-1],
            },
            "val_mae": {
                "min": float(np.min(val_maes)),
                "max": float(np.max(val_maes)),
                "mean": float(np.mean(val_maes)),
                "first": val_maes[0],
                "last": val_maes[-1],
            },
            "train_r2": {
                "min": float(np.min(train_r2s)),
                "max": float(np.max(train_r2s)),
                "mean": float(np.mean(train_r2s)),
                "first": train_r2s[0],
                "last": train_r2s[-1],
            },
            "val_r2": {
                "min": float(np.min(val_r2s)),
                "max": float(np.max(val_r2s)),
                "mean": float(np.mean(val_r2s)),
                "first": val_r2s[0],
                "last": val_r2s[-1],
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

        run_metrics["global_summary"] = global_summary
        run_metrics["training_performance"] = performance
        run_metrics["training_stability"] = stability
        run_metrics["epoch_history"] = epoch_history
        run_metrics["status"] = "success"

        # ── Generate plots ──────────────────────────────────────
        print("\nGenerating training plots...")
        save_loss_curves(epoch_history, plots_dir)
        save_convergence_plot(epoch_history, plots_dir)
        save_epoch_timing_plot(epoch_history, plots_dir)
        save_gradient_norm_plot(epoch_history, plots_dir)
        save_resource_utilization_plot(epoch_history, plots_dir)
        print(f"Plots saved to {plots_dir}/")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total time       : {total_training_time:.2f}s")
        print(f"  Avg epoch time   : {performance['avg_time_per_epoch_sec']:.2f}s")
        print(f"  Final train loss : {convergence['final_train_loss']:.6f}")
        print(f"  Final val loss   : {convergence['final_val_loss']:.6f}")
        print(f"  Best val loss    : {convergence['best_val_loss']:.6f} (epoch {convergence['best_val_loss_epoch']})")
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

        with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
            json.dump(run_metrics, f, indent=4, default=str)

        print(f"Metrics saved to {artifact_dir}/metrics.json")
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
