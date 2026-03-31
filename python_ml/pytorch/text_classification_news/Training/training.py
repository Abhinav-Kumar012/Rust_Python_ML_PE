import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import math
import time
import platform
import sys
import os
import json
import random
import psutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Configuration
# ==========================================================

class Config:
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    num_classes = 4  # AG News has 4 classes
    vocab_size = 28996 # bert-base-cased vocab size
    max_seq_len = 256
    batch_size = 8
    num_epochs = 5
    lr = 1e-2
    warmup_steps = 1000
    weight_decay = 5e-5
    norm_first = True
    quiet_softmax = True # Burn specific, PyTorch standard softmax is fine
    seed = 42

# Artifact directories
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pytorch_text_classification")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")


# ==========================================================
# Model
# ==========================================================

class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        
        # Embeddings
        self.embedding_token = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_pos = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            norm_first=config.norm_first
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model) if config.norm_first else None
        )
        
        # Output
        self.output = nn.Linear(config.d_model, config.num_classes)
        
        self.max_seq_len = config.max_seq_len

    def forward(self, tokens, mask_pad=None):
        batch_size, seq_length = tokens.shape
        device = tokens.device
        
        # Positions: [0, 1, 2, ..., seq_length-1] repeated for batch
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        embedded_tokens = self.embedding_token(tokens)
        embedded_positions = self.embedding_pos(positions)
        
        # Burn implementation: (pos + tok) / 2
        embedding = (embedded_positions + embedded_tokens) / 2
        
        src_key_padding_mask = None
        if mask_pad is not None:
            src_key_padding_mask = mask_pad
            
        encoded = self.transformer(embedding, src_key_padding_mask=src_key_padding_mask)
        
        output = self.output(encoded) # [batch, seq, num_classes]
        
        # Slice [:, 0, :] -> [batch, num_classes]
        output_classification = output[:, 0, :]
        
        return output_classification


# ==========================================================
# LR Scheduler
# ==========================================================

class NoamLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.factor * (self.model_size ** (-0.5)) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale for _ in self.base_lrs]


# ==========================================================
# Utilities
# ==========================================================

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
    """Plot train/validation accuracy curves."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [h["train_accuracy"] for h in history], "b-o", label="Train Acc", markersize=4)
    ax.plot(epochs, [h["val_accuracy"] for h in history], "r-o", label="Val Acc", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
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


def save_confusion_matrix_plot(targets, preds, class_names, plots_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation Set)")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def save_lr_schedule_plot(history, plots_dir):
    """Plot learning rate schedule."""
    epochs = [h["epoch"] for h in history]
    lrs = [h.get("learning_rate", 0) for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lrs, "g-o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Noam)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lr_schedule.png"), dpi=150)
    plt.close(fig)


# ==========================================================
# Tokenizer + Collate
# ==========================================================

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    encoding = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=Config.max_seq_len,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    tokens = encoding['input_ids']
    mask_pad = (encoding['attention_mask'] == 0)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return tokens, labels, mask_pad


# ==========================================================
# Training
# ==========================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    config = Config()
    set_seed(config.seed)

    # AG News class names
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    # Metrics container
    run_metrics = {
        "timestamp": time.time(),
        "status": "pending",
        "config": {
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "num_classes": config.num_classes,
            "max_seq_len": config.max_seq_len,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
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
        run_metrics["environment"]["gpu_name"] = torch.cuda.get_device_name(device)

    start_time = time.time()
    nan_crash_log = []
    epoch_history = []

    try:
        # Data
        print("Loading dataset...")
        dataset = load_dataset(
            "ag_news",
            trust_remote_code=False,
            download_mode="reuse_cache_if_exists",
            verification_mode="no_checks",
        )

        train_dataset = dataset['train'].shuffle(seed=config.seed).select(range(50000))
        test_dataset = dataset['test'].shuffle(seed=config.seed).select(range(5000))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Model
        model = TextClassificationModel(config).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run_metrics["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay)
        
        # Scheduler
        scheduler = NoamLR(optimizer, config.d_model, config.warmup_steps, factor=config.lr)
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
        print("Starting training...")
        print("=" * 60)

        for epoch in range(config.num_epochs):
            epoch_start = time.time()

            model.train()
            train_loss = 0.0
            train_correct = 0
            total_train = 0
            train_batches = 0
            epoch_nan_detected = False
            last_grad_stats = {}
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
            
            for tokens, labels, mask_pad in progress_bar:
                tokens, labels, mask_pad = tokens.to(device), labels.to(device), mask_pad.to(device)
                
                optimizer.zero_grad()
                outputs = model(tokens, mask_pad)
                loss = criterion(outputs, labels)

                # NaN check
                loss_check = check_for_nans(loss, name=f"epoch_{epoch+1}_train")
                if not loss_check["is_stable"]:
                    epoch_nan_detected = True
                    nan_crash_log.append({
                        "epoch": epoch + 1,
                        "phase": "train",
                        "batch": train_batches,
                        "detail": loss_check,
                    })

                loss.backward()

                # Gradient stats (sampled periodically)
                if train_batches % 500 == 0:
                    last_grad_stats = compute_gradient_stats(model)

                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * tokens.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
                
            avg_train_loss = train_loss / total_train
            train_acc = train_correct / total_train
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            total_val = 0
            all_val_preds = []
            all_val_targets = []
            
            with torch.no_grad():
                for tokens, labels, mask_pad in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                    tokens, labels, mask_pad = tokens.to(device), labels.to(device), mask_pad.to(device)
                    
                    outputs = model(tokens, mask_pad)
                    loss = criterion(outputs, labels)

                    loss_check = check_for_nans(loss, name=f"epoch_{epoch+1}_val")
                    if not loss_check["is_stable"]:
                        nan_crash_log.append({
                            "epoch": epoch + 1,
                            "phase": "validation",
                            "detail": loss_check,
                        })
                    
                    val_loss += loss.item() * tokens.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_val_preds.extend(predicted.cpu().numpy().tolist())
                    all_val_targets.extend(labels.cpu().numpy().tolist())
                    
            avg_val_loss = val_loss / total_val
            val_acc = val_correct / total_val

            # Per-epoch classification metrics
            val_precision = precision_score(all_val_targets, all_val_preds, average="macro", zero_division=0)
            val_recall = recall_score(all_val_targets, all_val_preds, average="macro", zero_division=0)
            val_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start

            # System utilization
            util = get_system_utilization(device)

            # Iteration speed
            iteration_speed = train_batches / epoch_time if epoch_time > 0 else 0.0

            # Current LR
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_precision_macro": float(val_precision),
                "val_recall_macro": float(val_recall),
                "val_f1_macro": float(val_f1),
                "learning_rate": current_lr,
                "epoch_time_sec": round(epoch_time, 3),
                "iteration_speed": round(iteration_speed, 2),
                "nan_detected": epoch_nan_detected,
                **last_grad_stats,
                **util,
            }
            epoch_history.append(epoch_record)
            
            print(
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"F1: {val_f1:.4f} | time={epoch_time:.1f}s"
            )

        # Save model
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "ag_news_model.pth"))
        print(f"\nModel saved to {ARTIFACT_DIR}/ag_news_model.pth")

        # Convergence metrics
        train_losses = [h["train_loss"] for h in epoch_history]
        val_losses = [h["val_loss"] for h in epoch_history]

        convergence = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_val_loss_epoch": int(np.argmin(val_losses) + 1),
            "final_val_accuracy": epoch_history[-1]["val_accuracy"],
            "best_val_accuracy": max(h["val_accuracy"] for h in epoch_history),
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

        # Global summary (aggregated across all epochs)
        grad_norms = [h.get("grad_norm_total", 0.0) for h in epoch_history]
        grad_means = [h.get("grad_norm_mean", 0.0) for h in epoch_history]
        grad_maxes = [h.get("grad_norm_max", 0.0) for h in epoch_history]
        cpu_pcts = [h.get("cpu_percent", 0.0) for h in epoch_history]
        cpu_mems = [h.get("cpu_mem_mb", 0.0) for h in epoch_history]
        iter_speeds = [h["iteration_speed"] for h in epoch_history]
        train_accs = [h["train_accuracy"] for h in epoch_history]
        val_accs = [h["val_accuracy"] for h in epoch_history]
        val_f1s = [h["val_f1_macro"] for h in epoch_history]
        val_precisions = [h["val_precision_macro"] for h in epoch_history]
        val_recalls = [h["val_recall_macro"] for h in epoch_history]

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
            "val_f1_macro": {
                "min": float(np.min(val_f1s)),
                "max": float(np.max(val_f1s)),
                "mean": float(np.mean(val_f1s)),
                "first": val_f1s[0],
                "last": val_f1s[-1],
            },
            "val_precision_macro": {
                "min": float(np.min(val_precisions)),
                "max": float(np.max(val_precisions)),
                "mean": float(np.mean(val_precisions)),
            },
            "val_recall_macro": {
                "min": float(np.min(val_recalls)),
                "max": float(np.max(val_recalls)),
                "mean": float(np.mean(val_recalls)),
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

        # Final test metrics
        run_metrics["test_metrics"] = {
            "accuracy": epoch_history[-1]["val_accuracy"],
            "precision_macro": epoch_history[-1]["val_precision_macro"],
            "recall_macro": epoch_history[-1]["val_recall_macro"],
            "f1_macro": epoch_history[-1]["val_f1_macro"],
        }
        run_metrics["global_summary"] = global_summary
        run_metrics["training_performance"] = performance
        run_metrics["training_stability"] = stability
        run_metrics["epoch_history"] = epoch_history
        run_metrics["status"] = "success"

        # Generate plots
        print("\nGenerating training plots...")
        save_loss_curves(epoch_history, PLOTS_DIR)
        save_accuracy_curve(epoch_history, PLOTS_DIR)
        save_convergence_plot(epoch_history, PLOTS_DIR)
        save_epoch_timing_plot(epoch_history, PLOTS_DIR)
        save_gradient_norm_plot(epoch_history, PLOTS_DIR)
        save_resource_utilization_plot(epoch_history, PLOTS_DIR)
        save_lr_schedule_plot(epoch_history, PLOTS_DIR)
        save_confusion_matrix_plot(all_val_targets, all_val_preds, class_names, PLOTS_DIR)
        print(f"Plots saved to {PLOTS_DIR}/")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total time       : {total_training_time:.2f}s")
        print(f"  Avg epoch time   : {performance['avg_time_per_epoch_sec']:.2f}s")
        print(f"  Final train loss : {convergence['final_train_loss']:.4f}")
        print(f"  Final val loss   : {convergence['final_val_loss']:.4f}")
        print(f"  Final val acc    : {convergence['final_val_accuracy']:.4f}")
        print(f"  Best val acc     : {convergence['best_val_accuracy']:.4f}")
        print(f"  Final F1         : {epoch_history[-1]['val_f1_macro']:.4f}")
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

        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
            json.dump(run_metrics, f, indent=4, default=str)

        print(f"Metrics saved to {ARTIFACT_DIR}/metrics.json")


if __name__ == "__main__":
    train()
