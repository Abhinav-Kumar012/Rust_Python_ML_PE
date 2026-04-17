import os
import json
import shutil
import time
import random
import psutil
import torch
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Constants
# ==========================================================

NUM_FEATURES = 8

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

FEATURES_MIN = torch.tensor([0.4999, 1., 0.8461, 0.375, 3., 0.6923, 32.54, -124.35], dtype=torch.float32)
FEATURES_MAX = torch.tensor([15., 52., 141.9091, 34.0667, 35682., 1243.3333, 41.95, -114.31], dtype=torch.float32)


# ==========================================================
# Dataset
# ==========================================================

class HousingDataset(Dataset):

    def __init__(self, split="train"):
        hf_dataset = load_dataset("gvlassis/california_housing", split=split)
        
        # Extract features in the correct order:
        # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        features = [
            hf_dataset["MedInc"],
            hf_dataset["HouseAge"],
            hf_dataset["AveRooms"],
            hf_dataset["AveBedrms"],
            hf_dataset["Population"],
            hf_dataset["AveOccup"],
            hf_dataset["Latitude"],
            hf_dataset["Longitude"]
        ]
        
        targets = hf_dataset["MedHouseVal"]
        
        inputs = torch.tensor(features, dtype=torch.float32).T
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        self.inputs = (inputs - FEATURES_MIN) / (FEATURES_MAX - FEATURES_MIN)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    @staticmethod
    def train():
        return HousingDataset("train")

    @staticmethod
    def validation():
        return HousingDataset("validation")


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


def cpu_metrics():

    process = psutil.Process(os.getpid())

    cpu_use = psutil.cpu_percent()
    memory = process.memory_info().rss / (1024 ** 2)

    try:
        temps = psutil.sensors_temperatures()
        cpu_temp = list(temps.values())[0][0].current
    except:
        cpu_temp = 0.0

    return cpu_use, cpu_temp, memory


# ==========================================================
# Training
# ==========================================================

def train_epoch(model, dataloader, optimizer, device):

    model.train()

    mse = nn.MSELoss()

    total_loss = 0

    start_time = time.time()

    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = mse(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    end_time = time.time()

    iteration_speed = len(dataloader) / (end_time - start_time)

    cpu_use, cpu_temp, cpu_mem = cpu_metrics()

    return {
        "loss": total_loss / len(dataloader),
        "iteration_speed": iteration_speed,
        "cpu_use": cpu_use,
        "cpu_temp": cpu_temp,
        "cpu_mem": cpu_mem
    }


def validate_epoch(model, dataloader, device):

    model.eval()

    mse = nn.MSELoss()

    total_loss = 0

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)

            outputs = model(inputs)

            loss = mse(outputs, targets)

            total_loss += loss.item()

    cpu_use, cpu_temp, cpu_mem = cpu_metrics()

    return {
        "loss": total_loss / len(dataloader),
        "cpu_use": cpu_use,
        "cpu_temp": cpu_temp,
        "cpu_mem": cpu_mem
    }


# ==========================================================
# Run (Equivalent to Rust run())
# ==========================================================

def run(artifact_dir, device):

    device = torch.device(device)

    create_artifact_dir(artifact_dir)

    config = ExpConfig()

    set_seed(config.seed)

    train_dataset = HousingDataset.train()
    valid_dataset = HousingDataset.validation()

    print("Train Dataset Size:", len(train_dataset))
    print("Valid Dataset Size:", len(valid_dataset))

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

    model = RegressionModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []

    for epoch in range(config.num_epochs):

        train_metrics = train_epoch(model, train_loader, optimizer, device)

        valid_metrics = validate_epoch(model, valid_loader, device)

        lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "iteration_speed": train_metrics["iteration_speed"],
            "learning_rate": lr,
            "train_cpu_use": train_metrics["cpu_use"],
            "valid_cpu_use": valid_metrics["cpu_use"],
            "train_cpu_temp": train_metrics["cpu_temp"],
            "valid_cpu_temp": valid_metrics["cpu_temp"],
            "train_cpu_mem": train_metrics["cpu_mem"],
            "valid_cpu_mem": valid_metrics["cpu_mem"]
        }

        history.append(metrics)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} "
            f"Train Loss: {metrics['train_loss']:.6f} "
            f"Valid Loss: {metrics['valid_loss']:.6f}"
        )

    with open(os.path.join(artifact_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=4)

    model_dir = os.path.join(artifact_dir, "model_pytorch_regression")
    os.makedirs(model_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(model_dir, "ag_news_model.pth")
    )

    print("Training complete. Artifacts saved to:", artifact_dir)


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    run(
        artifact_dir=os.path.join(OUTPUT_DIR, "artifacts"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )