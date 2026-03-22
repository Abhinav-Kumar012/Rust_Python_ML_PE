import os
import json
import shutil
import time
import random
import urllib.request
import psutil
import torch
import numpy as np
from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Constants
# ==========================================================

NUM_FEATURES = 13

DATASET_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz"
RAW_DATA_FILE = "boston_housing.npz"

TRAIN_FILE = "train_data.npz"
VALID_FILE = "valid_data.npz"


# ==========================================================
# Dataset preparation
# ==========================================================

def prepare_dataset():

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

    torch.save(
        model.state_dict(),
        os.path.join(artifact_dir, "model.pt")
    )

    print("Training complete. Artifacts saved to:", artifact_dir)


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    run(
        artifact_dir="artifacts",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )