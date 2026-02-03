# main.py
import torch
from python_ml.pytorch.mnist.Training.training import run

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(device)

