# model.py
# Exact translation of model.rs

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10


# -----------------------------
# ConvBlock (Rust: ConvBlock)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=0)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pool:
            x = self.pool(x)
        return x


# -----------------------------
# Model (Rust: Model<B>)
# -----------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 64, 3, pool=True)
        self.conv2 = ConvBlock(64, 64, 3, pool=True)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

        self.dropout = nn.Dropout(0.25)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, 28, 28]
        b, h, w = x.shape
        x = x.view(b, 1, h, w).detach()  # matches Rust detach

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        return self.fc3(x)