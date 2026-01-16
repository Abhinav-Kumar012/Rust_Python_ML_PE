# model.py
# Exact translation of model.rs

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10


# -----------------------------
# Model (Rust: Model<B>)
# -----------------------------
class Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=512, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.linear1 = nn.Linear(16 * 8 * 8, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, num_classes, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, 28, 28] (expected input after reshape in training loop or here)
        # The training loop reshapes to [B, 1, H, W], so input here is likely [B, H, W] or [B, 1, H, W]
        # Rust `forward`:
        # let [batch_size, height, width] = images.dims();
        # let x = images.reshape([batch_size, 1, height, width]);
        
        if x.ndim == 3:
            b, h, w = x.shape
            x = x.view(b, 1, h, w)
        
        # Rust:
        # let x = self.conv1.forward(x); // [batch_size, 8, 26, 26]
		# let x = self.dropout.forward(x);
		# let x = self.conv2.forward(x); // [batch_size, 16, 24, 24]
		# let x = self.dropout.forward(x);
		# let x = self.activation.forward(x);
        
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.activation(x)

        # Rust:
        # let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
		# let x = x.reshape([batch_size, 16 * 8 * 8]);
		# let x = self.linear1.forward(x);
		# let x = self.dropout.forward(x);
		# let x = self.activation.forward(x);
        
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)

        # Rust:
        # self.linear2.forward(x) // [batch_size, num_classes]
        
        x = self.linear2(x)
        
        return x