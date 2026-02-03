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

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # Rust: Conv2dConfig::new([1, 8], [3, 3]) (Defaults: stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        # Rust: Conv2dConfig::new([8, 16], [3, 3])
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        # Rust: AdaptiveAvgPool2dConfig::new([8, 8])
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Rust: LinearConfig::new(16 * 8 * 8, hidden_size) (Defaults: bias=True)
        self.linear1 = nn.Linear(16 * 8 * 8, hidden_size, bias=True)
        
        # Rust: LinearConfig::new(hidden_size, num_classes)
        self.linear2 = nn.Linear(hidden_size, num_classes, bias=True)

        # Rust: DropoutConfig::new(dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Rust: Relu::new()
        self.activation = nn.ReLU()

    def forward(self, x):
        # Rust:
        # let [batch_size, height, width] = images.dims();
        # let x = images.reshape([batch_size, 1, height, width]);
        if x.ndim == 3:
            b, h, w = x.shape
            x = x.view(b, 1, h, w)
        
        # Rust:
        # let x = self.conv1.forward(x);
        # let x = self.dropout.forward(x);
        # let x = self.conv2.forward(x);
        # let x = self.dropout.forward(x);
        # let x = self.activation.forward(x);
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.activation(x)

        # Rust:
        # let x = self.pool.forward(x);
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
        # self.linear2.forward(x)
        x = self.linear2(x)
        
        return x
