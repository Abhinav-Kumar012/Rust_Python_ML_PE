# model.py
# TensorFlow equivalent of model.rs

import tensorflow as tf
from tensorflow.keras import layers, models


NUM_CLASSES = 10


# -----------------------------
# ConvBlock
# -----------------------------
class ConvBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, kernel, pool=True):
        super().__init__()
        self.conv = layers.Conv2D(
            out_ch,
            kernel_size=kernel,
            padding="valid",
            use_bias=True,
        )
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
        self.pool = layers.MaxPooling2D(2) if pool else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        if self.pool:
            x = self.pool(x)
        return x


# -----------------------------
# Model
# -----------------------------
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 64, 3, pool=True)
        self.conv2 = ConvBlock(64, 64, 3, pool=True)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(NUM_CLASSES)

        self.dropout = layers.Dropout(0.25)
        self.act = layers.Activation("gelu")

    def call(self, x, training=False):
        # x: [B,28,28]
        x = tf.expand_dims(x, axis=-1)  # [B,28,28,1]
        x = tf.stop_gradient(x)         # matches Rust detach()

        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x, training=training)

        return self.fc3(x)
