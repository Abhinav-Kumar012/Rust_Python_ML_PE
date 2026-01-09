# training.py
# TensorFlow equivalent of training.rs

import os
import math
import random
import tensorflow as tf

from data import Transform, build_dataset
from model import Model


ARTIFACT_DIR = "/tmp/burn-example-mnist"


def generate_idents(num_samples_base):
    idents = []
    for shear in [None, Transform.Shear]:
        for scale in [None, Transform.Scale]:
            for rot in [None, Transform.Rotation]:
                for tr in [None, Transform.Translate]:
                    current = [t for t in [shear, scale, rot, tr] if t is not None]
                    size = None
                    if num_samples_base:
                        size = num_samples_base * max(1, len(current))
                    idents.append((current, size))
    return idents


def run():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    tf.random.set_seed(42)
    random.seed(42)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_tr = x_train[:55_000]
    y_tr = y_train[:55_000]
    x_val = x_train[55_000:60_000]
    y_val = y_train[55_000:60_000]

    train_idents = generate_idents(10_000)
    val_idents = generate_idents(None)

    train_sets = []
    for transforms, _ in train_idents:
        train_sets.append(
            build_dataset(x_tr, y_tr, transforms, repeat=False, shuffle=True)
        )

    val_sets = []
    for transforms, _ in val_idents:
        val_sets.append(
            build_dataset(x_val, y_val, transforms, repeat=False, shuffle=False)
        )

    train_ds = train_sets[0]
    for ds in train_sets[1:]:
        train_ds = train_ds.concatenate(ds)

    val_ds = val_sets[0]
    for ds in val_sets[1:]:
        val_ds = val_ds.concatenate(ds)

    train_ds = train_ds.batch(256).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(256).prefetch(tf.data.AUTOTUNE)

    model = Model()

    # Optimizer (AdamW)
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=1.0,
        weight_decay=5e-5,
    )

    # Step-based LR schedule
    def lr_schedule(step):
        if step < 2000:
            return 1e-8 + step * (1.0 - 1e-8) / 2000
        elif step < 4000:
            return 0.5 * (1 + math.cos(math.pi * (step - 2000) / 2000))
        else:
            return max(
                1e-6,
                1e-2 - (step - 4000) * (1e-2 - 1e-6) / 10000,
            )

    optimizer.learning_rate = lr_schedule

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[early_stop],
    )

    model.save(os.path.join(ARTIFACT_DIR, "model"))
