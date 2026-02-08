# data.py
# TensorFlow equivalent of data.rs

import math
import random
from enum import Enum
from typing import List

import tensorflow as tf


# -----------------------------
# Transform enum
# -----------------------------
class Transform(Enum):
    Translate = "Tr"
    Shear = "Sr"
    Scale = "Sc"
    Rotation = "Rot"

    def __str__(self):
        return self.value


# -----------------------------
# Image preparation
# -----------------------------
def prepare_image(image, label, transforms: List[Transform]):
    # image: [28,28], uint8
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=-1)  # [28,28,1]

    # Normalize (exact constants)
    image = (image / 255.0 - 0.1307) / 0.3081

    if transforms:
        image = mangle_image(image, transforms)

    return image, label


def mangle_image(image, transforms):
    angle = 0.0
    translate_x = 0.0
    translate_y = 0.0
    scale = 1.0
    shear_x = 0.0
    shear_y = 0.0

    for t in transforms:
        if t == Transform.Translate:
            translate_x += random.uniform(-0.2, 0.2) * 28
            translate_y += random.uniform(-0.2, 0.2) * 28
        elif t == Transform.Shear:
            shear_x += random.uniform(-0.6, 0.6)
            shear_y += random.uniform(-0.6, 0.6)
        elif t == Transform.Scale:
            scale *= random.uniform(0.6, 1.5)
        elif t == Transform.Rotation:
            angle += random.uniform(-math.pi / 4, math.pi / 4)

    # Compose affine transform
    transform = tf.keras.layers.experimental.preprocessing.RandomRotation(0.0)
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image.numpy(),
        theta=angle * 180 / math.pi,
        tx=translate_x,
        ty=translate_y,
        shear=shear_x,
        zx=scale,
        zy=scale,
        fill_mode="nearest",
    )

    return tf.convert_to_tensor(image, dtype=tf.float32)


# -----------------------------
# Dataset composition
# -----------------------------
def build_dataset(images, labels, transforms, repeat, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        ds = ds.shuffle(10_000)

    ds = ds.map(
        lambda x, y: prepare_image(x, y, transforms),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if repeat:
        ds = ds.repeat()

    return ds