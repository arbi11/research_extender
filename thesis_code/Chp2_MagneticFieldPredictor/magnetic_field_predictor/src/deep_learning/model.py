"""
Deep Learning model for magnetic field prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model(img_size=(400, 400)):
    """Create the CNN model for field prediction."""
    inputs = keras.Input(shape=img_size + (1,))

    inp = layers.BatchNormalization()(inputs)

    x1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.5)(x1)

    x2 = layers.Conv2D(64, 3, activation="relu", padding="same")(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.5)(x2)

    x3 = layers.Conv2D(128, 3, activation="relu", padding="same")(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.5)(x3)

    x4 = layers.Conv2D(128, 3, activation="relu", padding="same")(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.5)(x4)

    x5 = layers.Conv2D(128, 3, padding="same", activation="relu")(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Dropout(0.5)(x5)

    x_out_4 = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x5)
    x_out_4 = layers.BatchNormalization()(x_out_4)
    x_out_4 = layers.Dropout(0.5)(x_out_4)
    x_out_4 = layers.Add()([x_out_4, x5])

    x_out_3 = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x_out_4)
    x_out_3 = layers.BatchNormalization()(x_out_3)
    x_out_3 = layers.Dropout(0.5)(x_out_3)
    x_out_3 = layers.Add()([x_out_3, x4])

    x_out_2 = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x_out_3)
    x_out_2 = layers.BatchNormalization()(x_out_2)
    x_out_2 = layers.Dropout(0.5)(x_out_2)
    x_out_2 = layers.Add()([x_out_2, x3])

    x_out_1 = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x_out_2)
    x_out_1 = layers.BatchNormalization()(x_out_1)
    x_out_1 = layers.Dropout(0.5)(x_out_1)
    x_out_1 = layers.Add()([x_out_1, x2])

    x_out_0 = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x_out_1)
    x_out_0 = layers.Add()([x_out_0, x1])

    outputs = layers.Conv2D(1, 1, activation="relu", padding="same")(x_out_0)

    model = keras.Model(inputs, outputs)
    return model
