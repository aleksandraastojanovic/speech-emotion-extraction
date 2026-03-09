"""
model.py — CNN architecture for RAVDESS emotion classification.

Architecture:
    Input (128, 128, 1)
    → 4× CNNBlock (32 → 64 → 128 → 256 filters)
    → GlobalAveragePooling2D
    → Dense(256, ReLU, L2)
    → Dropout(0.5)
    → Dense(8, Softmax)
"""

import tensorflow as tf
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

L2 = 1e-4


class CNNBlock:
    

    def __init__(self, filters: int):
        self.filters = filters

    def build(self, inputs):
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,          # BN has its own bias term
            kernel_regularizer=regularizers.l2(L2),
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x


class EmotionCNN:
    

    def build_model(self) -> Model:
        inputs = Input(shape=(*IMG_SIZE, 1), name="mel_spectrogram")

        x = CNNBlock(32).build(inputs)   # → (64, 64, 32)
        x = CNNBlock(64).build(x)        # → (32, 32, 64)
        x = CNNBlock(128).build(x)       # → (16, 16, 128)
        x = CNNBlock(256).build(x)       # → (8,  8,  256)

        x = layers.GlobalAveragePooling2D()(x)   # → (256,)
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2),
            name="dense_256",
        )(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

        model = Model(inputs=inputs, outputs=outputs, name="EmotionCNN")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        model = self.build_model()
        model.summary()
        return model


if __name__ == "__main__":
    EmotionCNN().summary()
