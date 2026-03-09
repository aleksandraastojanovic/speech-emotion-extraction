"""
model.py — CNN architecture for RAVDESS emotion classification.

Architecture:
    Input (128, 128, 1)
    → CNNBlock(32)  → CNNBlock(64)
    → CNNBlock(128, dropout=0.3) → CNNBlock(256, dropout=0.3)
    → GlobalAveragePooling2D
    → Dense(256, ReLU, L2)
    → Dropout(0.5)
    → Dense(8, Softmax)

Regularization stack:
    BatchNorm (every block) + Conv Dropout (blocks 3-4) + Dense Dropout(0.5)
    + L2(1e-4) on weights + label smoothing(0.1) in loss
"""

import tensorflow as tf
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

L2 = 1e-4


class CNNBlock:

    def __init__(self, filters: int, dropout_rate: float = 0.0):
        self.filters      = filters
        self.dropout_rate = dropout_rate

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
        if self.dropout_rate > 0.0:
            x = layers.Dropout(self.dropout_rate)(x)
        return x


class EmotionCNN:
    

    def build_model(self) -> Model:
        inputs = Input(shape=(*IMG_SIZE, 1), name="mel_spectrogram")

        x = CNNBlock(32).build(inputs)              # → (64, 64, 32)
        x = CNNBlock(64).build(x)                  # → (32, 32, 64)
        x = CNNBlock(128, dropout_rate=0.3).build(x)  # → (16, 16, 128)
        x = CNNBlock(256, dropout_rate=0.3).build(x)  # → (8,  8,  256)

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
            loss=self._label_smoothed_loss,
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def _label_smoothed_loss(y_true, y_pred, smoothing: float = 0.1):
        """
        Label smoothing for sparse (integer) labels.
        Converts to one-hot internally, then applies smoothing:
            smooth_label = true_label * (1 - ε) + ε / n_classes
        Prevents overconfident predictions and helps generalisation.
        """
        n_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        smooth    = y_true_oh * (1.0 - smoothing) + smoothing / tf.cast(n_classes, tf.float32)
        return tf.keras.losses.categorical_crossentropy(smooth, y_pred)

    def summary(self):
        model = self.build_model()
        model.summary()
        return model


if __name__ == "__main__":
    EmotionCNN().summary()
