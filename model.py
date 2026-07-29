"""
model.py — CNN architecture for RAVDESS emotion classification.

Architecture:
    Input (128, 128, 3)  — mel + delta + delta-delta channels
    → CNNBlock(32)  → CNNBlock(64)
    → CNNBlock(128, dropout=0.3) → CNNBlock(256, dropout=0.3)
    → GlobalAveragePooling2D
    → Dense(256, ReLU, L2)
    → Dropout(0.5)
    → Dense(8, Softmax)

Regularization stack:
    BatchNorm (every block) + Conv Dropout (blocks 3-4) + Dense Dropout(0.5)
    + L2(1e-4) on weights + focal loss (γ=2.0)
"""

import tensorflow as tf
import keras
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, N_CHANNELS, NUM_CLASSES, LEARNING_RATE, FOCAL_GAMMA

L2 = 1e-4


@keras.saving.register_keras_serializable(package="EmotionCNN")
def focal_loss(y_true, y_pred):
    """
    Focal loss (Lin et al., 2017): FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Down-weights easy examples where the model is already confident,
    forcing it to focus training signal on hard/misclassified samples
    (fearful, sad, surprised). γ=2 is the standard default.
    """
    n_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
    p_t       = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
    p_t       = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(-tf.pow(1.0 - p_t, FOCAL_GAMMA) * tf.math.log(p_t))


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
        inputs = Input(shape=(*IMG_SIZE, N_CHANNELS), name="mel_spectrogram")

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
            loss=focal_loss,
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        model = self.build_model()
        model.summary()
        return model


if __name__ == "__main__":
    EmotionCNN().summary()
