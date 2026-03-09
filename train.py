"""
train.py — Training loop for RAVDESS emotion CNN.

Receives a compiled Keras model and numpy arrays from preprocess.py.
Handles class weights, callbacks, and history plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import BATCH_SIZE, EPOCHS, MODEL_PATH, RESULTS_DIR


class Trainer:
    def __init__(self, model: tf.keras.Model,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val:   np.ndarray, y_val:   np.ndarray):
        self.model   = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

    # ------------------------------------------------------------------
    def compute_class_weights(self) -> dict[int, float]:
        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.y_train,
        )
        return dict(zip(classes.tolist(), weights.tolist()))

    # ------------------------------------------------------------------
    def train(self) -> tf.keras.callbacks.History:
        class_weight = self.compute_class_weights()
        print(f"[Trainer] class weights: {class_weight}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            shuffle=True,
        )
        return history

    # ------------------------------------------------------------------
    def plot_history(self, history: tf.keras.callbacks.History) -> None:
        hist      = history.history
        epochs    = range(1, len(hist["loss"]) + 1)
        best_ep   = int(np.argmin(hist["val_loss"])) + 1   # 1-indexed

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # — Accuracy —
        ax1.plot(epochs, hist["accuracy"],     label="train")
        ax1.plot(epochs, hist["val_accuracy"], label="val")
        ax1.axvline(best_ep, color="gray", linestyle="--", label=f"best epoch {best_ep}")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # — Loss —
        ax2.plot(epochs, hist["loss"],     label="train")
        ax2.plot(epochs, hist["val_loss"], label="val")
        ax2.axvline(best_ep, color="gray", linestyle="--", label=f"best epoch {best_ep}")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        fig.tight_layout()
        out_path = f"{RESULTS_DIR}/training_history.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Trainer] history plot saved → {out_path}")


if __name__ == "__main__":
    print("train.py loaded OK")
