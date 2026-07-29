"""
train.py — Training loop for RAVDESS emotion CNN.

Receives a compiled Keras model and numpy arrays from preprocess.py.
Handles class weights, callbacks, and history plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, MODEL_PATH, RESULTS_DIR, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE


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
    def train(self) -> tf.keras.callbacks.History:
        # No class_weight — per-class augmentation in preprocess.py already
        # balances the training distribution exactly.
        # Model selection tracks val_accuracy: with focal loss on a small,
        # noisy val set, val_loss keeps picking lucky early epochs.
        # ReduceLROnPlateau stays on val_loss (smoother signal for LR).
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
        )
        return history

    # ------------------------------------------------------------------
    def plot_history(self, history: tf.keras.callbacks.History) -> None:
        hist      = history.history
        epochs    = range(1, len(hist["loss"]) + 1)
        best_ep   = int(np.argmax(hist["val_accuracy"])) + 1   # matches checkpoint

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
