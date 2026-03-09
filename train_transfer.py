"""
train_transfer.py — Two-phase training loop for TransferEmotionCNN.

Phase 1 (frozen base)
    Only the new classification head is trained.
    Runs for PHASE1_EPOCHS epochs with Adam(TRANSFER_LR=1e-3).

Phase 2 (fine-tuning)
    Top FINETUNE_LAYERS of EfficientNetB0 are unfrozen.
    Runs for PHASE2_EPOCHS additional epochs with Adam(FINETUNE_LR=1e-4).
    A separate ModelCheckpoint saves to MODEL_TRANSFER_PATH so it never
    overwrites the scratch CNN checkpoint.

History plot
    Single figure with accuracy and loss panels. A vertical dashed line marks
    the Phase 1 / Phase 2 boundary, and another marks the best val_loss epoch.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    BATCH_SIZE,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    MODEL_TRANSFER_PATH,
    RESULTS_DIR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
)
from model_transfer import TransferEmotionModel


class TransferTrainer:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

    # ── public ──────────────────────────────────────────────────────────────
    def train(self) -> tuple[tf.keras.callbacks.History,
                              tf.keras.callbacks.History]:
        """
        Run both phases. Returns (history_phase1, history_phase2).
        The best model is saved to MODEL_TRANSFER_PATH by ModelCheckpoint.
        """
        builder = TransferEmotionModel()
        model   = builder.build_model()

        print("\n[TransferTrainer] ── Phase 1: training head (base frozen) ──")
        h1 = self._fit(model, PHASE1_EPOCHS, phase=1)

        print("\n[TransferTrainer] ── Phase 2: fine-tuning top layers ──")
        model = builder.prepare_finetuning(model)
        h2 = self._fit(model, PHASE2_EPOCHS, phase=2)

        return h1, h2

    # ── private ─────────────────────────────────────────────────────────────
    def _fit(
        self,
        model: tf.keras.Model,
        epochs: int,
        phase: int,
    ) -> tf.keras.callbacks.History:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_TRANSFER_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        return model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
        )

    # ── plotting ────────────────────────────────────────────────────────────
    def plot_history(
        self,
        h1: tf.keras.callbacks.History,
        h2: tf.keras.callbacks.History,
    ) -> None:
        """
        Plot combined accuracy and loss curves for both phases.
        A vertical dashed line marks the Phase 1/2 boundary.
        Another marks the best val_loss epoch across both phases.
        """
        # Concatenate metrics from both phases
        def concat(key):
            return h1.history.get(key, []) + h2.history.get(key, [])

        acc      = concat("accuracy")
        val_acc  = concat("val_accuracy")
        loss     = concat("loss")
        val_loss = concat("val_loss")

        total_epochs = len(acc)
        epochs       = range(1, total_epochs + 1)
        phase_split  = len(h1.history.get("loss", []))
        best_ep      = int(np.argmin(val_loss)) + 1   # 1-indexed

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # — Accuracy —
        ax1.plot(epochs, acc,     label="train")
        ax1.plot(epochs, val_acc, label="val")
        ax1.axvline(phase_split, color="blue",  linestyle=":",
                    label=f"phase 2 starts (ep {phase_split + 1})")
        ax1.axvline(best_ep,     color="gray",  linestyle="--",
                    label=f"best epoch {best_ep}")
        ax1.set_title("Accuracy — Transfer Learning")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(fontsize=8)

        # — Loss —
        ax2.plot(epochs, loss,     label="train")
        ax2.plot(epochs, val_loss, label="val")
        ax2.axvline(phase_split, color="blue",  linestyle=":",
                    label=f"phase 2 starts (ep {phase_split + 1})")
        ax2.axvline(best_ep,     color="gray",  linestyle="--",
                    label=f"best epoch {best_ep}")
        ax2.set_title("Loss — Transfer Learning")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        out_path = f"{RESULTS_DIR}/training_history_transfer.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[TransferTrainer] history plot saved → {out_path}")


if __name__ == "__main__":
    print("train_transfer.py loaded OK")
