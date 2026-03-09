"""
evaluate.py — Evaluation metrics and plots for the trained emotion CNN.

Produces:
  results/confusion_matrix.png
  results/classification_report.txt
  results/prediction_examples.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report as sk_report
import tensorflow as tf

from config import RESULTS_DIR


class Evaluator:
    def __init__(self, model: tf.keras.Model,
                 X_test: np.ndarray, y_test: np.ndarray,
                 class_names: np.ndarray):
        self.model       = model
        self.X_test      = X_test
        self.y_test      = y_test
        self.class_names = class_names
        self._y_pred     = None   # cached after first call

    # ------------------------------------------------------------------
    def _predict(self) -> np.ndarray:
        if self._y_pred is None:
            probs        = self.model.predict(self.X_test, verbose=0)
            self._y_pred = np.argmax(probs, axis=1)
        return self._y_pred

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f"\n[Evaluator] Test loss:     {loss:.4f}")
        print(f"[Evaluator] Test accuracy: {acc:.4f}  ({acc*100:.1f}%)")
        return {"loss": loss, "accuracy": acc}

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self) -> None:
        y_pred = self._predict()
        cm     = confusion_matrix(self.y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (%)")
        fig.tight_layout()

        out = f"{RESULTS_DIR}/confusion_matrix.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[Evaluator] confusion matrix saved → {out}")

    # ------------------------------------------------------------------
    def classification_report(self) -> None:
        y_pred  = self._predict()
        report  = sk_report(self.y_test, y_pred, target_names=self.class_names)
        print("\n[Evaluator] Classification Report:\n")
        print(report)

        out = f"{RESULTS_DIR}/classification_report.txt"
        with open(out, "w") as f:
            f.write(report)
        print(f"[Evaluator] classification report saved → {out}")

    # ------------------------------------------------------------------
    def plot_examples(self, n: int = 5) -> None:
        y_pred   = self._predict()
        correct  = np.where(y_pred == self.y_test)[0]
        wrong    = np.where(y_pred != self.y_test)[0]

        # sample up to n from each group
        rng      = np.random.default_rng(42)
        correct  = rng.choice(correct, size=min(n, len(correct)), replace=False)
        wrong    = rng.choice(wrong,   size=min(n, len(wrong)),   replace=False)

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        fig.suptitle("Top row: Correct   |   Bottom row: Incorrect", fontsize=12)

        for col, idx in enumerate(correct):
            ax = axes[0, col]
            ax.imshow(self.X_test[idx, :, :, 0], cmap="gray", vmin=-3, vmax=3,
                      origin="lower", aspect="auto")
            ax.set_title(f"True: {self.class_names[self.y_test[idx]]}\n"
                         f"Pred: {self.class_names[y_pred[idx]]}", fontsize=8)
            ax.axis("off")

        for col, idx in enumerate(wrong):
            ax = axes[1, col]
            ax.imshow(self.X_test[idx, :, :, 0], cmap="gray", vmin=-3, vmax=3,
                      origin="lower", aspect="auto")
            ax.set_title(f"True: {self.class_names[self.y_test[idx]]}\n"
                         f"Pred: {self.class_names[y_pred[idx]]}", fontsize=8)
            ax.axis("off")

        fig.tight_layout()
        out = f"{RESULTS_DIR}/prediction_examples.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[Evaluator] prediction examples saved → {out}")


if __name__ == "__main__":
    print("evaluate.py loaded OK")
