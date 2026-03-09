"""
main.py — Full pipeline entry point for RAVDESS emotion CNN.

Usage:
    python main.py                    # full pipeline
    python main.py --skip-preprocess  # use cached processed/ .npy files
    python main.py --eval-only        # skip training, evaluate saved model
"""

import argparse
import tensorflow as tf

from config import MODEL_PATH
from data_loader import DataLoader
from preprocess import SpectrogramDataset
from model import EmotionCNN
from train import Trainer
from evaluate import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAVDESS Speech Emotion Recognition — CNN Pipeline"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip audio→spectrogram conversion and use cached .npy files",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the saved model_best.keras directly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Data exploration ────────────────────────────────────────────
    print("\n=== [1/4] Data Exploration ===")
    df = DataLoader().run_exploration()

    # ── 2. Preprocessing ──────────────────────────────────────────────
    print("\n=== [2/4] Preprocessing ===")
    dataset = SpectrogramDataset(df)
    force   = not args.skip_preprocess
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.build(
        force_recompute=force
    )
    class_names = dataset.label_encoder.classes_
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Classes: {list(class_names)}")

    # ── 3. Training ───────────────────────────────────────────────────
    print("\n=== [3/4] Training ===")
    if args.eval_only:
        print("  --eval-only set: loading saved model, skipping training.")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model   = EmotionCNN().build_model()
        trainer = Trainer(model, X_train, y_train, X_val, y_val)
        history = trainer.train()
        trainer.plot_history(history)
        # Always load best weights saved by ModelCheckpoint
        model = tf.keras.models.load_model(MODEL_PATH)
        print("  Best model reloaded from disk.")

    # ── 4. Evaluation ─────────────────────────────────────────────────
    print("\n=== [4/4] Evaluation ===")
    evaluator = Evaluator(model, X_test, y_test, class_names)
    evaluator.evaluate()
    evaluator.plot_confusion_matrix()
    evaluator.classification_report()
    evaluator.plot_examples()

    print("\nDone. Results saved to results/")


if __name__ == "__main__":
    main()
