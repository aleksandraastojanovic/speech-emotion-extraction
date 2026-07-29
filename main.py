"""
main.py — Full pipeline entry point for RAVDESS emotion CNN.

Usage:
    python main.py                              # CNN from scratch (default)
    python main.py --mode transfer              # EfficientNetB0 transfer learning
    python main.py --mode ensemble              # average all model_*.keras (no training)
    python main.py --skip-preprocess            # use cached processed/ .npy files
    python main.py --eval-only                  # skip training, evaluate saved model
    python main.py --mode transfer --eval-only  # evaluate saved transfer model
"""

import argparse
import glob
import os
import tensorflow as tf

from config import BASE_DIR, MODEL_PATH, MODEL_TRANSFER_PATH, RANDOM_SEED
from data_loader import DataLoader
from preprocess import SpectrogramDataset
from model import EmotionCNN
from train import Trainer
from model_transfer import TransferEmotionModel
from train_transfer import TransferTrainer
from evaluate import Evaluator, SoftmaxAverageEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAVDESS Speech Emotion Recognition — CNN Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["cnn", "transfer", "ensemble"],
        default="cnn",
        help="cnn = train from scratch (default) | transfer = EfficientNetB0 "
             "fine-tuning | ensemble = average all saved model_*.keras",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip audio→spectrogram conversion and use cached .npy files",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the saved model directly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(RANDOM_SEED)   # reproducible weight init

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
    print(f"\n=== [3/4] Training  (mode: {args.mode}) ===")

    if args.mode == "cnn":
        model_path = MODEL_PATH
        if args.eval_only:
            print("  --eval-only set: loading saved CNN model, skipping training.")
            model = tf.keras.models.load_model(model_path)
        else:
            model   = EmotionCNN().build_model()
            trainer = Trainer(model, X_train, y_train, X_val, y_val)
            history = trainer.train()
            trainer.plot_history(history)
            model = tf.keras.models.load_model(model_path)
            print("  Best CNN model reloaded from disk.")

    elif args.mode == "transfer":
        model_path = MODEL_TRANSFER_PATH
        if args.eval_only:
            print("  --eval-only set: loading saved transfer model, skipping training.")
            model = tf.keras.models.load_model(model_path)
        else:
            trainer = TransferTrainer(X_train, y_train, X_val, y_val)
            h1, h2  = trainer.train()
            trainer.plot_history(h1, h2)
            model = tf.keras.models.load_model(model_path)
            print("  Best transfer model reloaded from disk.")

    else:  # ensemble — no training, average every saved model_*.keras
        files = sorted(glob.glob(os.path.join(BASE_DIR, "model_*.keras")))
        if len(files) < 2:
            raise SystemExit(
                "Ensemble mode needs at least 2 model_*.keras files in the "
                "project root — train the cnn and transfer models first."
            )
        print("  Ensemble members:")
        for f in files:
            print(f"    - {os.path.basename(f)}")
        model = SoftmaxAverageEnsemble(
            [tf.keras.models.load_model(f) for f in files]
        )

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
