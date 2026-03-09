"""
data_loader.py — Step 1: Data Exploration
==========================================
Responsibilities
----------------
1. Scan the RAVDESS directory and decode emotion labels from filenames.
2. Display a bar-chart of samples per class (the "histogram of sample
   distributions" required by the project specification).
3. Show one representative mel-spectrogram per class.
4. Report basic dataset statistics and flag the class imbalance.

Design notes
------------
- RAVDESSParser  : pure data-parsing logic, no side-effects.
- DataVisualizer : all matplotlib / display concerns isolated here.
- DataLoader     : high-level facade; the only class client code needs.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from collections import Counter

import config


# ─────────────────────────────────────────────────────────────────────────────
class RAVDESSParser:
    """
    Parses RAVDESS filenames into structured metadata.

    RAVDESS filename format:
        MM-VV-EE-II-SS-RR-AA.wav
        ^^  ^^ ^^
        |   |  └─ Emotion (01-08)
        |   └──── Vocal channel (01=speech, 02=song)
        └──────── Modality (03=audio-only)

    We keep only speech files (VV=01) to avoid mixing speech and song.
    """

    FILENAME_EMOTION_POS = 2   # 0-indexed position after splitting on '-'

    def __init__(self, data_glob: str = config.DATA_DIR):
        self.data_glob = data_glob
        self._records: list[dict] = []

    # ── public ──────────────────────────────────────────────────────────────
    def parse(self) -> pd.DataFrame:
        """Walk all Actor_* folders, return a DataFrame of (path, emotion_id, emotion_name)."""
        wav_files = sorted(glob.glob(os.path.join(self.data_glob, "*.wav")))

        if not wav_files:
            raise FileNotFoundError(
                f"No .wav files found under '{self.data_glob}'.\n"
                "Check that DATA_DIR in config.py points to the right folder."
            )

        records = []
        skipped = 0
        for path in wav_files:
            fname   = os.path.splitext(os.path.basename(path))[0]
            parts   = fname.split("-")

            # Keep only audio-speech files (modality=03, vocal_channel=01)
            if parts[0] != "03" or parts[1] != "01":
                skipped += 1
                continue

            emotion_id   = parts[self.FILENAME_EMOTION_POS]
            emotion_name = config.EMOTION_MAP.get(emotion_id)
            if emotion_name is None:
                skipped += 1
                continue

            records.append({
                "path":         path,
                "emotion_id":   emotion_id,
                "emotion_name": emotion_name,
                "actor":        int(parts[6]),
            })

        print(f"[RAVDESSParser] Found {len(records)} valid speech files "
              f"({skipped} skipped).")

        self._records = records
        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
class DataVisualizer:
    """
    Handles all plots required for Step 1:
    - Bar chart of class distribution
    - One sample mel-spectrogram per class
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._emotion_order = list(config.EMOTION_MAP.values())

    # ── public ──────────────────────────────────────────────────────────────
    def plot_class_distribution(self, save_path: str | None = None):
        """
        Bar chart: samples per emotion class.
        Deliverable: 'Histogram of sample distributions' [cite: 22].
        """
        counts = (
            self.df["emotion_name"]
            .value_counts()
            .reindex(self._emotion_order)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index, counts.values,
                      color=plt.cm.tab10.colors[:len(counts)],
                      edgecolor="black", linewidth=0.6)

        # Annotate counts on top of each bar
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    str(count), ha="center", va="bottom", fontsize=10)

        ax.set_title("RAVDESS — Sample Distribution per Emotion Class",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotion", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_ylim(0, counts.max() * 1.15)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[DataVisualizer] Distribution plot saved → {save_path}")
        plt.show()

    def plot_sample_spectrograms(self, save_path: str | None = None):
        """
        Show one mel-spectrogram example per class.
        Deliverable: 'One example sample from each class' [cite: 21].
        """
        n_classes = len(self._emotion_order)
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("One Mel-Spectrogram Example per Emotion Class",
                     fontsize=15, fontweight="bold")

        for idx, emotion in enumerate(self._emotion_order, start=1):
            row = self.df[self.df["emotion_name"] == emotion].iloc[0]
            mel_db = self._load_mel(row["path"])

            ax = fig.add_subplot(2, 4, idx)
            librosa.display.specshow(
                mel_db,
                sr=config.SAMPLE_RATE,
                hop_length=config.HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                cmap="magma",
            )
            ax.set_title(f"{emotion.capitalize()}\n(Actor {row['actor']})",
                         fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[DataVisualizer] Spectrogram grid saved → {save_path}")
        plt.show()

    # ── private ─────────────────────────────────────────────────────────────
    @staticmethod
    def _load_mel(path: str) -> np.ndarray:
        """Load a .wav file and return a mel-spectrogram in dB scale."""
        y, sr = librosa.load(path, sr=config.SAMPLE_RATE,
                             duration=config.DURATION)
        # Pad if shorter than DURATION
        target_len = int(config.SAMPLE_RATE * config.DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        S = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
        )
        return librosa.power_to_db(S, ref=np.max)


# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """
    High-level facade.

    Usage
    -----
        loader = DataLoader()
        df     = loader.run_exploration()
    """

    def __init__(self):
        self.parser     = RAVDESSParser()
        self.df: pd.DataFrame | None = None
        self.visualizer: DataVisualizer | None = None

    # ── public ──────────────────────────────────────────────────────────────
    def run_exploration(self) -> pd.DataFrame:
        """Parse files, print stats, and generate both required plots."""

        # 1. Parse
        self.df         = self.parser.parse()
        self.visualizer = DataVisualizer(self.df)

        # 2. Print statistics
        self._print_statistics()

        # 3. Plots
        dist_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")
        spec_path = os.path.join(config.RESULTS_DIR, "sample_spectrograms.png")

        self.visualizer.plot_class_distribution(save_path=dist_path)
        self.visualizer.plot_sample_spectrograms(save_path=spec_path)

        return self.df

    # ── private ─────────────────────────────────────────────────────────────
    def _print_statistics(self):
        df = self.df
        counts = df["emotion_name"].value_counts()

        print("\n" + "=" * 55)
        print("  RAVDESS DATASET — EXPLORATION SUMMARY")
        print("=" * 55)
        print(f"  Total samples   : {len(df)}")
        print(f"  Actors          : {df['actor'].nunique()}")
        print(f"  Emotion classes : {df['emotion_name'].nunique()}")
        print()
        print("  Samples per class:")
        for emotion in config.EMOTION_MAP.values():
            n   = counts.get(emotion, 0)
            bar = "█" * (n // 6)
            print(f"    {emotion:<10s} {n:>4d}  {bar}")
        print()

        # Imbalance check
        min_c, max_c = counts.min(), counts.max()
        ratio = max_c / min_c
        if ratio > 1.5:
            print(f"  [!] CLASS IMBALANCE DETECTED  (ratio {ratio:.1f}:1)")
            print(f"      min={min_c} ({counts.idxmin()}), "
                  f"max={max_c} ({counts.idxmax()})")
            print("      Strategy: use class weights during training.")
        else:
            print("  [OK] Classes are balanced.")
        print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.run_exploration()
