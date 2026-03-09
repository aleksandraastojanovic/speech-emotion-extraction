"""
preprocess.py — Step 2: Audio Preprocessing & Dataset Construction
===================================================================

Classes
-------
AudioPreprocessor
    Converts a single .wav file into a normalized 128×128 mel-spectrogram.
    Optionally applies waveform-level augmentation before conversion.

SpectrogramDataset
    Orchestrates the full pipeline across all files:
      - Calls AudioPreprocessor for every sample
      - Applies augmentation strategy (all train + extra for neutral)
      - Stratified train / val / test split
      - Saves / loads pre-computed .npy arrays to avoid re-running librosa

Design note on augmentation
----------------------------
Augmentation is applied to the RAW WAVEFORM before mel conversion.
Augmenting spectrogram pixels directly produces unphysical artifacts
(e.g. blurring time-frequency ridges), whereas waveform augmentation
always yields a valid audio signal.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


# ─────────────────────────────────────────────────────────────────────────────
class AudioPreprocessor:
    """
    Converts one .wav file → normalized (128, 128, 1) numpy array.

    Pipeline per file
    -----------------
    1. Load waveform at fixed sample rate
    2. Crop or zero-pad to exactly DURATION seconds
    3. [Optional] Apply waveform augmentation
    4. Compute mel-spectrogram (N_MELS × time_frames)
    5. Convert power → dB scale
    6. Resize to IMG_SIZE with bilinear interpolation
    7. Per-sample Z-score normalization
    8. Add channel dim → (H, W, 1)

    Why per-sample Z-score?
    -----------------------
    Different recordings have different loudness levels. Z-scoring each
    spectrogram independently centres and scales it so every sample enters
    the network on equal footing, regardless of recording volume. This
    prevents the network from using absolute loudness as a shortcut.
    It also mirrors what BatchNorm does inside the network, giving it a
    clean, well-conditioned input distribution.
    """

    def __init__(self, augment: bool = False):
        """
        Parameters
        ----------
        augment : bool
            If True, randomly applies time-stretch, pitch-shift, and/or
            Gaussian noise to the waveform before spectrogram conversion.
        """
        self.augment = augment

    # ── public ──────────────────────────────────────────────────────────────
    def process(self, path: str) -> np.ndarray:
        """Load a .wav file and return a (128, 128, 1) float32 array."""
        y = self._load_and_pad(path)

        if self.augment:
            y = self._augment_waveform(y)

        mel_db = self._waveform_to_mel(y)
        mel_resized = self._resize(mel_db)
        mel_norm = self._normalize(mel_resized)

        if self.augment:
            mel_norm = self._spec_augment(mel_norm)

        return mel_norm[..., np.newaxis].astype(np.float32)  # (H, W, 1)

    # ── private — loading ────────────────────────────────────────────────────
    def _load_and_pad(self, path: str) -> np.ndarray:
        """Load waveform; crop if too long, zero-pad if too short."""
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE,
                            duration=config.DURATION)
        target = int(config.SAMPLE_RATE * config.DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return y

    # ── private — augmentation ───────────────────────────────────────────────
    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        """
        Three independent augmentations, each applied with probability AUG_PROB.

        Time stretching
            Speeds up or slows down audio without altering pitch.
            Rate < 1 → slower (stretched); rate > 1 → faster (compressed).
            Horizontally shifts patterns in the spectrogram.

        Pitch shifting
            Shifts pitch up/down by N semitones without changing duration.
            Moves all frequency content vertically in the spectrogram.

        Gaussian noise
            Adds very low-amplitude random noise. Simulates microphone noise
            and prevents the network from relying on pure silence regions.
        """
        # Time stretch
        if random.random() < config.AUG_PROB:
            rate = random.uniform(*config.TIME_STRETCH_RANGE)
            y = librosa.effects.time_stretch(y, rate=rate)
            # Re-pad/crop after stretch since length changes
            target = int(config.SAMPLE_RATE * config.DURATION)
            if len(y) < target:
                y = np.pad(y, (0, target - len(y)))
            else:
                y = y[:target]

        # Pitch shift
        if random.random() < config.AUG_PROB:
            steps = random.uniform(*config.PITCH_SHIFT_RANGE)
            y = librosa.effects.pitch_shift(
                y, sr=config.SAMPLE_RATE, n_steps=steps)

        # Gaussian noise
        if random.random() < config.AUG_PROB:
            noise = np.random.normal(0, config.NOISE_STD, len(y))
            y = y + noise

        return y.astype(np.float32)

    # ── private — SpecAugment ────────────────────────────────────────────────
    @staticmethod
    def _spec_augment(img: np.ndarray) -> np.ndarray:
        """
        SpecAugment: mask random frequency bands and time strips.

        Applied AFTER Z-score normalization so masked values (0) equal the
        approximate mean, avoiding any distributional shift from masking.

        Frequency masking: zero out up to SPEC_AUG_FREQ_MASK consecutive
        mel bins, repeated SPEC_AUG_NUM_MASKS times. Forces the model to
        recognise emotions without relying on any specific frequency region.

        Time masking: zero out up to SPEC_AUG_TIME_MASK consecutive time
        frames, repeated SPEC_AUG_NUM_MASKS times. Forces temporal robustness.
        """
        img = img.copy()
        H, W = img.shape  # (128, 128)

        for _ in range(config.SPEC_AUG_NUM_MASKS):
            # Frequency mask
            f = random.randint(0, config.SPEC_AUG_FREQ_MASK)
            f0 = random.randint(0, H - f)
            img[f0:f0 + f, :] = 0.0

            # Time mask
            t = random.randint(0, config.SPEC_AUG_TIME_MASK)
            t0 = random.randint(0, W - t)
            img[:, t0:t0 + t] = 0.0

        return img

    # ── private — spectrogram ────────────────────────────────────────────────
    @staticmethod
    def _waveform_to_mel(y: np.ndarray) -> np.ndarray:
        """Waveform → mel-spectrogram in dB scale."""
        S = librosa.feature.melspectrogram(
            y=y,
            sr=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
        )
        return librosa.power_to_db(S, ref=np.max)

    @staticmethod
    def _resize(mel_db: np.ndarray) -> np.ndarray:
        """Resize to IMG_SIZE (H, W) using bilinear interpolation."""
        h, w = config.IMG_SIZE
        return cv2.resize(mel_db, (w, h),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """
        Per-sample Z-score normalization.
        Each spectrogram is normalized with its own mean and std so that
        different recording loudness levels don't affect the network input.
        A small epsilon prevents division by zero for silent clips.
        """
        mean = img.mean()
        std  = img.std()
        return (img - mean) / (std + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
class SpectrogramDataset:
    """
    Orchestrates the full preprocessing pipeline across all files.

    Augmentation strategy
    ---------------------
    - Every training sample is augmented once → 2× training set size.
    - Class imbalance (neutral) is handled via class weights at training
      time, NOT by extra augmentation here.
    - Validation and test sets are NEVER augmented (they must reflect
      real-world distribution).

    Saved files (in PROCESSED_DIR)
    --------------------------------
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    label_encoder_classes.npy   ← maps integer indices → emotion names
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Output of RAVDESSParser.parse() with columns:
            [path, emotion_id, emotion_name, actor]
        """
        self.df = df
        self.label_encoder = LabelEncoder()
        self._plain_proc  = AudioPreprocessor(augment=False)
        self._aug_proc    = AudioPreprocessor(augment=True)

    # ── public ──────────────────────────────────────────────────────────────
    def build(self, force_recompute: bool = False):
        """
        Build and save all splits. Returns (X_train, y_train, X_val, y_val,
        X_test, y_test) as numpy arrays ready for model.fit().

        Parameters
        ----------
        force_recompute : bool
            If True, re-runs librosa even if .npy files already exist.
        """
        if not force_recompute and self._cache_exists():
            print("[SpectrogramDataset] Loading pre-computed arrays from disk...")
            return self._load_cache()

        print("[SpectrogramDataset] Pre-computing spectrograms...")
        X_train, y_train, X_val, y_val, X_test, y_test = self._run_pipeline()
        self._save_cache(X_train, y_train, X_val, y_val, X_test, y_test)
        self._print_summary(X_train, y_train, X_val, y_val, X_test, y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ── private — pipeline ───────────────────────────────────────────────────
    def _run_pipeline(self):
        # 1. Encode labels
        y_all         = self.label_encoder.fit_transform(self.df["emotion_name"])
        emotion_names = self.df["emotion_name"].values
        paths         = self.df["path"].values
        np.save(
            os.path.join(config.PROCESSED_DIR, "label_encoder_classes.npy"),
            self.label_encoder.classes_,
        )

        # 2. Stratified random split (speaker-dependent)
        X_idx = np.arange(len(paths))
        idx_trainval, idx_test = train_test_split(
            X_idx, test_size=config.TEST_SIZE,
            stratify=y_all, random_state=config.RANDOM_SEED,
        )
        val_fraction = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_fraction,
            stratify=y_all[idx_trainval], random_state=config.RANDOM_SEED,
        )

        # 3. Process splits
        X_val,  y_val  = self._process_split(paths[idx_val],  y_all[idx_val],
                                              augment=False, label="val")
        X_test, y_test = self._process_split(paths[idx_test], y_all[idx_test],
                                              augment=False, label="test")
        X_train, y_train = self._process_train_split(
            paths[idx_train], y_all[idx_train], emotion_names[idx_train]
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _process_split(self, paths, labels, augment: bool, label: str):
        """Process a list of paths without augmentation."""
        proc = self._aug_proc if augment else self._plain_proc
        X, y = [], []
        for path, lbl in tqdm(zip(paths, labels),
                               total=len(paths), desc=f"  {label:>5s} (plain)"):
            X.append(proc.process(path))
            y.append(lbl)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def _process_train_split(self, paths, labels, emotion_names):
        """
        Per-class augmentation weighting.

        Default (angry, calm, disgust): 1 plain + 1 aug = 2× total
        Hard classes (fearful, sad, surprised, happy): 1 plain + 2 aug = 3×
        Minority class (neutral): 1 plain + 3 aug = 4×

        Multipliers defined in config.AUG_EXTRA. Hard classes had the lowest
        recall in the baseline run; extra copies give the model more signal
        on those categories without touching the val/test splits.
        """
        X, y = [], []

        # Pass 1 — plain (all classes)
        for path, lbl in tqdm(zip(paths, labels),
                               total=len(paths), desc="  train (plain) "):
            X.append(self._plain_proc.process(path))
            y.append(lbl)

        # Passes 2..max_aug — only for classes that need that many copies
        max_aug = max(config.AUG_EXTRA.values())
        for aug_idx in range(1, max_aug + 1):
            desc = f"  train (aug×{aug_idx})"
            for path, lbl, ename in tqdm(zip(paths, labels, emotion_names),
                                          total=len(paths), desc=desc):
                if aug_idx <= config.AUG_EXTRA.get(ename, 1):
                    X.append(self._aug_proc.process(path))
                    y.append(lbl)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        # Shuffle so augmented copies are not contiguous
        shuffle_idx = np.random.RandomState(config.RANDOM_SEED).permutation(len(X_arr))
        return X_arr[shuffle_idx], y_arr[shuffle_idx]

    # ── private — cache ──────────────────────────────────────────────────────
    def _cache_exists(self) -> bool:
        required = ["X_train.npy", "y_train.npy",
                    "X_val.npy",   "y_val.npy",
                    "X_test.npy",  "y_test.npy"]
        return all(
            os.path.exists(os.path.join(config.PROCESSED_DIR, f))
            for f in required
        )

    def _save_cache(self, X_train, y_train, X_val, y_val, X_test, y_test):
        p = config.PROCESSED_DIR
        np.save(os.path.join(p, "X_train.npy"), X_train)
        np.save(os.path.join(p, "y_train.npy"), y_train)
        np.save(os.path.join(p, "X_val.npy"),   X_val)
        np.save(os.path.join(p, "y_val.npy"),   y_val)
        np.save(os.path.join(p, "X_test.npy"),  X_test)
        np.save(os.path.join(p, "y_test.npy"),  y_test)
        print(f"[SpectrogramDataset] Arrays saved to '{config.PROCESSED_DIR}/'")

    def _load_cache(self):
        p = config.PROCESSED_DIR
        X_train = np.load(os.path.join(p, "X_train.npy"))
        y_train = np.load(os.path.join(p, "y_train.npy"))
        X_val   = np.load(os.path.join(p, "X_val.npy"))
        y_val   = np.load(os.path.join(p, "y_val.npy"))
        X_test  = np.load(os.path.join(p, "X_test.npy"))
        y_test  = np.load(os.path.join(p, "y_test.npy"))

        # Reload label encoder classes
        classes = np.load(
            os.path.join(p, "label_encoder_classes.npy"), allow_pickle=True)
        self.label_encoder.classes_ = classes

        self._print_summary(X_train, y_train, X_val, y_val, X_test, y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _print_summary(self, X_train, y_train, X_val, y_val, X_test, y_test):
        from collections import Counter
        classes = self.label_encoder.classes_

        print("\n" + "=" * 55)
        print("  PREPROCESSING SUMMARY")
        print("=" * 55)
        print(f"  Spectrogram shape : {X_train.shape[1:]}")
        print(f"  Train samples     : {len(X_train)}")
        print(f"  Val   samples     : {len(X_val)}")
        print(f"  Test  samples     : {len(X_test)}")
        print()
        print("  Train class counts (after augmentation):")
        for idx, name in enumerate(classes):
            print(f"    {name:<10s} {Counter(y_train)[idx]:>4d}")
        print("=" * 55 + "\n")
