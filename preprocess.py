"""
preprocess.py — Step 2: Audio Preprocessing & Dataset Construction
===================================================================

Classes
-------
AudioPreprocessor
    Converts a single .wav file into a normalized 128×128×3 image
    (mel-spectrogram + delta + delta-delta channels).
    Optionally applies waveform-level augmentation before conversion.

SpectrogramDataset
    Orchestrates the full pipeline across all files:
      - Calls AudioPreprocessor for every sample (in parallel via joblib)
      - Applies augmentation strategy (class-balanced oversampling)
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
import random
import numpy as np
import pandas as pd
import librosa
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


# ─────────────────────────────────────────────────────────────────────────────
class AudioPreprocessor:
    """
    Converts one .wav file → normalized (128, 128, 3) numpy array.

    Pipeline per file
    -----------------
    1. Load full waveform at fixed sample rate
    2. Trim leading/trailing silence (RAVDESS clips start with ~0.5 s
       of silence — without trimming, a quarter of the 3 s window is
       wasted and the end of the utterance is often cut off)
    3. Crop or zero-pad to exactly DURATION seconds
    4. [Optional] Apply waveform augmentation
    5. Compute mel-spectrogram (N_MELS × time_frames), convert to dB
    6. Add delta and delta-delta channels (velocity/acceleration of the
       spectral envelope over time — standard SER features that capture
       the *dynamics* of speech, which a static spectrogram misses)
    7. Resize each channel to IMG_SIZE, Z-score each independently
    8. Stack → (H, W, 3)

    Why per-sample Z-score?
    -----------------------
    Different recordings have different loudness levels. Z-scoring each
    spectrogram independently centres and scales it so every sample enters
    the network on equal footing, regardless of recording volume. This
    prevents the network from using absolute loudness as a shortcut.
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
    def process(self, path: str, seed: int | None = None) -> np.ndarray:
        """Load a .wav file and return a (128, 128, 3) float32 array.

        `seed` makes augmentation deterministic per sample, so the exact
        training set can be regenerated (also safe under parallel workers).
        """
        rng = random.Random(seed)
        y = self._load_and_pad(path)

        if self.augment:
            y = self._augment_waveform(y, rng)

        mel_db = self._waveform_to_mel(y)
        # Delta (velocity) and delta-delta (acceleration) along time
        d1 = librosa.feature.delta(mel_db)
        d2 = librosa.feature.delta(mel_db, order=2)

        img = np.stack(
            [self._normalize(self._resize(c)) for c in (mel_db, d1, d2)],
            axis=-1,
        )

        if self.augment:
            img = self._spec_augment(img, rng)

        return img.astype(np.float32)  # (H, W, 3)

    # ── private — loading ────────────────────────────────────────────────────
    def _load_and_pad(self, path: str) -> np.ndarray:
        """Load waveform, trim silence; crop if too long, zero-pad if short."""
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=config.TRIM_DB)
        target = int(config.SAMPLE_RATE * config.DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return y

    # ── private — augmentation ───────────────────────────────────────────────
    def _augment_waveform(self, y: np.ndarray, rng: random.Random) -> np.ndarray:
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
        if rng.random() < config.AUG_PROB:
            rate = rng.uniform(*config.TIME_STRETCH_RANGE)
            y = librosa.effects.time_stretch(y, rate=rate)
            # Re-pad/crop after stretch since length changes
            target = int(config.SAMPLE_RATE * config.DURATION)
            if len(y) < target:
                y = np.pad(y, (0, target - len(y)))
            else:
                y = y[:target]

        # Pitch shift
        if rng.random() < config.AUG_PROB:
            steps = rng.uniform(*config.PITCH_SHIFT_RANGE)
            y = librosa.effects.pitch_shift(
                y, sr=config.SAMPLE_RATE, n_steps=steps)

        # Gaussian noise
        if rng.random() < config.AUG_PROB:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(
                0, config.NOISE_STD, len(y))
            y = y + noise

        return y.astype(np.float32)

    # ── private — SpecAugment ────────────────────────────────────────────────
    @staticmethod
    def _spec_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
        """
        SpecAugment: mask random frequency bands and time strips.

        Applied AFTER Z-score normalization so masked values (0) equal the
        approximate mean, avoiding any distributional shift from masking.
        Masks span all channels so mel/delta stay aligned.

        Frequency masking: zero out up to SPEC_AUG_FREQ_MASK consecutive
        mel bins, repeated SPEC_AUG_NUM_MASKS times. Forces the model to
        recognise emotions without relying on any specific frequency region.

        Time masking: zero out up to SPEC_AUG_TIME_MASK consecutive time
        frames, repeated SPEC_AUG_NUM_MASKS times. Forces temporal robustness.
        """
        img = img.copy()
        H, W = img.shape[:2]  # (128, 128)

        for _ in range(config.SPEC_AUG_NUM_MASKS):
            # Frequency mask
            f = rng.randint(0, config.SPEC_AUG_FREQ_MASK)
            f0 = rng.randint(0, H - f)
            img[f0:f0 + f, :, :] = 0.0

            # Time mask
            t = rng.randint(0, config.SPEC_AUG_TIME_MASK)
            t0 = rng.randint(0, W - t)
            img[:, t0:t0 + t, :] = 0.0

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
    - Every training sample gets AUG_EXTRA augmented copies; neutral
      (the minority class) gets double, so the final training set is
      exactly class-balanced.
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
                                              label="val")
        X_test, y_test = self._process_split(paths[idx_test], y_all[idx_test],
                                              label="test")
        X_train, y_train = self._process_train_split(
            paths[idx_train], y_all[idx_train], emotion_names[idx_train]
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _process_split(self, paths, labels, label: str):
        """Process a list of paths without augmentation (parallel)."""
        X = Parallel(n_jobs=-1)(
            delayed(self._plain_proc.process)(path)
            for path in tqdm(paths, desc=f"  {label:>5s} (plain)")
        )
        return np.array(X, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _process_train_split(self, paths, labels, emotion_names):
        """
        Class-balanced augmentation (see config.AUG_EXTRA):
        every file gets 1 plain + AUG_EXTRA augmented copies; neutral gets
        double the copies so all 8 classes end up with the same count.

        Each augmented copy has a deterministic seed (RANDOM_SEED + job
        index), so the exact training set is reproducible even though
        processing runs on all cores.
        """
        # Build one job per (path, label, seed) — plain copies use seed None
        jobs = [(p, l, None) for p, l in zip(paths, labels)]
        for i, (path, lbl, ename) in enumerate(zip(paths, labels, emotion_names)):
            n_aug = config.AUG_EXTRA.get(ename, config.AUG_EXTRA_DEFAULT)
            for k in range(n_aug):
                jobs.append((path, lbl, config.RANDOM_SEED + i * 100 + k))

        X = Parallel(n_jobs=-1)(
            delayed(self._aug_proc.process if seed is not None
                    else self._plain_proc.process)(path, seed)
            for path, _, seed in tqdm(jobs, desc="  train (plain+aug)")
        )
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array([lbl for _, lbl, _ in jobs], dtype=np.int32)

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
