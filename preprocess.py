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


class AudioPreprocessor:

    def __init__(self, augment: bool = False):
        self.augment = augment

    def process(self, path: str, seed: int | None = None) -> np.ndarray:
        rng = random.Random(seed)
        y = self._load_and_pad(path)

        if self.augment:
            y = self._augment_waveform(y, rng)

        mel_db = self._waveform_to_mel(y)
        d1 = librosa.feature.delta(mel_db)
        d2 = librosa.feature.delta(mel_db, order=2)

        img = np.stack(
            [self._normalize(self._resize(c)) for c in (mel_db, d1, d2)],
            axis=-1,
        )

        if self.augment:
            img = self._spec_augment(img, rng)

        return img.astype(np.float32)

    def _load_and_pad(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=config.TRIM_DB)
        target = int(config.SAMPLE_RATE * config.DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return y

    def _augment_waveform(self, y: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() < config.AUG_PROB:
            rate = rng.uniform(*config.TIME_STRETCH_RANGE)
            y = librosa.effects.time_stretch(y, rate=rate)
            target = int(config.SAMPLE_RATE * config.DURATION)
            if len(y) < target:
                y = np.pad(y, (0, target - len(y)))
            else:
                y = y[:target]

        if rng.random() < config.AUG_PROB:
            steps = rng.uniform(*config.PITCH_SHIFT_RANGE)
            y = librosa.effects.pitch_shift(
                y, sr=config.SAMPLE_RATE, n_steps=steps)

        if rng.random() < config.AUG_PROB:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(
                0, config.NOISE_STD, len(y))
            y = y + noise

        return y.astype(np.float32)

    @staticmethod
    def _spec_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
        img = img.copy()
        H, W = img.shape[:2]

        for _ in range(config.SPEC_AUG_NUM_MASKS):
            f = rng.randint(0, config.SPEC_AUG_FREQ_MASK)
            f0 = rng.randint(0, H - f)
            img[f0:f0 + f, :, :] = 0.0

            t = rng.randint(0, config.SPEC_AUG_TIME_MASK)
            t0 = rng.randint(0, W - t)
            img[:, t0:t0 + t, :] = 0.0

        return img

    @staticmethod
    def _waveform_to_mel(y: np.ndarray) -> np.ndarray:
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
        h, w = config.IMG_SIZE
        return cv2.resize(mel_db, (w, h),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        mean = img.mean()
        std  = img.std()
        return (img - mean) / (std + 1e-6)


class SpectrogramDataset:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.label_encoder = LabelEncoder()
        self._plain_proc  = AudioPreprocessor(augment=False)
        self._aug_proc    = AudioPreprocessor(augment=True)

    def build(self, force_recompute: bool = False):
        if not force_recompute and self._cache_exists():
            print("[SpectrogramDataset] Loading pre-computed arrays from disk...")
            return self._load_cache()

        print("[SpectrogramDataset] Pre-computing spectrograms...")
        X_train, y_train, X_val, y_val, X_test, y_test = self._run_pipeline()
        self._save_cache(X_train, y_train, X_val, y_val, X_test, y_test)
        self._print_summary(X_train, y_train, X_val, y_val, X_test, y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _run_pipeline(self):
        y_all         = self.label_encoder.fit_transform(self.df["emotion_name"])
        emotion_names = self.df["emotion_name"].values
        paths         = self.df["path"].values
        np.save(
            os.path.join(config.PROCESSED_DIR, "label_encoder_classes.npy"),
            self.label_encoder.classes_,
        )

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

        X_val,  y_val  = self._process_split(paths[idx_val],  y_all[idx_val],
                                              label="val")
        X_test, y_test = self._process_split(paths[idx_test], y_all[idx_test],
                                              label="test")
        X_train, y_train = self._process_train_split(
            paths[idx_train], y_all[idx_train], emotion_names[idx_train]
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _process_split(self, paths, labels, label: str):
        X = Parallel(n_jobs=-1)(
            delayed(self._plain_proc.process)(path)
            for path in tqdm(paths, desc=f"  {label:>5s} (plain)")
        )
        return np.array(X, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _process_train_split(self, paths, labels, emotion_names):
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

        shuffle_idx = np.random.RandomState(config.RANDOM_SEED).permutation(len(X_arr))
        return X_arr[shuffle_idx], y_arr[shuffle_idx]

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
