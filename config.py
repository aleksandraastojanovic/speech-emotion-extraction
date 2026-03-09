"""
config.py — Central configuration for all hyperparameters and paths.
Keeping everything here means a single place to change any setting.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "ravdess", "Actor_*")
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODEL_PATH     = os.path.join(BASE_DIR, "model_best.keras")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)

# ── Audio / Spectrogram ────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # Hz — native RAVDESS sample rate
DURATION       = 3.0     # seconds — crop/pad all clips to this length
N_MELS         = 128     # mel filter bank size  → image height
HOP_LENGTH     = 512     # STFT hop → controls time resolution
N_FFT          = 2048    # STFT window size
IMG_SIZE       = (128, 128)  # final (H, W) fed to the CNN

# ── Dataset ────────────────────────────────────────────────────────────────
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
NUM_CLASSES = len(EMOTION_MAP)   # 8

# ── Training ───────────────────────────────────────────────────────────────
TEST_SIZE      = 0.10   # 10 % test
VAL_SIZE       = 0.15   # 15 % validation  (taken from remaining 90 %)
RANDOM_SEED    = 42
BATCH_SIZE     = 32
EPOCHS                = 60
LEARNING_RATE         = 1e-3
EARLY_STOPPING_PATIENCE  = 20   # increased from 10 — val set is noisy (120 samples)
REDUCE_LR_PATIENCE       = 10   # increased from 5

# ── Augmentation ──────────────────────────────────────────────────────────
AUG_PROB       = 0.50   # probability of applying each augmentation
TIME_STRETCH_RANGE = (0.85, 1.15)
PITCH_SHIFT_RANGE  = (-2, 2)     # semitones
NOISE_STD          = 0.005

# ── SpecAugment ────────────────────────────────────────────────────────────
SPEC_AUG_FREQ_MASK  = 20   # max mel bins to zero per frequency mask
SPEC_AUG_TIME_MASK  = 20   # max time frames to zero per time mask
SPEC_AUG_NUM_MASKS  = 2    # number of masks of each type applied per sample

# ── Speaker-independent split ──────────────────────────────────────────────
VAL_ACTORS   = [21, 22]   # held-out actors for validation
TEST_ACTORS  = [23, 24]   # held-out actors for test
# Train: all remaining actors (1-20)

# ── Focal loss ────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0   # focusing parameter — higher = more attention on hard examples

# ── Per-class extra augmentation ──────────────────────────────────────────
# Default for unlisted classes: 1 plain + 1 aug = 2× total
# Hard classes get extra aug passes to counteract low recall
AUG_EXTRA = {
    "fearful":   2,   # 1 plain + 2 aug = 3×  (21% recall in baseline)
    "sad":       2,   # 1 plain + 2 aug = 3×  (16% recall in baseline)
    "surprised": 2,   # 1 plain + 2 aug = 3×  (30% recall in baseline)
    "happy":     2,   # 1 plain + 2 aug = 3×  (37% recall in baseline)
    "neutral":   3,   # 1 plain + 3 aug = 4×  (minority class, 96 samples)
}
