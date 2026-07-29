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
MODEL_PATH          = os.path.join(BASE_DIR, "model_best.keras")
MODEL_TRANSFER_PATH = os.path.join(BASE_DIR, "model_transfer_best.keras")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)

# ── Audio / Spectrogram ────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # Hz — RAVDESS (48 kHz native) is resampled to this
DURATION       = 3.0     # seconds — crop/pad all clips to this length
TRIM_DB        = 30      # silence-trim threshold (dB below peak)
N_MELS         = 128     # mel filter bank size  → image height
HOP_LENGTH     = 512     # STFT hop → controls time resolution
N_FFT          = 2048    # STFT window size
IMG_SIZE       = (128, 128)  # final (H, W) fed to the CNN
N_CHANNELS     = 3       # mel + delta + delta-delta

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
BATCH_SIZE     = 64   # larger batch → steadier BatchNorm stats & gradients
EPOCHS                = 80
LEARNING_RATE         = 3e-4   # 1e-3 was unstable: val_loss spiked, early
                               # stopping restored a deeply underfit epoch

# ── Transfer learning ──────────────────────────────────────────────────────
PHASE1_EPOCHS  = 40   # frozen base: val acc was still rising at 20 epochs
PHASE2_EPOCHS  = 60   # unfreeze top layers: val acc was still rising at 40
TRANSFER_LR    = 1e-3  # Phase 1 learning rate
FINETUNE_LR    = 5e-5  # Phase 2 LR — 1e-4 overfit within a few epochs
EARLY_STOPPING_PATIENCE  = 20   # val set is small (216) and noisy
REDUCE_LR_PATIENCE       = 5    # halve LR quickly when val_loss stalls

# ── Augmentation ──────────────────────────────────────────────────────────
AUG_PROB       = 0.50   # probability of applying each augmentation
TIME_STRETCH_RANGE = (0.85, 1.15)
PITCH_SHIFT_RANGE  = (-2, 2)     # semitones
NOISE_STD          = 0.005

# ── SpecAugment ────────────────────────────────────────────────────────────
SPEC_AUG_FREQ_MASK  = 20   # max mel bins to zero per frequency mask
SPEC_AUG_TIME_MASK  = 20   # max time frames to zero per time mask
SPEC_AUG_NUM_MASKS  = 2    # number of masks of each type applied per sample

# ── Focal loss ────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0   # focusing parameter — higher = more attention on hard examples

# ── Per-class extra augmentation ──────────────────────────────────────────
# Every class: 1 plain + AUG_EXTRA aug copies. Neutral has half the samples
# of every other class (96 vs 192), so it gets double the aug passes —
# the final training set is exactly class-balanced (432 per class).
AUG_EXTRA_DEFAULT = 2   # 1 plain + 2 aug = 3×  → 144·3 = 432 per class
AUG_EXTRA = {
    "neutral": 5,        # 1 plain + 5 aug = 6×  →  72·6 = 432
}
