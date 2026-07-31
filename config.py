import os

# ── Putanje ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "ravdess", "Actor_*")
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")   # kes .npy
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODEL_PATH          = os.path.join(BASE_DIR, "model_best.keras")
MODEL_TRANSFER_PATH = os.path.join(BASE_DIR, "model_transfer_best.keras")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)

# ── Audio / spektrogram ─────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # ravdess je 48k, resample
DURATION       = 3.0
TRIM_DB        = 30      # ispod -30dB od peaka = tisina, sece se
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048
IMG_SIZE       = (128, 128)
N_CHANNELS     = 3       # mel + delta + delta-delta

# ── Dataset ─────────────────────────────────────────────────────────────────
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
NUM_CLASSES = len(EMOTION_MAP)

# ── Trening ─────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.10
VAL_SIZE       = 0.15
RANDOM_SEED    = 42
BATCH_SIZE     = 64     # na 32 batchnorm statistike previse skacu
EPOCHS         = 80
LEARNING_RATE  = 3e-4   # sa 1e-3 divergira

# ── Transfer learning ───────────────────────────────────────────────────────
PHASE1_EPOCHS  = 40
PHASE2_EPOCHS  = 60
TRANSFER_LR    = 1e-3
FINETUNE_LR    = 5e-5   # 1e-4 odmah overfituje
EARLY_STOPPING_PATIENCE  = 20   # val je mali pa metrika skace
REDUCE_LR_PATIENCE       = 5

# ── Augmentacija ────────────────────────────────────────────────────────────
AUG_PROB       = 0.50
TIME_STRETCH_RANGE = (0.85, 1.15)
PITCH_SHIFT_RANGE  = (-2, 2)  
NOISE_STD          = 0.005

# ── SpecAugment ─────────────────────────────────────────────────────────────
SPEC_AUG_FREQ_MASK  = 20
SPEC_AUG_TIME_MASK  = 20
SPEC_AUG_NUM_MASKS  = 2

# ── Focal loss ──────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0

# ── Balans klasa ────────────────────────────────────────────────────────────
# neutral ima duplo manje snimaka -> duplo vise aug kopija
# 144*3 = 72*6 = 432 po klasi
AUG_EXTRA_DEFAULT = 2
AUG_EXTRA = {
    "neutral": 5,
}
