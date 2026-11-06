"""Configuration for the self-contained S70 strict-valence project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "s70"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

EMOTION_CODE = {
    "b": "sad",
    "g": "happy",
    "h": "fear",
    "j": "surprise",
    "p": "neutral",
    "s": "angry",
    "x": "disgust",
}

# Strict valence: surprise is excluded because its valence is ambiguous.
STRICT_VALENCE = {
    "happy": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
}

VALENCE_CLASSES = ("negative", "neutral", "positive")
VALENCE_TO_INDEX = {label: index for index, label in enumerate(VALENCE_CLASSES)}
INDEX_TO_VALENCE = {index: label for label, index in VALENCE_TO_INDEX.items()}

SPLIT_SPEAKERS = {
    "train": (
        "F11", "F12", "F17", "F18", "F20", "M02", "M03",
        "M04", "M06", "M08", "M09", "M13", "M14", "M19",
    ),
    "valid": ("F15", "M01", "M10"),
    "test": ("F16", "M05", "M07"),
}

SAMPLE_RATE = 8000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
LABEL_SMOOTHING = 0.1
CNN_DROPOUT = 0.20

AUG_PROB = 0.5
TIME_STRETCH_RANGE = (0.8, 1.2)
PITCH_SHIFT_SEMITONES = 2.0
VOLUME_DB_RANGE = 3.0
TIME_MASK_WAV_PARAM = 0.15
FREQ_MASK_PARAM = 10
TIME_MASK_SPEC_PARAM = 20
N_FREQ_MASKS = 1
N_TIME_MASKS = 2
