import torch

# Dataset
DATA_PATH = "./data/IEMOCAP_full_release"

# Emotion Labels
EMOTIONS = {
    "ang": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "neu": 3
}

NUM_CLASSES = 4

# Audio
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 15.0

# Model
MODEL_NAME = "facebook/wav2vec2-base"

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 4
OPTIMIZER = "Adam"
LOSS_FUNCTION = "CrossEntropyLoss"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
