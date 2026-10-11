import sys
import torch
import torchaudio
import transformers
import librosa
import sklearn
import numpy as np

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("Transformers:", transformers.__version__)
print("Librosa:", librosa.__version__)
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
