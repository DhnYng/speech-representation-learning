import torch
import librosa
import soundfile as sf

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"

extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def preprocess_audio(path):
    # Load audio
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz
    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000,
        )

    # Normalize and convert to tensor
    inputs = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    return inputs
