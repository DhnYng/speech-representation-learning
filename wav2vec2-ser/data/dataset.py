import os
import re
import glob

import torch
import torchaudio

from torch.utils.data import Dataset

from config import (
    DATA_PATH,
    EMOTIONS,
    SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
)


class IEMOCAPDataset(Dataset):

    def __init__(
        self,
        data_root=DATA_PATH,
        sessions=None,
        sample_rate=SAMPLE_RATE,
        max_audio_length=MAX_AUDIO_LENGTH
    ):

        self.data_root = data_root
        self.sessions = sessions or [1, 2, 3, 4, 5]
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length

        self.emotions = EMOTIONS
        self.samples = []
        self._load_metadata()


    def _load_metadata(self):

        for session in self.sessions:
            
            session_name = f"Session{session}"

            session_path = os.path.join(
                self.data_root,
                session_name
            )

            annotation_paths = glob.glob(
                os.path.join(
                    session_path,
                    "dialog",
                    "EmoEvaluation",
                    "*.txt"
                )
            )

            for annotation_path in sorted(set(annotation_paths)):
                self._parse_annotation(
                    annotation_path,
                    session_path,
                    session
                )

        print(f"Loaded samples: {len(self.samples)}")


    def _parse_annotation(
        self,
        annotation_path,
        session_path,
        session
    ):

        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:

            match = re.search(r"\]\s+(\S+)\s+\[([a-zA-Z]+)\]", line)

            if match is None:
                continue

            utterance_id = match.group(1)
            emotion = match.group(2).lower()

            if emotion not in self.emotions:
                continue

            dialogue_id = utterance_id.rsplit("_", 1)[0]

            audio_path = os.path.join(
                session_path,
                "sentences",
                "wav",
                dialogue_id,
                utterance_id + ".wav"
            )

            if not os.path.exists(audio_path):

                candidates = glob.glob(
                    os.path.join(
                        session_path,
                        "**",
                        utterance_id + ".wav"
                    ),
                    recursive=True
                )

                if not candidates:
                    continue

                audio_path = candidates[0]

            self.samples.append({
                "audio_path": audio_path,
                "utterance_id": utterance_id,
                "emotion": self.emotions[emotion],
                "speaker": utterance_id.split("_")[-1][0],
                "session": session
            })


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        waveform, sr = torchaudio.load(sample["audio_path"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )

            waveform = resampler(waveform)

        max_samples = int(self.sample_rate * self.max_audio_length)

        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        waveform = waveform.squeeze(0)

        label = sample["emotion"]

        return {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "utterance_id": sample["utterance_id"],
            "speaker": sample["speaker"],
            "session": sample["session"]
        }
