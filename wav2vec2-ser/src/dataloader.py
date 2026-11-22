import torch

from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from config import MODEL_NAME, BATCH_SIZE
from dataset import IEMOCAPDataset


feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


def collate_fn(batch):

    waveforms = [
        item["waveform"].numpy()
        for item in batch
    ]

    inputs = feature_extractor(
        waveforms,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    labels = torch.stack([
        item["label"]
        for item in batch
    ])

    return {

        "input_values": inputs["input_values"],

        "attention_mask": inputs["attention_mask"],

        "labels": labels,

        "utterance_ids": [
            item["utterance_id"]
            for item in batch
        ],

        "speakers": [
            item["speaker"]
            for item in batch
        ],

        "sessions": [
            item["session"]
            for item in batch
        ]

    }


def create_dataloader(
    sessions,
    batch_size=BATCH_SIZE,
    shuffle=False
):

    dataset = IEMOCAPDataset(
        sessions=sessions
    )

    dataloader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=collate_fn,

        num_workers=0,

        pin_memory=torch.cuda.is_available()

    )

    return dataloader
