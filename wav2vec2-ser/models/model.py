import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

def masked_mean_pooling(hidden_states, attention_mask):
    """
    hidden_states:  [B, T, D]
    attention_mask: [B, T] (유효 구간=1, padding=0)

    returns: [B, D]
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)

    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    return summed / counts

class FrozenWav2Vec2(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        num_classes=4,
    ):
        super.__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # freeze pretrained encoder
        self.encoder.requires_grad_(False)

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes,
        )

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        features = outputs.last_hidden_state

        if attention_mask is not None:
            # Original waveform mask -> CNN feature mask
            feature_mask = self.encoder._get_feature_vector_attention_mask(
                features.shape[1],
                attention_mask,
            )

            pooled = masked_mean_pooling(features, feature_mask)

        else:
            pooled = features.mean(dim=1)

        logits = self.classifier(pooled)

        return logits, hidden_states

class FineTunedWav2Vec2(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        num_classes=4,
    ):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # trainable encoder and classifier
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes,
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        features = outputs.last_hidden_state

        if attention_mask is not None:
            # Original waveform mask -> CNN feature mask
            feature_mask = self.encoder._get_feature_vector_attention_mask(
                features.shape[1],
                attention_mask,
            )

            pooled = masked_mean_pooling(features, feature_mask)

        else:
            pooled = features.mean(dim=1)

        logits = self.classifier(pooled)

        return logits, hidden_states
