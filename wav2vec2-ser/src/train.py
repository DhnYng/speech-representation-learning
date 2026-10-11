import torch
import torch.nn as nn
from model import FrozenWav2Vec2, FineTunedWav2Vec2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FrozenWav2Vec2(num_classes=4)
# model = FineTunedWav2Vec2(num_classes=4)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# requires_grad=True인 파라미터만 optimizer에 전달
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
)

model.train()

"""
for batch in train_loader:
    input_values = batch["input_values"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    logits = model(input_values)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
"""
