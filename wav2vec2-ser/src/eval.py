import torch
from model import FrozenWav2Vec2, FineTunedWav2Vec2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FrozenWav2Vec2(num_classes=4)

checkpoint = torch.load(
    "best_model.pt",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

with torch.inference_mode():
    for batch in test_loader:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_values)
        predictions = logits.argmax(dim=-1)

torch.save(model.state_dict(), "best_model.pt")
