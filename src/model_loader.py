import torch
from torchvision import models
import torch.nn as nn

def load_model(weights_path: str, device: str = "cpu"):
    # Build architecture
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 4)

    # Force CPU loading always (fixes Streamlit CUDA deserialization errors)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    model.to(device)
    return model
