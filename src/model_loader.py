import torch
from torchvision import models
import torch.nn as nn
import os

# Loads the EfficientNet-B0 model with custom classifier head.

def load_model(weights_path: str, device: str = "cpu"):
   
    #Load base architecture
    model = models.efficientnet_b0(weights=None)
    
    # Replace classifier head to match your 4 classes
    model.classifier[1] = nn.Linear(1280, 4)
    
    # Load state dict
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    
    # Set evaluation mode
    model.eval()
    
    # Move to device
    model.to(device)
    
    return model
