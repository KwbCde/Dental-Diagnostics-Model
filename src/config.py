import os
import torch

# Build ROOT_DIR relative to file location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#  absolute model path
MODEL_PATH = os.path.join(ROOT_DIR, "ml", "models", "dental_classifier.pth")

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
