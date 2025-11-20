import os

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "dental_classifier.pth")

IMG_SIZE = 224
DEVICE = "cuda"