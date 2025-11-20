import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import datasets

dataset = datasets.ImageFolder("ml/data/train")
CLASS_NAMES = dataset.classes

def get_inference_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image_tensor(path, device, img_size=224):
    img = Image.open(path).convert("RGB")
    transform = get_inference_transform(img_size)
    tensor_img = transform(img).unsqueeze(0).to(device)
    return tensor_img


# Convert model output to prediction + softmax
def predict_from_logits(logits):
    """
    logits: model output (1, num_classes)
    Returns: predicted_class_index, probabilities (numpy array)
    """

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))

    return pred_idx, probs



# Pretty formatting for UI
def format_prediction(pred_idx, probs):
    class_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return class_name, confidence
