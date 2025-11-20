import torch
import torch.nn.functional as F

from .model_loader import load_model
from .utils import load_image_tensor, CLASS_NAMES
from .gradcam import GradCAM
from .config import MODEL_PATH, DEVICE


class InferenceEngine:

    def __init__(self, weights_path, device="cuda"):
        self.device = device
        self.model = load_model(weights_path, device)
        self.model.eval()

        # Preload GradCAM object
        target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, target_layer)

    def predict(self, image_path):

        img_tensor = load_image_tensor(image_path, self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)[0]

        class_id = int(torch.argmax(probs).item())
        class_name = CLASS_NAMES[class_id]
        confidence = float(probs[class_id].item())

        return class_id, class_name, confidence, probs.cpu().numpy().tolist()

    def predict_with_gradcam(self, image_path):

        img_tensor = load_image_tensor(image_path, self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)[0]

        class_id = int(torch.argmax(probs).item())

        # Generate CAM
        cam = self.gradcam.generate(img_tensor, class_id)

        return {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(probs[class_id].item()),
            "probs": probs.cpu().numpy().tolist(),
            "cam": cam,
            "image_tensor": img_tensor.cpu()
        }
engine = InferenceEngine(MODEL_PATH, DEVICE)
