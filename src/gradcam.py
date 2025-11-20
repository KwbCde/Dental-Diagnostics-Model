import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Capture activations
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # Ensure its tensor
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self.gradients = grad.detach()

    def generate(self, image_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(image_tensor)
        target = output[0, class_idx]
        target.backward()

        # GAP over gradients
        weights = torch.mean(self.gradients, dim=(2, 3))  # (C)

        # Weighted sum
        cam = torch.zeros(self.activations.shape[2:], device=image_tensor.device)
        for c, w in enumerate(weights[0]):
            cam += w * self.activations[0, c]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy()


def overlay_heatmap(image_tensor, cam, alpha=0.4):
    img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = img * (1 - alpha) + heatmap * alpha
    overlay = np.clip(overlay, 0, 1)

    return img, heatmap, overlay
