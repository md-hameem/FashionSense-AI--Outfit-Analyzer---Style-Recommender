"""
Grad-CAM for ViT-B/16.

Standard Grad-CAM doesn't directly apply to transformers because there are
no spatial conv feature maps. We adapt it to ViT by:
  1. Hooking the last attention layer's output (the key-value projection)
  2. Computing gradients of the predicted class score w.r.t. those activations
  3. Global-averaging the gradient weights → per-patch importance
  4. Reshaping the 14×14 patch grid back to 224×224 via bilinear upsampling

Reference: "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-Based Localization" — Selvaraju et al., ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class ViTGradCAM:
    """
    Computes Grad-CAM saliency maps for a FashionVisualEncoder.

    Usage:
        cam = ViTGradCAM(model.visual_encoder)
        heatmap = cam.generate(pixel_values, target_class=3)
        overlay  = cam.overlay(original_pil_image, heatmap, alpha=0.4)
    """

    def __init__(self, visual_encoder: nn.Module):
        self.encoder = visual_encoder
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Target: the last transformer block's layer norm output (before attention)
        # For HuggingFace CLIPVisionModel, the encoder layers are at:
        #   vision_model.encoder.layers[-1]
        target_layer = self.encoder.vision_model.encoder.layers[-1]

        def forward_hook(module, input, output):
            # output is the layer output tensor (B, seq_len, hidden_dim)
            self._activations = output[0].detach()  # take first element if tuple

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self,
        pixel_values: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Args:
            pixel_values : (1, 3, 224, 224) — single image on the same device as the encoder
            target_class : if None, uses the predicted class

        Returns:
            heatmap : (224, 224) float32 numpy array in [0, 1]
        """
        self.encoder.eval()
        pixel_values = pixel_values.requires_grad_(True)

        embedding, logits = self.encoder(pixel_values)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.encoder.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # activations / gradients: (1, seq_len, hidden_dim)
        # seq_len = 1 (CLS) + 14*14 (patches) = 197
        activations = self._activations  # (1, 197, 768)
        gradients   = self._gradients    # (1, 197, 768)

        # Drop the CLS token (index 0); keep the 196 patch tokens
        patch_acts  = activations[:, 1:, :]   # (1, 196, 768)
        patch_grads = gradients[:, 1:, :]

        # Global average pool over hidden_dim → weight per patch
        weights = patch_grads.mean(dim=-1, keepdim=True)  # (1, 196, 1)
        cam = (weights * patch_acts).sum(dim=-1)           # (1, 196)
        cam = F.relu(cam)                                  # keep positive influence

        # Reshape to spatial grid 14×14 then upsample to 224×224
        cam = cam.reshape(1, 1, 14, 14)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Overlays the heatmap on the original image at `alpha` opacity.
        Returns a PIL Image suitable for saving or sending to the frontend.
        """
        import matplotlib.cm as cm

        cmap = cm.get_cmap(colormap)
        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(colored).resize(original_image.size, Image.BILINEAR)

        base = original_image.convert("RGB")
        blended = Image.blend(base, heatmap_img, alpha=alpha)
        return blended
