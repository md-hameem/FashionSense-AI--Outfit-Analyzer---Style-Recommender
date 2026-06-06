"""
Augmentation pipeline for FashionSense AI training.

CLIP normalization stats are fixed — do not change them; they match the
pretrained openai/clip-vit-base-patch16 preprocessor exactly.
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# CLIP's exact normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

IMAGE_SIZE = 224


class CutOut:
    """Randomly masks out square patches to force global feature learning."""

    def __init__(self, num_holes: int = 1, max_h_size: int = 40, max_w_size: int = 40):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = torch.ones_like(img)
        for _ in range(self.num_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - self.max_h_size // 2)
            y2 = min(h, y + self.max_h_size // 2)
            x1 = max(0, x - self.max_w_size // 2)
            x2 = min(w, x + self.max_w_size // 2)
            mask[:, y1:y2, x1:x2] = 0
        return img * mask


class MixUpCollator:
    """
    Applies MixUp at the batch level (collate-time), not per-sample.
    alpha=0.2 is standard for image classification.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.dist = torch.distributions.Beta(alpha, alpha)

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        lam = self.dist.sample().item()
        idx = torch.randperm(images.size(0))

        mixed_images = lam * images + (1 - lam) * images[idx]
        return mixed_images, labels, labels[idx], lam


def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        CutOut(num_holes=1, max_h_size=32, max_w_size=32),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(IMAGE_SIZE + 32),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse CLIP normalization for visualization (e.g., Grad-CAM overlay)."""
    mean = torch.tensor(CLIP_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
