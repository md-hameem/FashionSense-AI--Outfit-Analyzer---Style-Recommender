"""
FashionSenseModel — top-level model combining the visual encoder and
compatibility module for joint training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import FashionVisualEncoder, FashionTextEncoder
from .compatibility import CompatibilityModel

NUM_CATEGORIES = 13


class FashionSenseModel(nn.Module):
    """
    Unified model for the FashionSense AI inference server.

    Classification mode  : encode image → (embedding, logits)
    Compatibility mode   : encode two images → compatibility score [0, 100]
    Cross-modal mode     : cosine similarity between image and text embeddings
    """

    def __init__(self, num_classes: int = NUM_CATEGORIES):
        super().__init__()
        self.visual_encoder = FashionVisualEncoder(num_classes=num_classes)
        self.text_encoder   = FashionTextEncoder()
        self.compatibility  = CompatibilityModel()

    # ── Inference helpers ────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor):
        """Returns (embedding, logits). Call during classification inference."""
        return self.visual_encoder(pixel_values)

    @torch.no_grad()
    def classify(self, pixel_values: torch.Tensor) -> Dict:
        embedding, logits = self.encode_image(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        top_idx  = probs.argmax(dim=-1)
        return {
            "embedding": embedding,
            "logits":    logits,
            "probs":     probs,
            "category_idx": top_idx,
        }

    @torch.no_grad()
    def get_compatibility_score(
        self,
        pixel_values_a: torch.Tensor,
        pixel_values_b: torch.Tensor,
    ) -> torch.Tensor:
        emb_a, _ = self.visual_encoder(pixel_values_a)
        emb_b, _ = self.visual_encoder(pixel_values_b)
        return self.compatibility.score(emb_a, emb_b)

    @torch.no_grad()
    def cross_modal_similarity(
        self,
        pixel_values: torch.Tensor,
        texts: list,
        device: torch.device,
    ) -> torch.Tensor:
        """Cosine similarity between image embedding and text embeddings."""
        img_emb, _ = self.visual_encoder(pixel_values)
        txt_emb = self.text_encoder.encode_text(texts, device)
        return (img_emb @ txt_emb.T)  # (B_img, B_text)

    # ── Training forward passes ──────────────────────────────────────────────

    def forward_classification(self, pixel_values: torch.Tensor):
        """Used in the classification training loop."""
        return self.visual_encoder(pixel_values)

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ):
        """Used in the compatibility training loop."""
        emb_a, _ = self.visual_encoder(anchor)
        emb_p, _ = self.visual_encoder(positive)
        emb_n, _ = self.visual_encoder(negative)
        return self.compatibility(emb_a, emb_p, emb_n)
