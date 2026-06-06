"""
Visual + text encoders built on top of openai/clip-vit-base-patch16.

Fine-tuning strategy (per SRS §3.2):
  - Freeze all ViT blocks except the last 4 transformer blocks + the final LayerNorm
  - Add a fresh 13-class classification head on top of the [CLS] embedding
  - Text encoder is frozen (used for zero-shot cross-modal matching only)
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from typing import Optional, Tuple

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
EMBEDDING_DIM   = 512
NUM_CATEGORIES  = 13


class FashionVisualEncoder(nn.Module):
    """
    CLIP ViT-B/16 visual encoder fine-tuned for fashion classification.

    Returns:
        embedding  : (B, 512)  — L2-normalized visual embedding
        logits     : (B, 13)   — raw class logits (before softmax)
    """

    def __init__(self, num_classes: int = NUM_CATEGORIES, freeze_early: bool = True):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection  # linear 768 → 512

        if freeze_early:
            self._freeze_early_layers()

        self.classifier = nn.Sequential(
            nn.LayerNorm(EMBEDDING_DIM),
            nn.Linear(EMBEDDING_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _freeze_early_layers(self):
        # Freeze embeddings + first 8 of 12 transformer blocks; unfreeze last 4
        for param in self.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in self.vision_model.pre_layrnorm.parameters():
            param.requires_grad = False

        encoder_layers = self.vision_model.encoder.layers
        num_layers = len(encoder_layers)  # 12 for ViT-B/16
        freeze_until = num_layers - 4     # freeze first 8

        for i, layer in enumerate(encoder_layers):
            requires_grad = i >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # pooled_output is the [CLS] token representation (B, 768)
        pooled = vision_outputs.pooler_output
        embedding = self.visual_projection(pooled)          # (B, 512)
        embedding = nn.functional.normalize(embedding, dim=-1)
        logits = self.classifier(embedding)                 # (B, 13)
        return embedding, logits


class FashionTextEncoder(nn.Module):
    """
    CLIP text encoder for encoding occasion/context strings.
    Fully frozen — used for zero-shot cross-modal matching only.

    Returns:
        embedding : (B, 512) — L2-normalized text embedding
    """

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection

        for param in self.parameters():
            param.requires_grad = False

        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)

    @torch.no_grad()
    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        text_outputs = self.text_model(**inputs)
        pooled = text_outputs.pooler_output
        embedding = self.text_projection(pooled)
        return nn.functional.normalize(embedding, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_outputs.pooler_output
        embedding = self.text_projection(pooled)
        return nn.functional.normalize(embedding, dim=-1)
