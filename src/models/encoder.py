"""
Visual + text encoders built on top of openai/clip-vit-base-patch16.

Fine-tuning strategy:
  - All 12 ViT transformer blocks trainable with layer-wise LR decay (LLRD)
  - Deeper blocks get higher LR; embedding layers get the smallest LR
  - Add a fresh 13-class classification head on top of the [CLS] embedding
  - Text encoder is frozen (used for zero-shot cross-modal matching only)
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from typing import Optional, Tuple, List, Dict

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
EMBEDDING_DIM   = 512
NUM_CATEGORIES  = 13


class FashionVisualEncoder(nn.Module):
    """
    CLIP ViT-B/16 visual encoder fine-tuned for fashion classification.

    All 12 transformer blocks are trainable. Use get_llrd_param_groups()
    to build an AdamW optimizer with layer-wise learning rate decay so
    early layers are updated conservatively and the head at full LR.

    Returns:
        embedding  : (B, 512)  — L2-normalized visual embedding
        logits     : (B, 13)   — raw class logits (before softmax)
    """

    def __init__(self, num_classes: int = NUM_CATEGORIES, freeze_early: bool = False):
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
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def _freeze_early_layers(self):
        # Legacy partial-freeze — kept for backward compatibility.
        # Prefer full fine-tuning via get_llrd_param_groups() instead.
        for param in self.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in self.vision_model.pre_layrnorm.parameters():
            param.requires_grad = False
        encoder_layers = self.vision_model.encoder.layers
        freeze_until = len(encoder_layers) - 4
        for i, layer in enumerate(encoder_layers):
            for param in layer.parameters():
                param.requires_grad = i >= freeze_until

    def get_llrd_param_groups(
        self, base_lr: float, decay_rate: float = 0.75
    ) -> List[Dict]:
        """
        Layer-wise learning rate decay (LLRD) for ViT-B/16.

        LR schedule (from head → embeddings):
          classifier head      : base_lr
          visual_projection    : base_lr * decay_rate
          post_layernorm       : base_lr * decay_rate^1
          transformer block 11 : base_lr * decay_rate^1   (last — deepest)
          transformer block 10 : base_lr * decay_rate^2
          ...
          transformer block 0  : base_lr * decay_rate^12  (first — shallowest)
          embeddings           : base_lr * decay_rate^13

        Args:
            base_lr    : LR for the classification head (highest)
            decay_rate : multiplicative decay per layer (0.75 recommended)
        """
        groups: List[Dict] = []

        # Head — full base LR
        groups.append({"params": list(self.classifier.parameters()), "lr": base_lr})

        # Visual projection — one step below head
        groups.append({
            "params": list(self.visual_projection.parameters()),
            "lr": base_lr * decay_rate,
        })

        # Post-LayerNorm
        groups.append({
            "params": list(self.vision_model.post_layernorm.parameters()),
            "lr": base_lr * (decay_rate ** 1),
        })

        # Transformer blocks — LLRD from deepest (block 11) to shallowest (block 0)
        encoder_layers = self.vision_model.encoder.layers
        num_layers = len(encoder_layers)  # 12
        for i in reversed(range(num_layers)):
            depth = num_layers - i          # 1 for block 11, 12 for block 0
            groups.append({
                "params": list(encoder_layers[i].parameters()),
                "lr": base_lr * (decay_rate ** depth),
            })

        # Embeddings + pre_layernorm — smallest LR
        groups.append({
            "params": (
                list(self.vision_model.embeddings.parameters()) +
                list(self.vision_model.pre_layrnorm.parameters())
            ),
            "lr": base_lr * (decay_rate ** (num_layers + 1)),
        })

        return groups

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
