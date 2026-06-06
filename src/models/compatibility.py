"""
Siamese compatibility scoring network trained on Polyvore outfit pairs.

Architecture: two shared-weight projection heads + a scoring MLP.
Trained with online triplet loss (anchor, positive, negative).
Output is a scalar in [0, 100] representing outfit compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CompatibilityHead(nn.Module):
    """Projects a 512-dim CLIP embedding to a 128-dim compatibility space."""

    def __init__(self, input_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CompatibilityModel(nn.Module):
    """
    Siamese network for outfit item compatibility scoring.

    During training: accepts triplets (anchor, positive, negative) and
    returns their projected embeddings for triplet loss computation.

    During inference: accepts two embeddings and returns a [0, 100] score.
    """

    def __init__(self, embedding_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.head = CompatibilityHead(embedding_dim, proj_dim)

        # Scoring MLP: takes concatenation of two proj embeddings → scalar
        self.scorer = nn.Sequential(
            nn.Linear(proj_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # output in (0, 1), scaled to 0-100 on read
        )

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)

    def score(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Returns compatibility score in [0, 100]."""
        proj_a = self.head(emb_a)
        proj_b = self.head(emb_b)
        combined = torch.cat([proj_a, proj_b], dim=-1)
        return self.scorer(combined).squeeze(-1) * 100.0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during training. Returns projected triplet embeddings."""
        return self.project(anchor), self.project(positive), self.project(negative)
