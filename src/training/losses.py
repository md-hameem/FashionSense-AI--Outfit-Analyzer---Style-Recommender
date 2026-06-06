"""
Loss functions for FashionSense AI training.

Classification  : CrossEntropyLoss (with MixUp support via soft targets)
Compatibility   : Online TripletLoss with margin
Combined        : FashionLoss wraps both with a tunable weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Online triplet loss for compatibility training.

    margin=0.3 is empirically solid for fashion compatibility tasks
    (see Han et al. 2017 — Bidirectional LSTMs for Fashion Compatibility).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class MixUpCrossEntropy(nn.Module):
    """
    CrossEntropy that handles MixUp-blended labels.
    When lam is None, falls back to standard integer-label CE.
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        if labels_b is None or lam == 1.0:
            return F.cross_entropy(logits, labels_a)
        loss_a = F.cross_entropy(logits, labels_a, reduction="none")
        loss_b = F.cross_entropy(logits, labels_b, reduction="none")
        return (lam * loss_a + (1 - lam) * loss_b).mean()


class FashionLoss(nn.Module):
    """
    Combined classification + compatibility loss.

        total = cls_loss + alpha * triplet_loss

    alpha=0.5 by default (equal weighting); tune via config.
    """

    def __init__(self, triplet_margin: float = 0.3, alpha: float = 0.5):
        super().__init__()
        self.cls_loss     = MixUpCrossEntropy()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.alpha        = alpha

    def classification_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        return self.cls_loss(logits, labels_a, labels_b, lam)

    def compatibility_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.triplet_loss(anchor, positive, negative)
