"""
Trainer for FashionSense AI.

Handles two-phase training:
  Phase 1 — Classification on DeepFashion2 (+ Fashion-MNIST warm-up)
  Phase 2 — Compatibility on Polyvore (encoder weights partially frozen)

Checkpoints are saved to /kaggle/working/ when running on Kaggle.
"""

import os
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from ..models.fashionsense import FashionSenseModel
from ..data.transforms import MixUpCollator
from .losses import FashionLoss

log = logging.getLogger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for FashionSenseModel.

    Config keys (all optional, sensible defaults provided):
        lr             : float  = 3e-5   (head LR; backbone gets LLRD-decayed fraction)
        lr_decay       : float  = 0.75   (LLRD decay rate per ViT block depth)
        weight_decay   : float  = 0.01
        epochs_cls     : int    = 30     (classification phase — more epochs for 95%+)
        epochs_compat  : int    = 10     (compatibility phase)
        warmup_epochs  : int    = 3      (linear warmup before cosine decay)
        batch_size     : int    = 64
        grad_clip      : float  = 1.0
        use_mixup      : bool   = True
        checkpoint_dir : str    = "/kaggle/working/checkpoints"
        log_every      : int    = 100    (steps between console logs)
    """

    DEFAULT_CONFIG = {
        "lr":             3e-5,
        "lr_decay":       0.75,
        "weight_decay":   0.01,
        "epochs_cls":     30,
        "epochs_compat":  10,
        "warmup_epochs":  3,
        "batch_size":     64,
        "grad_clip":      1.0,
        "use_mixup":      True,
        "checkpoint_dir": "/kaggle/working/checkpoints",
        "log_every":      100,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Training on {self.device}")

        self.model = FashionSenseModel().to(self.device)
        self.loss_fn = FashionLoss()
        self.scaler = GradScaler()  # AMP for faster GPU training

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Classification ───────────────────────────────────────────────

    def _build_warmup_cosine_scheduler(
        self, optimizer, num_epochs: int, warmup_epochs: int
    ) -> LambdaLR:
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)

    def train_classification(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        log.info("=== Phase 1: Classification Training ===")

        # LLRD optimizer: each ViT layer gets a decayed LR from deep → shallow
        visual_enc = self.model.visual_encoder
        if hasattr(visual_enc, "get_llrd_param_groups"):
            param_groups = visual_enc.get_llrd_param_groups(
                base_lr=self.cfg["lr"],
                decay_rate=self.cfg.get("lr_decay", 0.75),
            )
            log.info(
                f"LLRD: {len(param_groups)} param groups | "
                f"head LR={self.cfg['lr']:.1e} | "
                f"embed LR={param_groups[-1]['lr']:.2e}"
            )
        else:
            param_groups = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        optimizer = AdamW(param_groups, weight_decay=self.cfg["weight_decay"])
        scheduler = self._build_warmup_cosine_scheduler(
            optimizer,
            num_epochs=self.cfg["epochs_cls"],
            warmup_epochs=self.cfg.get("warmup_epochs", 3),
        )

        best_val_acc = 0.0
        for epoch in range(1, self.cfg["epochs_cls"] + 1):
            train_loss, train_acc = self._cls_epoch(train_loader, optimizer, epoch)
            val_loss, val_acc     = self._cls_eval(val_loader)
            scheduler.step()

            log.info(
                f"Epoch {epoch}/{self.cfg['epochs_cls']} | "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint("best_classifier.pt")
                log.info(f"  → New best val acc: {val_acc:.4f} (saved)")

        self._save_checkpoint("final_classifier.pt")
        log.info(f"Classification training complete. Best val acc: {best_val_acc:.4f}")

    def _cls_epoch(self, loader: DataLoader, optimizer, epoch: int):
        self.model.train()
        total_loss = correct = total = 0

        use_mixup = self.cfg["use_mixup"] and isinstance(loader.collate_fn, MixUpCollator)

        for step, batch in enumerate(loader, 1):
            if use_mixup:
                images, labels_a, labels_b, lam = [x.to(self.device) if hasattr(x, 'to') else x for x in batch]
            else:
                images, labels_a = batch[0].to(self.device), batch[1].to(self.device)
                labels_b, lam = None, 1.0

            optimizer.zero_grad()
            with autocast():
                _, logits = self.model.forward_classification(images)
                loss = self.loss_fn.classification_loss(logits, labels_a, labels_b, lam)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels_a).sum().item()
            total += labels_a.size(0)

            if step % self.cfg["log_every"] == 0:
                log.info(f"  Step {step}/{len(loader)} loss={loss.item():.4f}")

        return total_loss / len(loader), correct / total

    @torch.no_grad()
    def _cls_eval(self, loader: DataLoader):
        self.model.eval()
        total_loss = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast():
                _, logits = self.model.forward_classification(images)
                loss = self.loss_fn.classification_loss(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    # ── Phase 2: Compatibility ────────────────────────────────────────────────

    def train_compatibility(self, train_loader: DataLoader, val_loader: DataLoader):
        log.info("=== Phase 2: Compatibility Training ===")

        # Freeze the visual encoder backbone; only train the compatibility head
        for param in self.model.visual_encoder.vision_model.parameters():
            param.requires_grad = False

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg["lr"] * 0.5,
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg["epochs_compat"])

        best_val_loss = float("inf")
        for epoch in range(1, self.cfg["epochs_compat"] + 1):
            train_loss = self._compat_epoch(train_loader, optimizer, epoch)
            val_loss   = self._compat_eval(val_loader)
            scheduler.step()

            log.info(
                f"Epoch {epoch}/{self.cfg['epochs_compat']} | "
                f"train triplet={train_loss:.4f} | val triplet={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best_compatibility.pt")
                log.info(f"  → New best val triplet loss: {val_loss:.4f} (saved)")

        self._save_checkpoint("final_compatibility.pt")
        log.info("Compatibility training complete.")

    def _compat_epoch(self, loader: DataLoader, optimizer, epoch: int):
        self.model.train()
        total_loss = 0
        for step, (anchor, positive, negative) in enumerate(loader, 1):
            anchor   = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            optimizer.zero_grad()
            with autocast():
                proj_a, proj_p, proj_n = self.model.forward_triplet(anchor, positive, negative)
                loss = self.loss_fn.compatibility_loss(proj_a, proj_p, proj_n)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            if step % self.cfg["log_every"] == 0:
                log.info(f"  Step {step}/{len(loader)} triplet_loss={loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _compat_eval(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0
        for anchor, positive, negative in loader:
            anchor   = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            with autocast():
                proj_a, proj_p, proj_n = self.model.forward_triplet(anchor, positive, negative)
                loss = self.loss_fn.compatibility_loss(proj_a, proj_p, proj_n)
            total_loss += loss.item()
        return total_loss / len(loader)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str):
        path = Path(self.cfg["checkpoint_dir"]) / filename
        torch.save({
            "model_state_dict":      self.model.state_dict(),
            "visual_encoder_state":  self.model.visual_encoder.state_dict(),
            "compatibility_state":   self.model.compatibility.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        log.info(f"Loaded checkpoint from {path}")
