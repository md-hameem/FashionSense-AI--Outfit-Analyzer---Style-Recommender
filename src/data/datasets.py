"""
PyTorch Dataset classes for all three training datasets.

Kaggle input paths (set automatically when datasets are added to a Kaggle notebook):
  DeepFashion2  → /kaggle/input/deepfashion2/
  Fashion-MNIST → /kaggle/input/fashionmnist/
  Polyvore      → /kaggle/input/polyvore-outfits/
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


# ── DeepFashion2 ─────────────────────────────────────────────────────────────

# 13 categories exactly as labeled in DeepFashion2
DF2_CATEGORIES = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]
DF2_LABEL_TO_IDX = {cat: i for i, cat in enumerate(DF2_CATEGORIES)}
# DeepFashion2 uses 1-based category IDs in annotations
DF2_ID_TO_LABEL = {i + 1: cat for i, cat in enumerate(DF2_CATEGORIES)}


class DeepFashion2Dataset(Dataset):
    """
    Loads images and category labels from DeepFashion2.

    Expected directory layout:
        root/
          train/image/  *.jpg
          train/annos/  *.json   (one JSON per image, same stem)
          validation/image/
          validation/annos/

    Each annotation JSON contains an 'item1' … 'itemN' dict with a
    'category_id' field (1-indexed, maps to DF2_ID_TO_LABEL).
    We take the *first* item's category as the image label.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "validation"), f"Unknown split: {split}"
        self.transform = transform

        image_dir = Path(root) / split / "image"
        anno_dir  = Path(root) / split / "annos"

        image_paths = sorted(image_dir.glob("*.jpg"))
        if max_samples:
            image_paths = image_paths[:max_samples]

        self.samples: List[Tuple[Path, int]] = []
        for img_path in image_paths:
            anno_path = anno_dir / img_path.with_suffix(".json").name
            if not anno_path.exists():
                continue
            with open(anno_path) as f:
                anno = json.load(f)
            # Find first item key (item1, item2, …)
            item_key = next((k for k in anno if k.startswith("item")), None)
            if item_key is None:
                continue
            cat_id = anno[item_key].get("category_id")
            label = DF2_ID_TO_LABEL.get(cat_id)
            if label is None:
                continue
            self.samples.append((img_path, DF2_LABEL_TO_IDX[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Fashion-MNIST ─────────────────────────────────────────────────────────────

FMNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNISTDataset(Dataset):
    """
    Loads Fashion-MNIST from the Kaggle CSV format.

    Expected files:
        root/fashion-mnist_train.csv
        root/fashion-mnist_test.csv

    Pixels are uint8 [0, 255] laid flat in 784 columns after 'label'.
    We upsample 28×28 → 224×224 with bicubic interpolation so CLIP's
    patch-based tokenizer has enough resolution.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        filename = "fashion-mnist_train.csv" if split == "train" else "fashion-mnist_test.csv"
        df = pd.read_csv(Path(root) / filename)
        self.labels = df["label"].values.astype(np.int64)
        self.pixels = df.drop(columns=["label"]).values.astype(np.uint8)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pixels = self.pixels[idx].reshape(28, 28)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        image = image.resize((224, 224), Image.BICUBIC)
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])


# ── Polyvore Outfits (Compatibility) ─────────────────────────────────────────

class PolyvoreDataset(Dataset):
    """
    Triplet dataset for compatibility training.

    Expected layout (from dnepozvannyi/polyvore-outfits on Kaggle):
        root/
          images/         <set_id>/<item_id>.jpg
          train_no_dup.json   list of {set_id, items: [item_id,...], scores: [...]}
          test_no_dup.json

    Each sample is a triplet (anchor, positive, negative):
      - anchor   : a random item from an outfit
      - positive : another item from the *same* outfit (compatible)
      - negative : an item from a *different* outfit (incompatible)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        json_file = "train_no_dup.json" if split == "train" else "test_no_dup.json"
        json_path = self.root / json_file
        with open(json_path) as f:
            outfits = json.load(f)

        # Build list of (set_id, [item_ids]) — filter outfits with ≥ 2 items
        self.outfits: List[Tuple[str, List[str]]] = []
        self.all_items: List[Tuple[str, str]] = []  # (set_id, item_id)

        for outfit in outfits:
            set_id = str(outfit["set_id"])
            items = [str(i) for i in outfit.get("items", [])]
            if len(items) >= 2:
                self.outfits.append((set_id, items))
            for item_id in items:
                self.all_items.append((set_id, item_id))

        if max_samples:
            self.outfits = self.outfits[:max_samples]

    def _load_image(self, set_id: str, item_id: str) -> Image.Image:
        path = self.root / "images" / set_id / f"{item_id}.jpg"
        return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.outfits)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        set_id, items = self.outfits[idx]

        # Anchor + positive from same outfit
        anc_id, pos_id = random.sample(items, 2)

        # Negative from a different outfit
        neg_set_id, neg_item_id = self.all_items[random.randint(0, len(self.all_items) - 1)]
        while neg_set_id == set_id:
            neg_set_id, neg_item_id = self.all_items[random.randint(0, len(self.all_items) - 1)]

        anchor   = self._load_image(set_id, anc_id)
        positive = self._load_image(set_id, pos_id)
        negative = self._load_image(neg_set_id, neg_item_id)

        if self.transform:
            anchor   = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
