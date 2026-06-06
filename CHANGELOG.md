# Changelog

All notable changes to FashionSense AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- FastAPI inference server (`backend/`)
- React SPA frontend (`frontend/`)
- Docker Compose for local full-stack development
- CI/CD pipeline (GitHub Actions)
- LLM-based outfit suggestion narrative generation
- UMAP embedding visualization (alternative to t-SNE)

---

## [0.1.0] — 2026-06-06

### Added
- **SRS v1.0** (`FashionSense_AI_SRS_v1.0.md`) — full software requirements specification covering ML pipeline, API design, UI/UX, DB schema, testing strategy, and milestones
- **Data pipeline** (`src/data/`)
  - `DeepFashion2Dataset` — loads 491K images + JSON annotations, maps 13 category IDs
  - `FashionMNISTDataset` — CSV loader with 28×28 → 224×224 bicubic upsample
  - `PolyvoreDataset` — online triplet sampler for outfit compatibility training
  - `CutOut` augmentation transform
  - `MixUpCollator` — batch-level MixUp with Beta(α, α) mixing ratio
  - CLIP-exact normalization constants (`CLIP_MEAN`, `CLIP_STD`)
  - Separate train/eval transform pipelines
- **Model architecture** (`src/models/`)
  - `FashionVisualEncoder` — ViT-B/16 from `openai/clip-vit-base-patch16`; freezes first 8 of 12 transformer blocks; adds 512→256→13 classification head
  - `FashionTextEncoder` — frozen CLIP text branch for zero-shot cross-modal matching
  - `CompatibilityModel` — Siamese network with 512→128 projection heads and 0–100 scoring MLP
  - `FashionSenseModel` — unified model combining encoder + compatibility module
- **Training pipeline** (`src/training/`)
  - `MixUpCrossEntropy` — supports soft MixUp targets and standard integer labels
  - `TripletLoss` — online triplet loss with configurable margin (default 0.3)
  - `FashionLoss` — combined classification + compatibility loss with α weighting
  - `Trainer` — two-phase training (classification then compatibility), AMP (fp16), cosine LR schedule, gradient clipping, best-checkpoint saving
- **Explainability** (`src/utils/gradcam.py`)
  - `ViTGradCAM` — hooks final ViT attention layer, computes patch-level saliency, upsamples 14×14 → 224×224, overlays on original image
- **Kaggle training notebook** (`notebooks/fashionsense_kaggle_training.ipynb`)
  - 9 sections: EDA → architecture diagrams → dataloaders → Phase 1 training → Phase 1 evaluation → Phase 2 training → Grad-CAM gallery → final dashboard
  - 22 saved visualization outputs (class distributions, sample grids, augmentation pipeline, architecture diagram, parameter breakdown, patch tokenization, LR schedule, training curves, confusion matrices, ROC curves, t-SNE, confidence analysis, worst predictions, compatibility pairs, Grad-CAM gallery, final dark-theme dashboard)
- **Repository scaffolding**
  - `requirements.txt` with full dependency set (ML, backend, auth, DB, cache, testing)
  - `.gitignore` covering Python, ML checkpoints, datasets, secrets, IDE files
  - `.env.example` with all required environment variables
  - `CONTRIBUTING.md` with branch strategy and PR workflow
  - `LICENSE` (MIT)

[Unreleased]: https://github.com/your-username/fashionsense-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/fashionsense-ai/releases/tag/v0.1.0
