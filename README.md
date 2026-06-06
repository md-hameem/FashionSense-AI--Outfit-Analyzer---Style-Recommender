# FashionSense AI — Outfit Analyzer & Style Recommender

> AI-powered multi-modal fashion analysis. Upload a clothing photo, get instant style scoring, outfit compatibility ratings, curated suggestions, and Grad-CAM visual explanations.

---

## Overview

FashionSense AI combines a fine-tuned **CLIP ViT-B/16** visual encoder with a **Siamese compatibility network** to deliver personalized outfit analysis from a single image. Users optionally provide context text (occasion, season, preference) for cross-modal zero-shot matching.

**Core outputs per analysis:**
- Category classification (13 DeepFashion2 classes)
- Style score 0–100 + 5-axis radar chart
- Occasion suitability breakdown
- Color palette + pattern chip
- 3–5 curated outfit suggestion cards
- Grad-CAM heatmap overlay (explainability)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React SPA (Frontend)                  │
│    /analyze  /history  /compare  /admin                  │
└────────────────────┬────────────────────────────────────┘
                     │ REST/JSON + JWT
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                         │
│   POST /api/v1/analyze   GET /api/v1/history            │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐  ┌───────────▼───────────────────┐
│  PyTorch Inference  │  │  PostgreSQL + Redis Cache      │
│  ViT-B/16 + Siamese │  │  S3 (images + heatmaps)       │
└─────────────────────┘  └───────────────────────────────┘
```

**ML Stack:**

| Component | Detail |
|---|---|
| Visual encoder | `openai/clip-vit-base-patch16` — last 4 of 12 transformer blocks fine-tuned |
| Text encoder | CLIP text branch (frozen) — 512-dim cosine similarity |
| Compatibility | Siamese network, triplet loss, margin=0.3 |
| Explainability | Grad-CAM on final ViT attention layer → 14×14 → 224×224 |

---

## Datasets

| Dataset | Role | Size |
|---|---|---|
| [DeepFashion2](https://www.kaggle.com/datasets/lyy1994/deepfashion2) | Primary classification training | 491,895 images, 13 categories |
| [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | Warm-up / baseline benchmarking | 70,000 images, 10 classes |
| [Polyvore Outfits](https://www.kaggle.com/datasets/dnepozvannyi/polyvore-outfits) | Compatibility module training | 21,799 outfit sets |

---

## Performance Targets

| Metric | Target |
|---|---|
| Top-1 category accuracy | ≥ 88% |
| Mean AUROC (one-vs-rest) | ≥ 0.91 |
| Compatibility AUC | ≥ 0.82 |
| CPU inference latency | ≤ 800ms |
| GPU inference latency | ≤ 150ms |
| Lighthouse score | ≥ 85 (mobile + desktop) |

---

## Project Structure

```
fashionsense-ai/
├── notebooks/
│   └── fashionsense_kaggle_training.ipynb   # Full training notebook (upload to Kaggle)
├── src/
│   ├── data/
│   │   ├── datasets.py       # DeepFashion2Dataset, FashionMNISTDataset, PolyvoreDataset
│   │   └── transforms.py     # CLIP-compatible augmentation (CutOut, MixUp, ColorJitter)
│   ├── models/
│   │   ├── encoder.py        # FashionVisualEncoder + FashionTextEncoder
│   │   ├── compatibility.py  # Siamese CompatibilityModel
│   │   └── fashionsense.py   # FashionSenseModel (combined, used by inference server)
│   ├── training/
│   │   ├── losses.py         # MixUpCrossEntropy + TripletLoss
│   │   └── trainer.py        # Two-phase Trainer with AMP + checkpointing
│   └── utils/
│       └── gradcam.py        # ViTGradCAM — patch-level saliency maps
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── FashionSense_AI_SRS_v1.0.md
```

---

## Quick Start — Training on Kaggle

1. **Create a new Kaggle Notebook** at [kaggle.com/code](https://www.kaggle.com/code)

2. **Add datasets** (Settings → Add data):
   - `lyy1994/deepfashion2`
   - `zalando-research/fashionmnist`
   - `dnepozvannyi/polyvore-outfits`

3. **Enable GPU** (Settings → Accelerator → GPU T4 x2 or P100)

4. **Upload** `notebooks/fashionsense_kaggle_training.ipynb` via File → Import Notebook

5. **Run All** — training runs in two phases:
   - Phase 1 (15 epochs): Classification on DeepFashion2
   - Phase 2 (10 epochs): Compatibility on Polyvore

6. **Download outputs** from `/kaggle/working/`:
   - `checkpoints/best_classifier.pt`
   - `checkpoints/best_compatibility.pt`
   - `checkpoints/final_model.pt`
   - `checkpoints/model_config.json`
   - `visualizations/` — 12 diagnostic plots

---

## Local Development Setup

```bash
# Clone
git clone https://github.com/your-username/fashionsense-ai.git
cd fashionsense-ai

# Create virtualenv
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and fill environment variables
cp .env.example .env
```

---

## Notebook Visualizations

The Kaggle notebook produces 12 saved plots in `/kaggle/working/visualizations/`:

| File | Contents |
|---|---|
| `df2_class_distribution.png` | Category bar charts for train + val splits |
| `df2_category_samples.png` | One sample image per category |
| `df2_image_dimensions.png` | Width / height / scatter resolution analysis |
| `fmnist_overview.png` | Class distribution + 2×5 sample grid |
| `fmnist_pixel_distribution.png` | Per-class pixel intensity histograms |
| `dataset_summary_table.png` | Styled comparison table for all 3 datasets |
| `augmentation_pipeline.png` | Step-by-step transform preview |
| `model_architecture.png` | Dark-theme ViT layer diagram (frozen vs trainable) |
| `model_params.png` | Parameter breakdown pie + frozen/trainable bar |
| `vit_patch_tokenization.png` | 16×16 patch grid + per-patch brightness |
| `lr_schedule.png` | Cosine LR curve + MixUp Beta distribution |
| `training_dashboard.png` | 4-panel: loss, accuracy, LR, overfitting monitor |
| `confusion_matrix.png` | Raw counts + row-normalized side by side |
| `per_class_metrics.png` | Precision / Recall / F1 grouped bars |
| `roc_curves.png` | 13 ROC curves with AUC per class |
| `tsne_embeddings.png` | t-SNE of 3,000 val embeddings |
| `confidence_analysis.png` | Correct vs wrong confidence + per-class bars |
| `worst_predictions.png` | 8 highest-confidence error images |
| `compatibility_training.png` | Triplet loss curves + score distributions |
| `compatibility_pairs.png` | Anchor / compatible / incompatible pair gallery |
| `gradcam_gallery.png` | All 13 categories × top-3 CAM overlays |
| `final_dashboard.png` | Dark-theme summary: KPIs + all training curves |

---

## Key References

- Radford et al. (2021). [CLIP](https://arxiv.org/abs/2103.00020). ICML 2021.
- Dosovitskiy et al. (2021). [ViT](https://arxiv.org/abs/2010.11929). ICLR 2021.
- Selvaraju et al. (2017). [Grad-CAM](https://arxiv.org/abs/1610.02391). ICCV 2017.
- Ge et al. (2019). [DeepFashion2](https://arxiv.org/abs/1901.07973). CVPR 2019.
- Han et al. (2017). [Fashion Compatibility with Bidirectional LSTMs](https://arxiv.org/abs/1707.05691). ACM MM 2017.

---

## License

MIT — see [LICENSE](LICENSE).
