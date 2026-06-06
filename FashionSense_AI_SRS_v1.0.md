FASHIONSENSE AI

Outfit Analyzer & Style Recommender

Software Requirements Specification

Version 1.0  |  June 2026

1. Introduction

1.1 Project Overview

FashionSense AI is an intelligent, multi-modal fashion analysis web application that combines computer vision and natural language processing to deliver personalized outfit analysis and style recommendations. Users upload a photo of any clothing item or outfit, optionally provide contextual text (e.g., the occasion, season, or personal preference), and receive a comprehensive style evaluation including a compatibility score, occasion suitability rating, and curated outfit suggestions.

The system leverages a fine-tuned CLIP (Contrastive Language-Image Pretraining) / Vision Transformer (ViT) model trained on the DeepFashion2 and Fashion-MNIST datasets from Kaggle, combined with an LLM-based recommendation engine. The frontend is a React.js single-page application providing a polished, interactive user experience.

1.2 Purpose of this Document

This Software Requirements Specification (SRS) defines all functional and non-functional requirements, system architecture, data pipeline, model training strategy, API design, UI/UX specifications, and project milestones for FashionSense AI. It serves as the master reference document for development, evaluation, and future scaling.

1.3 Scope

Accepts clothing/outfit images (JPEG, PNG, WebP) and optional text descriptions as input

Classifies clothing items by category, color palette, pattern, and style archetype

Generates a style score (0–100) and occasion suitability breakdown

Returns 3–5 curated outfit suggestion cards with item descriptions

Displays Grad-CAM visual attention heatmaps for model explainability

Tracks user session history for comparison across uploads

Deployed as a React web app with a FastAPI backend and PyTorch inference server

1.4 Definitions & Abbreviations

1.5 Document Conventions

Requirements are labeled FR-XX (Functional) and NFR-XX (Non-Functional)

Priority levels: Critical > High > Medium > Low

All API endpoints follow REST conventions and return JSON

Model performance targets are defined as minimum acceptable thresholds

2. System Overview

2.1 Problem Statement

Fashion decision-making is a significant pain point for consumers globally. Studies show that people spend an average of 17 minutes per day deciding what to wear, and over 60% of online fashion purchases result in returns due to poor outfit compatibility or occasion mismatch. Existing fashion apps either rely on manual curation or require full outfit photos, failing users who want quick, intelligent feedback on individual items.

FashionSense AI solves this by providing instant AI-powered outfit analysis and personalized recommendations from a single clothing item photo, with an intuitive React interface requiring zero fashion expertise from the user.

2.2 System Architecture

2.3 Key Components

2.4 User Roles

3. Dataset & Machine Learning Pipeline

3.1 Datasets

3.1.1 DeepFashion2 (Primary — Kaggle)

491,895 high-resolution images across 13 clothing categories

Bounding boxes, landmarks, style labels, and cross-item matching pairs

Categories: short/long sleeve top, short/long sleeve outwear, vest, sling, shorts, trousers, skirt, short/long sleeve dress, sling dress, vest dress

Usage: primary classification and embedding training

3.1.2 Fashion-MNIST (Supplementary — Kaggle)

70,000 grayscale 28×28 images in 10 classes

Usage: rapid prototyping, baseline benchmarking, transfer learning warm-up

3.1.3 Polyvore Outfits Dataset (Compatibility Training)

21,799 outfit sets with item images and compatibility scores

Usage: outfit compatibility scoring sub-module training

3.1.4 Custom Augmented Dataset

Synthetic augmentation pipeline: random crop, horizontal flip, color jitter, MixUp, CutOut

Target: 600K training samples after augmentation

3.2 Model Architecture

3.2.1 Visual Encoder: ViT-B/16 Fine-tuned

Base: openai/clip-vit-base-patch16 (pretrained on 400M image-text pairs)

Fine-tuned layers: final 4 transformer blocks + classification head

Input resolution: 224×224 RGB

Output: 512-dim embedding + 13-class softmax for category classification

3.2.2 Text Encoder: CLIP Text Branch

Encodes occasion context text to 512-dim embedding

Cosine similarity with visual embedding for cross-modal matching

Enables zero-shot generalization to new occasion descriptions

3.2.3 Compatibility Scoring Module

Siamese network on Polyvore outfit pairs

Triplet loss training: anchor (item), positive (compatible), negative (incompatible)

Output: 0–100 compatibility score

3.2.4 Grad-CAM Explainability

Applied to the final ViT attention layer

Generates per-pixel saliency map highlighting decision-influencing regions

Overlaid on original image with 40% opacity in frontend

3.3 Training Pipeline

3.4 Model Evaluation Metrics

Top-1 Category Classification Accuracy (target: ≥ 88%)

AUROC per category (target: ≥ 0.91 mean)

Outfit Compatibility AUC (target: ≥ 0.82)

Inference latency on CPU (target: ≤ 800ms per request)

Inference latency on GPU (target: ≤ 150ms per request)

Grad-CAM sanity score (manual review on 500-sample test set)

4. Functional Requirements

4.1 Core Feature Requirements

4.2 API Functional Requirements

5. Non-Functional Requirements

5.1 Performance

5.2 Reliability & Availability

5.3 Security

5.4 Scalability

5.5 Usability

6. System Interfaces & API Design

6.1 Primary API: POST /api/v1/analyze

Request

Response (200 OK)

6.2 Auth Endpoints

6.3 History & Feedback Endpoints

6.4 Frontend-Backend Interface

Frontend communicates exclusively via the REST API (no direct DB access)

Axios instance with JWT interceptor auto-attaches Authorization header

Image compression (max 2MB, 1200px) applied client-side before upload using browser-image-compression

Heatmap displayed as <img> tag with S3 pre-signed URL (5-min expiry)

7. UI/UX Specification

7.1 Design Language

7.2 Page Specifications

7.2.1 Landing Page (/)

Hero section: large headline, sub-copy, CTA button 'Analyze My Outfit'

Demo section: 3 before/after example analysis cards (static, no API)

Feature strip: 4 icons — Instant Analysis, Style Scoring, Outfit Suggestions, Heatmap

Trusted by section: dataset logos (DeepFashion2, Kaggle, Polyvore)

7.2.2 Analyzer Page (/analyze)

Left panel (40%): dropzone with drag-and-drop, URL input tab, context text field, Analyze button

Right panel (60%): placeholder → animated loading spinner → results

Results: category badge, radar chart for 5 style sub-scores, occasion suitability bar chart

Below results: color palette row, pattern chip, Grad-CAM toggle

Outfit suggestion cards: horizontal scroll, each card with title, description, item chips

7.2.3 History Page (/history) — Authenticated

Grid of past analysis cards with thumbnail, date, category, overall score

Click to expand full result; delete button per card

Filter bar: date range, category, score range

7.2.4 Comparison Page (/compare)

Two upload panels side by side

Run comparison → displays both results with diff highlights

'Which is better for X?' recommendation using LLM

7.2.5 Admin Dashboard (/admin)

KPI cards: total analyses today/week/month, avg latency, model accuracy

Line chart: request volume over time (last 30 days)

Bar chart: category distribution of uploaded items

Table: recent feedback ratings with comments

7.3 Key UX Interactions

User lands on /analyze → sees drag-and-drop zone prominently centered

User drags an image → zone highlights with violet border + preview appears

User types context 'casual Friday' in text field → character counter shows

User clicks 'Analyze' → button shows loading spinner → skeleton loaders appear in result panel

Results stream in: category badge appears first → score animates counting up → suggestions fade in

User clicks 'Show Heatmap' → heatmap overlay fades in over original image with slider for opacity

User clicks suggestion card → expands to full-width modal with detailed item descriptions

User clicks 'Save' → prompted to log in if not authenticated → redirected back after auth

8. Data Model

8.1 Database Schema (PostgreSQL)

users table

analysis_sessions table

feedback table

8.2 S3 Object Structure

9. Technology Stack

10. Project Plan & Milestones

10.1 Development Phases

10.2 Milestones

10.3 Risk Register

11. Testing Strategy

11.1 Model Testing

Hold-out test set: 10% of DeepFashion2 (≈49K images), never seen during training

Metrics: Top-1 accuracy, Top-5 accuracy, per-class AUROC, confusion matrix

Grad-CAM sanity check: manual review of 500 randomly sampled heatmaps by 3 reviewers

Compatibility AUC on Polyvore held-out outfit pairs

11.2 Backend Testing

Unit tests: all service functions (image processing, cache, auth utilities) — target 85% coverage

Integration tests: /analyze endpoint with real model (5 fixture images across 5 categories)

Load test: Locust simulating 50 concurrent users, 200 requests/min for 5 minutes

Security test: OWASP Top 10 checklist, input fuzzing on file upload endpoint

11.3 Frontend Testing

Component tests: Vitest + React Testing Library for all major components

E2E tests: Playwright — analyze flow, auth flow, history flow

Accessibility: axe-core automated scan, manual keyboard navigation review

Visual regression: Percy (free tier) for UI snapshot comparison

11.4 Acceptance Criteria

User uploads valid image → analysis result displayed within 5 seconds (CPU) / 2 seconds (GPU)

Category classification correct on 88%+ of manual test set (50 images, human verified)

Outfit suggestions rated 4+ stars by 70%+ of beta testers (n=10 users)

All 17 functional requirements pass automated test suite

Lighthouse Performance score ≥ 85 on mobile and desktop

12. Future Enhancements

13. Appendix

13.1 Kaggle Dataset Links

DeepFashion2: https://www.kaggle.com/datasets/lyy1994/deepfashion2

Fashion-MNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist

Polyvore Outfits: https://www.kaggle.com/datasets/dnepozvannyi/polyvore-outfits

13.2 Key References

Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.

Dosovitskiy et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). ICLR 2021.

Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV 2017.

Ge et al. (2019). DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images. CVPR 2019.

Han et al. (2017). Learning Fashion Compatibility with Bidirectional LSTMs. ACM Multimedia 2017.

13.3 Development Environment Setup

13.4 Environment Variables

— End of Document —

FashionSense AI  |  SRS v1.0  |  June 2026