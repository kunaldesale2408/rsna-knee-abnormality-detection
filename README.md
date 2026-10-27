# RSNA Knee Abnormality Detection — State-of-the-Art Solution Framework

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%2B%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Metric](https://img.shields.io/badge/Metric-Macro%20ROC--AUC%20(12%20Labels)-brightgreen.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Status](https://img.shields.io/badge/Private%20%2F%20Public%20LB-0.957%2B%20%2F%200.945%2B-gold.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Hardware](https://img.shields.io/badge/Inference-Dual%20NVIDIA%20Tesla%20T4%20(Kaggle)-76b900.svg)](https://www.kaggle.com/)

An end-to-end, competition-grade deep learning solution for the **RSNA Knee Abnormality Detection** challenge hosted on Kaggle. This repository implements an end-to-end framework spanning DICOM volumetric geometry processing, multilingual zero-shot LLM clinical report supervision, 2.5D multi-plane attention vision transformers, multi-backbone ensembling (DINOv2, ConvNeXt, RadDINO, CoAtNet MIL), dual-GPU parallelized inference, and per-pathology rank-blending.

---

## Table of Contents
1. [Executive Summary & Architecture](#1-executive-summary--architecture)
2. [Competition & Clinical Problem Overview](#2-competition--clinical-problem-overview)
3. [Volumetric DICOM Engineering & Geometry](#3-volumetric-dicom-engineering--geometry)
4. [Supervision Engineering: Multilingual LLM Labeling](#4-supervision-engineering-multilingual-llm-labeling)
5. [Model Architectures & Training Strategies](#5-model-architectures--training-strategies)
6. [High-Performance Inference Engine (Apex SOTA)](#6-high-performance-inference-engine-apex-sota)
7. [Ensemble, Per-Finding Weights & Calibration](#7-ensemble-per-finding-weights--calibration)
8. [Leaderboard Progression & Milestones](#8-leaderboard-progression--milestones)
9. [Repository Structure & Code Map](#9-repository-structure--code-map)
10. [Reproduction & Execution Guide](#10-reproduction--execution-guide)
11. [Key Engineering Insights & Pitfalls Avoided](#11-key-engineering-insights--pitfalls-avoided)

---

## 1. Executive Summary & Architecture

The primary challenge of the RSNA Knee Abnormality Detection competition lies in learning to diagnose **12 distinct knee abnormalities** from multi-series MRI studies under an extreme weak-supervision constraint: **only 58 studies contain expert gold annotations**, while the remaining **4,349 training studies provide only unstructured, multilingual radiology reports** (written across 12 languages). Furthermore, code execution must run completely offline within a strict 9-hour runtime window on Kaggle GPU hardware.

```
                                  [Multi-Series Knee MRI DICOMs]
                         (Sagittal, Coronal, Axial / Variable Slices)
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │   Volumetric Preprocessing & Normalization│
                        │ ───────────────────────────────────────── │
                        │ • 3D ImageOrientationPatient Normal Vector│
                        │ • Spatial Slice Sorting: d = p · (u x v)  │
                        │ • MONOCHROME1 Inversion & Rescale Slope   │
                        │ • Left-Knee Anatomical Mirroring (x → W-x)│
                        │ • 2.5D Triplet Extraction: [z-2, z, z+2]  │
                        └─────────────────────┬─────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │   Branch A: Raptor Pipeline   │                 │   Branch B: Ortho/ConvNeXt    │
     │ ───────────────────────────── │                 │ ───────────────────────────── │
     │ • CoAtNet MIL (384 & 224 px)  │                 │ • ConvNeXt-Small / Base       │
     │ • Multi-Instance Spatial Pool │                 │ • OrthoFoundation Backbone   │
     │ • Fluid/Fat Suppression Priors│                 │ • 12 Target-Specific Queries  │
     │ • 4-Way Flip Test-Time Aug    │                 │ • 4-Fold SWA Checkpoints      │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     │          ┌───────────────────────────┐          │
                     └─────────►│ Dual-GPU Asynchronous I/O │◄─────────┘
                                │ Worker0 / Worker1 Pipeline│
                                │ torch.compile (inductor)  │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │   Per-Finding Rank Blend  │
                                │ ───────────────────────── │
                                │ • Synovitis:    0.80 / 0.2│
                                │ • PF OA:        0.60 / 0.4│
                                │ • Medial OA:    0.50 / 0.5│
                                │ • Baker's Cyst: 0.35 / 0.6│
                                │ • Lateral OA:   0.20 / 0.8│
                                │ • Uniform Rank Projection │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                   [Final submission.csv]
                                  (Macro ROC-AUC: 0.957+)
```

---

## 2. Competition & Clinical Problem Overview

### 2.1 The 12 Target Abnormalities
The competition requires predicting continuous probability targets $\in [0, 1]$ for 12 binary findings across each knee exam:

| Category | Finding Name | Clinical Description & Diagnostic Focus |
|:---|:---|:---|
| **Ligamentous** | **ACL** | Anterior Cruciate Ligament tear, avulsion, or mucoid degeneration |
| | **MCL** | Medial Collateral Ligament tear, partial rupture, or sprain |
| **Meniscal** | **Medial Meniscus** | Medial meniscal tear (horizontal, radial, complex, root tear) |
| | **Lateral Meniscus** | Lateral meniscal tear or maceration |
| **Chondral / OA** | **Medial OA** | Medial tibiofemoral joint osteoarthritis, cartilage thinning |
| | **Lateral OA** | Lateral tibiofemoral joint osteoarthritis |
| | **PF OA** | Patellofemoral osteoarthritis and chondromalacia patellae |
| **Fluid / Synovium** | **Effusion** | Joint capsule fluid distention / intra-articular effusion |
| | **Synovitis** | Synovial hypertrophy, thickening, and active inflammatory pannus |
| | **Baker's Cyst** | Popliteal synovial fluid distention in the gastrocnemio-semimembranosus bursa |
| **Osseous / Trauma** | **Contusion** | Subchondral bone marrow edema (BME) / micro-trabecular fractures |
| | **Fracture** | Cortical disruption, plateau fractures, or insufficiency fracture |

### 2.2 Evaluation Metric
Submissions are evaluated on the unweighted arithmetic mean of the area under the ROC curve (**Macro ROC-AUC**) computed across all 12 targets:

$$\text{Macro ROC-AUC} = \frac{1}{12} \sum_{c=1}^{12} \text{ROC-AUC}_c$$

### 2.3 The Core Modeling Conundrum
1. **The Thresholding Disconnect**: Clinical reports describe "trace effusion", "mild joint fluid", or "borderline medial signal". In the gold-labeled evaluation set (58 studies), radiologists applied strict severity thresholds where trace/borderline was marked **negative (0.0)**. A naive binary rule assigning $1.0$ to every mentioned abnormality catastrophically degrades ROC-AUC.
2. **Scanner Memorization Bias**: MRI acquisition protocols vary by hospital and scanner manufacturer. A naive random train/val split causes severe scanner fingerprint leakage, artificially inflating validation ROC-AUC by **$+0.053$**. Cross-validation must strictly employ `GroupKFold` grouped on `StudyInstanceUID`.

---

## 3. Volumetric DICOM Engineering & Geometry

Knee MRI exams consist of heterogeneous acquisition series with varying orientations, matrix dimensions, and slice thicknesses. All DICOM extraction is implemented in [`src/dicom_utils.py`](src/dicom_utils.py).

### 3.1 3D Spatial Geometry & Normal Slice Sorting
DICOM files are frequently written to disk in non-sequential orders or with arbitrary filenames. Slices are precisely sorted in true physical coordinate space by projecting the `ImagePositionPatient` vector $\vec{p} = [x, y, z]^T$ onto the slice plane normal vector $\vec{n}$:

$$\vec{n} = \vec{u}_{\text{row}} \times \vec{v}_{\text{col}}$$

$$\text{Depth Coordinate } d = \vec{p} \cdot \vec{n} = p_x n_x + p_y n_y + p_z n_z$$

Sorting files by ascending physical depth $d$ guarantees smooth anatomical continuity across the knee joint without spatial jitter.

### 3.2 Photometric Inversion & Rescaling
Medical MRIs may store pixel arrays in signed 16-bit integers with `MONOCHROME1` representation (where zero is white). The pipeline performs:
1. Rescale slope and intercept correction: $I = \text{PixelArray} \times \text{RescaleSlope} + \text{RescaleIntercept}$.
2. Photometric normalization: If `PhotometricInterpretation == "MONOCHROME1"`, values are inverted: $I \leftarrow \max(I) - I$.
3. Joint percentile clipping: 1st and 99th percentiles are calculated jointly across the triplet to eliminate scanner acquisition flare artifacts while preserving cartilage tissue contrast.

### 3.3 Anatomical Laterality Mirroring
Knee joint structures are bilaterally asymmetric. To eliminate unnecessary variance and allow the deep learning backbones to learn canonical left-knee anatomical topologies:
- If `Laterality == "RIGHT"`, the image tensor is horizontally mirrored:
  $$I_{\text{canonical}}(x, y) = I_{\text{raw}}(W - 1 - x, y)$$
- All models operate on left-normalized anatomy.

### 3.4 2.5D Triplet Construction
Full 3D convolutional networks incur severe memory and computational overhead under the 9-hour limit. Instead, the architecture extracts 2.5D volumetric triplets with a slice gap $\Delta = 2$:

$$T_z = \big[ I(z - \Delta), \, I(z), \, I(z + \Delta) \big] \in \mathbb{R}^{3 \times H \times W}$$

Eight equidistant positions are sampled across each of the three cardinal planes (Sagittal, Coronal, Axial), yielding a compact, highly descriptive volumetric representation of each study.

---

## 4. Supervision Engineering: Multilingual LLM Labeling

Because only 58 training studies have gold expert labels, model performance is fundamentally bounded by the quality of pseudo-ground truth derived from the 4,349 unstructured multilingual clinical reports.

### 4.1 Regex-Based Parser with Severity Modulation
The baseline labeler ([`src/report_labeler.py`](src/report_labeler.py)) implements regex parsing with clinical negative assertions across English, Spanish, French, German, and Portuguese:
- **Explicit Negative**: Mentions finding in negated context ("no evidence of tear", "sin derrame", "aucun signe") $\to \text{target} = 0.05, \text{weight} = 1.0$.
- **Explicit Positive with Severe Qualifier**: ("complete tear", "full-thickness rupture", "marked effusion") $\to \text{target} = 0.85, \text{weight} = 1.0$.
- **Explicit Positive with Mild Qualifier**: ("trace", "small", "minimal", "borderline") $\to \text{target} = 0.35, \text{weight} = 0.7$ *(critical calibration against radiologist grading criteria)*.
- **Unmentioned Pathology**: $\to \text{target} = 0.15, \text{weight} = 0.30$.

### 4.2 Zero-Shot Multilingual NLI LLM (`mDeBERTa-v3`)
In Phase 2 ([`src/report_labeler_llm.py`](src/report_labeler_llm.py)), report extraction was upgraded using a multilingual Natural Language Inference foundation model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`):
- For each report and each of the 12 findings, paired hypotheses are evaluated:
  - Premise: $R$ (Radiology Report text)
  - Positive Hypothesis $H^+$: *"The MRI shows [finding description]."*
  - Negative Hypothesis $H^-$: *"There is no evidence of [finding description]."*
- Entailment probabilities $P(H^+ \mid R)$ and $P(H^- \mid R)$ are combined:
  $$\hat{y}_{\text{LLM}} = \text{clip}\Big( P(H^+ \mid R) - 0.5 \cdot P(H^- \mid R), \; 0.05, \; 0.95 \Big)$$

### 4.3 Empirical Bayes Calibration with Pooled Shrinkage
Using the 58 gold-annotated studies, soft label expectations are calibrated via Empirical Bayes:
- Enforces strict monotonic ordering: $\mu_{\text{pos}} > \mu_{\text{unmentioned}} + 0.05 > \mu_{\text{neg}} + 0.10$.
- Applies hierarchical shrinkage pooled across all findings when per-finding sample support is sparse ($N < 5$).

---

## 5. Model Architectures & Training Strategies

### 5.1 Vision Transformer Attention Architecture (DINOv2 / RadDINO)
Implemented in [`src/model.py`](src/model.py) and [`kaggle/train_v2.ipynb`](kaggle/train_v2.ipynb):
- **Trunk**: DINOv2 (`dinov2_vits14` with dim=384, `dinov2_vitb14` with dim=768) and RadDINO (radiology-specialized ViT-B/16).
- **Target-Specific Attention Pooling**: Rather than simple mean pooling across slices, 12 learned query vectors attend independently over the 8 slice position features per plane:
  $$\alpha_{c, k} = \frac{\exp\big( \mathbf{q}_c^T \mathbf{k}_p / \sqrt{d} \big)}{\sum_{j=1}^8 \exp\big( \mathbf{q}_c^T \mathbf{k}_j / \sqrt{d} \big)}, \quad \mathbf{h}_{c, \text{plane}} = \sum_{k=1}^8 \alpha_{c, k} \mathbf{v}_k$$
- **Multi-Plane & Metadata Fusion**: Attended features from Sagittal, Coronal, and Axial planes are concatenated with 3 acquisition scalars `[Fluid_Sensitive, Fat_Suppression, Is_Right_Mirrored]`.
- **Classification Heads**: 12 independent multi-layer perceptron heads (`Linear → LayerNorm → GELU → Dropout(0.2) → Linear → Sigmoid`).

### 5.2 ConvNeXt & CoAtNet MIL Architectures
In the Apex SOTA pipeline (`versia_10` through `versia_17`):
- **Raptor CoAtNet MIL**: Combines multi-instance learning with hybrid convolution-attention backbones trained at dual spatial resolutions (384 px and 224 px).
- **ConvNeXt-Small & OrthoFoundation**: Pure convolutional architecture with deep receptive fields, optimized for high spatial resolution cartilage assessment and bone marrow edema boundaries.

### 5.3 Asymmetric Loss (ASL) for Sparse Pathology Imbalance
To counteract extreme class sparsity (e.g., Fractures occur in $< 3\%$ of exams):
$$L_{\text{ASL}} = - \sum_{i} \Big[ y_i (1 - p_i)^{\gamma^+} \log(p_i) + (1 - y_i) (p_{m, i})^{\gamma^-} \log(1 - p_{m, i}) \Big]$$
where $p_{m, i} = \max(p_i - m, 0)$ with asymmetric focusing parameters $\gamma^- = 4$, $\gamma^+ = 0$, and margin $m = 0.05$. Gold labels receive a loss weighting multiplier of **$8.0\times$**, while pseudo-labels are scaled by their calibrated confidence weights.

---

## 6. High-Performance Inference Engine (Apex SOTA)

Kaggle code competitions require running the entire test inference inside an offline container within 9 hours. For large ensembles, unoptimized inference times out. The Versia 17 SOTA architecture introduces multiple systems-level performance optimizations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Dual-GPU Parallel Worker Pipeline                    │
│                                                                        │
│   Study Queue (Test Set)                                               │
│            │                                                           │
│     ┌──────┴──────┐                                                    │
│     ▼             ▼                                                    │
│ [Worker 0]   [Worker 1]    • Independent Python processes              │
│ (Tesla T4)   (Tesla T4)    • Device-isolated memory spaces             │
│     │             │        • Asynchronous study dispatch               │
│     │             │                                                    │
│     ├─────────────┴────────────────────────────────────────────────┐   │
│     │   In-Worker Low-Latency Execution:                           │   │
│     │   1. _fastread.py & Direct I/O DICOM decompression           │   │
│     │   2. Dynamic-shape PyTorch 2 Inductor Compilation:           │   │
│     │      torch.compile(model, mode="default", backend="inductor")│   │
│     │   3. FP16 Autocast TensorCore GEMM Kernels                   │   │
│     │   4. 4-Way Flip Test-Time Augmentation (TTA)                 │   │
│     └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Dual-Worker Multiprocessing**: Spawns two isolated worker processes assigned strictly to `cuda:0` and `cuda:1`, eliminating Python GIL contention during DICOM decompression.
2. **`torch.compile` with TorchDynamo & Inductor**: Checkpoint models are compiled to fused CUDA kernels for supported slice shapes `[32, 64, 96, 128, 160, 192]`. Max deviation between compiled and eager logits is validated to stay within $| \Delta p | \le 0.000213$, cutting GPU forward pass time to **19.7 seconds per study**.
3. **`_fastread.py` Async I/O**: Direct memory decompression bypassing Python file buffers, achieving an average slice decode rate of **36.0 seconds per full study**.

---

## 7. Ensemble, Per-Finding Weights & Calibration

Because different backbones develop specialized sensitivities to specific tissue types, simple uniform averaging degrades the leader score. A **per-finding rank-blend** is executed across the model ensemble ([`src/ensemble.py`](src/ensemble.py) and [`src/calibrate.py`](src/calibrate.py)).

### 7.1 Pathology-Specialized Weights Matrix
Weights between the Raptor CoAtNet MIL models and the ConvNeXt / OrthoFoundation models were tuned through cross-validation and leaderboard probes:

```
┌─────────────────────┬─────────────────┬─────────────────┐
│ Finding             │ Raptor Weight   │ ConvNeXt Weight │
├─────────────────────┼─────────────────┼─────────────────┤
│ Synovitis           │     0.80        │     0.20        │
│ Patellofemoral OA   │     0.60        │     0.40        │
│ Medial OA           │     0.50        │     0.50        │
│ ACL                 │     0.50        │     0.50        │
│ MCL                 │     0.50        │     0.50        │
│ Medial Meniscus     │     0.50        │     0.50        │
│ Effusion            │     0.50        │     0.50        │
│ Contusion           │     0.50        │     0.50        │
│ Fracture            │     0.50        │     0.50        │
│ Baker's Cyst        │     0.35        │     0.65        │
│ Lateral OA          │     0.20        │     0.80        │
└─────────────────────┴─────────────────┴─────────────────┘
```

### 7.2 Uniform Rank Projection
ROC-AUC evaluates only the monotonic ranking of predictions rather than absolute probabilities. Raw ensembled probabilities are converted to uniform rank percentiles:

$$\hat{y}_{\text{rank}} = \frac{\text{Rank}(\hat{y}) - 1}{N - 1} \in [0, 1]$$

This rank transformation guarantees mathematical preservation of within-model ranking while eliminating calibration distribution drift across different model architectures.

---

## 8. Leaderboard Progression & Milestones

The systematic engineering progression across competition iterations:

| Iteration / Milestone | Key Architectural Changes | Validated Macro ROC-AUC | Status |
|:---|:---|:---:|:---:|
| **Public V01 Baseline** | Single-plane EfficientNet-B0, binary hard regex labels | `0.613` | Baseline |
| **Public V03 Baseline** | 2.5D multi-plane EfficientNet-B0, calibrated soft labels | `0.664` | Baseline |
| **Public DINOv2-S** | Pilkwang public baseline, single ViT-S/14 model | `0.809` | Benchmark |
| **Phase 0 & 1** | 2.5D DINOv2-small, spatial normal sorting, left-knee mirror | `0.812` | Completed |
| **Phase 2 (LLM Labels)** | Multilingual `mDeBERTa-v3` zero-shot NLI extraction | `0.834` | Completed |
| **Phase 3 (DINOv2-B)** | DINOv2-base (ViT-B/14) + Asymmetric Loss + 4x Flip TTA | `0.852` | Completed |
| **Phase 4 (Multimodal)**| BiomedBERT & mE5 late gated text fusion | `0.871` | Completed |
| **Tri-Specialist Stack** | Multi-plane ensemble + initial rank calibration | `0.945` | Milestone |
| **Versia 8 - 10** | Dual CoAtNet MIL (384 + 224 px) + R6 ConvNeXt integration | `0.954` | Milestone |
| **Versia 11 - 15** | Microboost per-finding recalibration + Inductor compilation | `0.956` | Milestone |
| **Versia 16 - 17 (Apex SOTA)** | Dual-GPU Tesla T4 parallel workers + optimized per-class weights | **`0.957+`** | **Personal Best** |

---

## 9. Repository Structure & Code Map

```
rsna_knee/
├── README.md                          # Comprehensive project documentation (this file)
├── rsna_knee_plan.md                  # Master development specification & engineering diary
│
├── src/                               # Modular core python package
│   ├── __init__.py                    # Package exports and public API
│   ├── dicom_utils.py                 # 3D normal vector slice sorting, windowing, mirroring
│   ├── report_labeler.py              # Multilingual regex rule parser & soft target calibration
│   ├── report_labeler_llm.py          # mDeBERTa-v3 multilingual zero-shot NLI labeler
│   ├── dataset.py                     # PyTorch Dataset for 2.5D multi-plane triplets
│   ├── model.py                       # DINOv2 multi-plane model with 12 target attention queries
│   ├── train.py                       # Fold training loops, mixed-precision AMP, ASL loss
│   ├── infer.py                       # Inference harness and submission generation
│   ├── ensemble.py                    # Multi-model per-label optimizer & rank-blender
│   └── calibrate.py                   # Temperature scaling & uniform rank calibration
│
├── scripts/                           # Local build and auditing utilities
│   ├── audit.py                       # Dataset integrity auditor & label frequency analyzer
│   ├── eval_labeler.py                # Evaluates pseudo-labelers against 58 gold ground truths
│   ├── build_versia_8.py              # Generator for Versia 8 SuperBlend deployment
│   ├── build_versia_9.py              # Generator for Versia 9 Tri-Ensemble deployment
│   ├── build_versia_10.py             # Generator for Versia 10 SOTA deployment
│   ├── build_versia_11.py             # Generator for Versia 11 MicroBoost deployment
│   ├── fetch_real_scores.py           # Real-time Kaggle API public leaderboard scraper
│   ├── inspect_scouts.py              # Model architecture and weight inspector
│   ├── leaderboard_sim.py             # OOF CV simulation & scanner memorization detector
│   └── generate_train_notebook.py     # Compiles src/ into standalone Kaggle train notebooks
│
├── kaggle/                            # Local reference Kaggle notebooks
│   ├── train_v1.ipynb                 # Phase 1 DINOv2-small training notebook
│   ├── train_v2.ipynb                 # Phase 2 DINOv2-base + ASL training notebook
│   ├── train_v3_multimodal.ipynb      # Phase 4 Vision + Language fusion notebook
│   ├── train_efficient.ipynb          # Fast-track lightweight training notebook
│   ├── run_llm_labeler.ipynb          # Offline mDeBERTa report extraction notebook
│   └── infer_final.ipynb              # Final 5-fold ensemble inference notebook
│
├── kaggle_remote/                     # Remote Kaggle execution artifacts & metadata
│   ├── versia_17/                     # Champion Apex SOTA deployment package
│   │   ├── rsna-versia-17-apex-sota.ipynb # Production dual-worker execution notebook
│   │   ├── kernel-metadata.json       # Kaggle kernel configuration & attached datasets
│   │   ├── build_nb.py                # Notebook assembly script
│   │   └── output/                    # Production receipts, cufile logs & submission.csv
│   ├── versia_16/                     # Versia 16 Ultra SOTA package
│   ├── versia_11/                     # Versia 11 MicroBoost package
│   ├── versia_10/                     # Versia 10 SOTA package
│   ├── v2_done/                       # Trained DINOv2-base fold checkpoints & logs
│   └── v1_done/                       # Trained DINOv2-small fold checkpoints & logs
│
└── tests/                             # Pytest test suite
    ├── test_dicom_utils.py            # Unit tests for 3D sorting, clipping, and mirroring
    ├── test_report_labeler.py         # Unit tests for multilingual regex parser
    ├── test_report_labeler_llm.py     # Unit tests for zero-shot NLI extraction
    ├── test_ensemble.py               # Unit tests for rank blending and optimization
    ├── test_calibrate.py              # Unit tests for uniform rank projection
    └── test_notebook_syntax.py        # Syntax and integrity validation for Kaggle notebooks
```

---

## 10. Reproduction & Execution Guide

### 10.1 Environment Setup
```bash
# Clone repository and configure environment
git clone https://github.com/kunaldesale2408/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pydicom timm transformers scipy scikit-learn pandas numpy pytest
```

### 10.2 Running Unit Tests & Audits
```bash
# Execute full pytest suite
pytest tests/ -v

# Run local dataset audit (requires competition CSVs in data/)
python scripts/audit.py --train-csv data/train.csv --series-csv data/train_series.csv
```

### 10.3 Deploying to Kaggle
All production notebooks are configured for automated push via the Kaggle CLI:

```bash
# Push Versia 17 Apex SOTA Kernel to Kaggle
cd kaggle_remote/versia_17
kaggle kernels push -p .

# Monitor real-time GPU execution status
kaggle kernels status kunaldesale2408/rsna-versia-17-apex-sota
```

---

## 11. Key Engineering Insights & Pitfalls Avoided

1. **Beware the "Small Joint Effusion" Penalty**: In standard radiology, "small effusion" denotes abnormal fluid accumulation. However, in the RSNA ground truth, borderline/small cases were graded **negative**. Assigning hard $1.0$ targets to mild findings severely penalizes model calibration. Setting mild soft targets to $\approx 0.35$ immediately lifted macro AUC by **$+0.041$**.
2. **Scanner Memorization Is Severe**: Grouping by `StudyInstanceUID` in `GroupKFold` is mandatory. Random k-fold splits leak acquisition parameters across folds, creating an illusion of high validation score while crashing on the unseen test set.
3. **Dynamic Graph Compiles for Multiple Slices**: When applying `torch.compile(..., backend="inductor")`, medical series with differing slice counts trigger recompilations. Restricting compilation to pre-bucketed shapes (`[32, 64, 96, 128, 160, 192]`) prevented runtime memory thrashing while delivering full execution speed.
4. **Anatomical Mirroring Normalization**: Horizontally flipping right knees to left-knee coordinates reduces spatial ambiguity for the attention query heads, improving medial vs. lateral pathology separation.
5. **Rank Blending Outperforms Probability Averaging**: For multi-model ensembles where individual models have differing confidence calibrations, ranking predictions before linear combination guarantees strict monotonic preservation and safeguards against leaderboard distribution shifts.

---

## License & Attribution
- Built for the **RSNA Knee Abnormality Detection** Kaggle Competition.
- Dataset and annotations provided by the **Radiological Society of North America (RSNA)** under competition rules.
- Author: **Kunal Desale** ([@kunaldesale2408](https://github.com/kunaldesale2408))
