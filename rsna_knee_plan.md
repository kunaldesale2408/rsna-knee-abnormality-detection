# RSNA Knee Abnormality Detection — Gemini CLI Build Plan
# Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
# Metric: Macro ROC-AUC across 12 binary labels | Deadline: Oct 22 2026
# Goal: Top 2% | Code competition — runs Kaggle-only, ≤9 h GPU, internet OFF at submit time

---

## COMPETITION SNAPSHOT (read first)

| Fact | Detail |
|---|---|
| Task | Multilabel classification — predict 12 binary knee-abnormality probs per study |
| Metric | Macro ROC-AUC (12-label average) |
| Input | Multi-series DICOM MRI (sagittal / coronal / axial) + multilingual radiology reports |
| Training studies | 4,407 total; only **58 have gold labels**; the other 4,349 have reports only |
| Gold label agreement | Report-derived labels agree ~82% with gold — noisy supervision |
| Key constraint | Code competition: final notebook must run end-to-end ≤9 h on a Kaggle GPU (T4/P100), internet OFF |
| Leading public score | ~0.809 (DINOv2 backbone; "pilkwang" public notebook) |
| Strong baseline score | 0.664 (2.5D EfficientNet-B0 with hierarchical calibrated soft labels) |

### The 12 Labels
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's Cyst, Contusion, Fracture

### Critical insight from public analysis
> The competition is won in how you turn "small joint effusion" into a number.
> Gold labels use explicit severity thresholds — "borderline" was graded negative.
> "small effusion" in a report → correct target ≈ 0.2, NOT 1.0.
> Random k-fold inflates AUC by ~0.053 through scanner memorisation → always use GroupKFold on StudyInstanceUID.

---

## HOW TO USE THIS PLAN WITH GEMINI CLI

Each PHASE below is a self-contained prompt block you paste into Gemini CLI.
- Work in a Kaggle Notebook (not VS Code) for actual training — no local GPU.
- Use VS Code only for: report-labeler development, data EDA, prompt iteration, file prep.
- All Kaggle notebooks must be uploaded via Kaggle CLI (`kaggle kernels push`).
- Attach the competition dataset when pushing: `--dataset rsna-knee-abnormality-detection`.

---

## PHASE 0 — Environment & Data Audit (VS Code, ~2 hours)

### Gemini CLI Prompt — Phase 0

```
You are helping me build a solution for the Kaggle competition "rsna-knee-abnormality-detection".

TASK: Set up the local project structure and write a data-audit script.

PROJECT STRUCTURE to create:
rsna_knee/
  data/            # competition CSVs only — no DICOMs locally (too large)
  src/
    report_labeler.py
    dicom_utils.py
    dataset.py
    model.py
    train.py
    infer.py
  kaggle/
    train_v1.ipynb
    infer_v1.ipynb
  results/

STEP 1 — Create all directories and __init__ files.

STEP 2 — Write src/report_labeler.py.
This module converts multilingual radiology report text into soft probability labels
for 12 findings. Requirements:
- Accept raw report text (any of 12 languages).
- Use regex keyword rules for each finding in at least English, Spanish, French, German, Portuguese.
- For each finding produce a state: "explicit_positive", "explicit_negative", "unmentioned".
- Return a dict {finding: (state, confidence_weight)} where:
  * explicit_positive  → soft_target = 0.85, weight = 1.0  (NOT 1.0 — borderline was graded neg)
  * explicit_negative  → soft_target = 0.05, weight = 1.0
  * unmentioned        → soft_target = 0.15, weight = 0.3  (very low weight)
- For severity qualifiers ("small", "mild", "trace", "minimal"):
  * Downgrade explicit_positive soft_target to 0.35
- For severity qualifiers ("large", "complete", "full-thickness", "severe"):
  * Keep explicit_positive soft_target at 0.85
- The 12 findings and their key English terms:
  * ACL: ["anterior cruciate", "acl", "a.c.l"]
  * MCL: ["medial collateral", "mcl", "m.c.l"]
  * Medial Meniscus: ["medial meniscus", "medial meniscal"]
  * Lateral Meniscus: ["lateral meniscus", "lateral meniscal"]
  * Medial OA: ["medial compartment", "medial tibiofemoral", "medial joint space narrowing"]
  * Lateral OA: ["lateral compartment", "lateral tibiofemoral", "lateral joint space narrowing"]
  * PF OA: ["patellofemoral", "pf compartment", "patellar cartilage"]
  * Effusion: ["effusion", "joint fluid", "knee fluid"]
  * Synovitis: ["synovitis", "synovial thickening", "synovial enhancement"]
  * Baker's: ["baker", "popliteal cyst", "posterior cyst"]
  * Contusion: ["contusion", "bone bruise", "bone marrow edema", "bme"]
  * Fracture: ["fracture", "fracture line", "insufficiency fracture"]
- Add a function calibrate_soft_labels(df_reports, df_gold_58) that:
  * Computes per-fold empirical-Bayes calibration
  * Enforces monotonic ordering: positive_mean > unmentioned_mean > negative_mean (with 0.05 margin)
  * Falls back to hierarchical shrinkage pooled across all 12 labels when per-label support < 5

STEP 3 — Write src/dicom_utils.py with these functions:
- select_best_series(series_df, study_uid): picks one series per plane (Sagittal/Coronal/Axial)
  Priority: Fluid_Sensitive=1 AND Fat_Suppression=1 > either alone > any
- load_dicom_series(dicom_paths, n_slices=32, img_size=320):
  * Sorts slices via ImagePositionPatient projection onto ImageOrientationPatient normal
  * Falls back to InstanceNumber, then filename
  * Applies RescaleSlope + RescaleIntercept
  * Handles MONOCHROME1 inversion and signed pixels
  * Joint clips triplet at 1st/99th percentile
  * Scales to [0,1], resizes to img_size x img_size, applies ImageNet normalization
  * Returns tensor of shape [n_slices, 3, img_size, img_size] (triplets with gap=2)
- mirror_right_knee(tensor): mirrors horizontally if laterality == "RIGHT" (so all knees are Left-normalized)

STEP 4 — Write a simple data audit script scripts/audit.py that:
- Reads train.csv and train_series.csv (paths passed as args)
- Prints: total studies, studies with gold labels, series count per plane, slice count stats
- Validates: all studies in train.csv have at least one Sagittal + Coronal + Axial series
- Saves weak labels using report_labeler.py to results/weak_labels_v0.csv
```

---

## PHASE 1 — Baseline Notebook (Kaggle, ~1 session)

### Gemini CLI Prompt — Phase 1

```
You are helping me build a Kaggle notebook for the "rsna-knee-abnormality-detection" competition.

CONTEXT:
- 4,407 training studies; 58 have gold labels; rest have only radiology reports
- Metric: macro ROC-AUC over 12 binary labels
- Code competition: notebook must finish in ≤9 h on Kaggle GPU (T4/P100), internet OFF
- GroupKFold on StudyInstanceUID (NEVER random split — scanner memorisation inflates AUC by ~0.05)

TASK: Write kaggle/train_v1.ipynb — a complete self-contained Kaggle training notebook.

ARCHITECTURE (2.5D multi-plane with DINOv2):
  For each study:
    - Load best series per plane (Sagittal, Coronal, Axial)
    - Sort slices spatially
    - Mirror right knees (laterality normalization)
    - Sample 8 positions per plane
    - For each position construct a 2.5D triplet: [slice-2, slice, slice+2]
    - Each triplet → shared DINOv2-small (ViT-S/14, dinov2_vits14) backbone
    - Per-plane: target-specific attention pooling across the 8 positions
      (12 query vectors, each attending over 8 position features → 12 attended features)
    - Concatenate 3-plane attended features + 3 metadata scalars
      [fluid_sensitive, fat_suppression, is_right_knee_mirrored]
    - 12 independent sigmoid heads (one MLP per label: Linear→ReLU→Linear→sigmoid)
    - Loss: confidence-weighted BCE with soft targets (from weak labels + gold labels)
      gold label weight = 8.0; report-derived weight = per-finding confidence from calibration

CONFIG (put in a dataclass cfg):
  img_size = 224
  n_samples_per_plane = 8
  triplet_gap = 2
  backbone = "dinov2_vits14"
  backbone_dim = 384
  n_folds = 5
  folds_to_train = (0,)   # start with fold 0 for speed; full run uses (0,1,2,3,4)
  batch_size = 1
  grad_accum = 4
  lr = 3e-5
  epochs = 5
  runtime_limit_hours = 8.5
  amp = True
  img_size = 224

WEAK LABELS:
  - Load results/weak_labels_v0.csv (pre-computed from Phase 0 report_labeler)
  - OR if not found, compute inline using the report_labeler logic (embed it)

TRAINING LOOP:
  - AMP (torch.cuda.amp.autocast)
  - AdamW optimizer, cosine scheduler with warmup (first 10% steps)
  - Gradient clipping (max_norm=1.0)
  - Per-epoch: compute macro AUC on gold-label validation subset (58 studies)
  - Save best checkpoint by: (a) soft-label val loss, (b) gold-AUC
  - Runtime guard: if elapsed > runtime_limit_hours, save last.pt and exit gracefully
  - Auto-resume: on startup, check for last.pt and resume training from there
  - After all folds complete: ensemble best.pt predictions → submission.csv
  - Print collapse diagnostic each epoch: per-label prediction std (warn if < 0.05)

VALIDATION:
  - GroupKFold(n_splits=5) on StudyInstanceUID
  - Log: per-class AUC (all 4,407 soft-label val), per-class AUC (58 gold val)
  - Log: macro AUC (soft), macro AUC (gold)

SUBMISSION:
  - Use sample_submission.csv column order
  - Ensemble across folds by averaging predicted probabilities
  - Write /kaggle/working/submission.csv

IMPORTANT CONSTRAINTS:
  - No internet calls (internet is OFF at submission time)
  - DINOv2 weights must be loaded from Kaggle Dataset (not torch.hub)
    Use: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
    then load weights from /kaggle/input/dinov2-weights/dinov2_vits14_pretrain.pth
  - The notebook must be a single .ipynb with no external imports beyond standard Kaggle env
  - Use pydicom for DICOM reading, not SimpleITK
```

---

## PHASE 2 — LLM-Enhanced Report Labeling (VS Code → Kaggle, ~3 hours)

> **Why**: The baseline rule-based labeler has ~82% agreement with gold.
> Better labels → free AUC lift on every model you train afterwards.

### Gemini CLI Prompt — Phase 2

```
You are improving the report labeler for the RSNA Knee competition.

CURRENT STATE: src/report_labeler.py uses regex keyword rules → ~82% agreement with 58 gold labels.

TASK: Write src/report_labeler_llm.py — a better labeler using a small multilingual model.

METHOD: Use a zero-shot NLI (Natural Language Inference) classifier to extract findings.
Model: "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" (multilingual, ~280MB, fits on Kaggle)
This model is available at HuggingFace and can be loaded offline from a Kaggle Dataset.

For each report text:
  For each of 12 findings:
    Build hypothesis: "The MRI shows [finding description]."
    Run NLI: entailment prob → positive confidence
    Also check negation: "There is no evidence of [finding description]."
    NLI entailment for negation → negative confidence
    Soft target = entailment_prob - 0.5 * negation_prob, clipped to [0.05, 0.95]

FINDING DESCRIPTIONS (use these as NLI hypotheses):
  ACL: "anterior cruciate ligament tear or abnormality"
  MCL: "medial collateral ligament tear or abnormality"
  Medial Meniscus: "medial meniscus tear or abnormality"
  Lateral Meniscus: "lateral meniscus tear or abnormality"
  Medial OA: "medial tibiofemoral osteoarthritis or joint space narrowing"
  Lateral OA: "lateral tibiofemoral osteoarthritis or joint space narrowing"
  PF OA: "patellofemoral osteoarthritis or cartilage loss"
  Effusion: "knee joint effusion or fluid"
  Synovitis: "synovitis or synovial thickening"
  Baker's: "Baker cyst or popliteal cyst"
  Contusion: "bone contusion or bone marrow edema"
  Fracture: "fracture"

Also write:
  - scripts/eval_labeler.py: compares labeler output vs 58 gold labels, prints per-label AUC
  - A Kaggle notebook snippet (markdown cell + code cell) to:
    1. Install transformers offline
    2. Load the mDeBERTa model from /kaggle/input/mdeberta-weights/
    3. Run labeler on all 4,407 reports
    4. Save results/weak_labels_llm.csv
    5. Compare against regex labeler on gold AUC

Target: LLM labeler should beat regex labeler by ≥0.03 macro AUC on the 58 gold studies.
If it doesn't, fall back to ensemble: 0.5 * regex_soft + 0.5 * llm_soft.
```

---

## PHASE 3 — Stronger Vision Backbone (Kaggle, ~2 sessions)

> **Why**: DINOv2-small (0.809 public) can be pushed further with ViT-B/14 or RadDINO.

### Gemini CLI Prompt — Phase 3

```
You are upgrading the vision backbone for the RSNA Knee competition notebook.

CONTEXT: Current best public score uses DINOv2-small (ViT-S/14, dim=384). 
We want to try DINOv2-base (ViT-B/14, dim=768) and RadDINO (medical-domain ViT-B).

TASK A: Modify kaggle/train_v1.ipynb into kaggle/train_v2.ipynb with these changes:

1. Make backbone configurable:
   cfg.backbone = "dinov2_vitb14"   # or "raddino" or "dinov2_vits14"
   cfg.backbone_dim = 768           # 384 for vits14
   cfg.img_size = 224

2. For "raddino" backbone:
   - Load microsoft/rad-dino from HuggingFace (ViT-B/16 trained on chest X-rays and radiology)
   - Available offline from /kaggle/input/raddino-weights/
   - It outputs 768-dim CLS token — use as drop-in replacement for DINOv2

3. Implement Asymmetric Loss (ASL) instead of weighted BCE:
   from torch.nn import functional as F
   def asymmetric_loss(pred, target, gamma_neg=4, gamma_pos=0, clip=0.05):
       # Standard ASL — helps with class imbalance on rare labels like Fracture
       xs_pos = pred
       xs_neg = 1 - pred
       xs_neg = (xs_neg + clip).clamp(max=1)
       los_pos = target * torch.log(xs_pos.clamp(min=1e-8))
       los_neg = (1 - target) * torch.log(xs_neg.clamp(min=1e-8))
       neg_weight = (1 - xs_neg) ** gamma_neg
       los_neg = neg_weight * los_neg
       loss = los_pos + los_neg
       return -loss.mean()
   
   Weight gold labels by 8.0 as before; use confidence weight for soft labels.

4. Add test-time augmentation (TTA) at inference:
   - Run inference 4x: original + horizontal flip + vertical flip + both flips
   - Average the 4 predictions
   (For laterality-normalized knees, flip augmentation is valid)

5. Train: full 5-fold, submit ensemble.

TASK B: Write a separate efficiency-track notebook kaggle/train_efficient.ipynb:
  - Use DINOv2-small (ViT-S/14)
  - Reduce n_samples_per_plane = 4 (from 8)
  - Reduce img_size = 160
  - Use torch.compile() if available
  - Target: full 5-fold under 4 hours on T4
  - This targets the competition's "efficiency track"
```

---

## PHASE 4 — Multimodal Fusion (Kaggle, ~1 session)

> **Why**: The competition explicitly provides radiology reports — top solutions likely fuse them at prediction time, not just at label time.

### Gemini CLI Prompt — Phase 4

```
You are adding text fusion to the RSNA Knee competition notebook.

CONTEXT: We have an image model (macro AUC ~0.82 target). Reports are in 12 languages.
The vision model already uses report-derived soft labels for training.
Now we want to use report embeddings at INFERENCE time (not just for label extraction).

TASK: Write kaggle/train_v3_multimodal.ipynb — adds a text branch:

1. Text encoder: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" (English reports)
   OR "intfloat/multilingual-e5-base" (for non-English reports — auto-detect language)
   Load from /kaggle/input/biomedbert-weights/ or /kaggle/input/me5-weights/

2. Language detection: use langdetect or simple heuristic (CLD3).
   Route English reports to BiomedBERT, others to mE5.

3. Text features: mean-pool last 4 hidden layers, project to 256-dim.

4. Fusion: late fusion
   - Image model outputs 12 logits (pre-sigmoid)
   - Text model outputs 12 projected logits (Linear(256, 12))
   - Gated fusion: gate = sigmoid(Linear(image_logit + text_logit, 12))
     final_logit = gate * image_logit + (1 - gate) * text_logit
   - Single BCE training signal on final_logit
   - Image branch and text branch trained jointly from scratch (both initialized with pretrained weights)
   - Freeze text encoder for first 2 epochs, then unfreeze

5. At inference on test set: reports ARE available (competition provides them)
   Use the same text branch at test time.

6. Ablation cell: compare image-only vs image+text on 58 gold validation studies.
   Print AUC delta per label.

IMPORTANT: text encoder weights must be loaded offline from Kaggle Dataset.
Use tokenizer.save_pretrained() locally, upload to Kaggle Dataset, load with:
  AutoTokenizer.from_pretrained('/kaggle/input/biomedbert-weights/')
  AutoModel.from_pretrained('/kaggle/input/biomedbert-weights/')
```

---

## PHASE 5 — Ensemble & Post-Processing (VS Code + Kaggle)

### Gemini CLI Prompt — Phase 5

```
You are building the final ensemble and post-processing pipeline for RSNA Knee competition.

CONTEXT:
- We have OOF predictions from multiple models:
  * v1: DINOv2-small, rule-based labels (macro AUC ~0.80)
  * v2: DINOv2-base + ASL + TTA, LLM labels (macro AUC ~0.83 target)
  * v3: multimodal image+text (macro AUC ~0.84 target)
- Final submission = ensemble of all

TASK A: Write src/ensemble.py
  - Load all OOF prediction CSVs
  - For each of 12 labels independently, find optimal per-model weights using scipy.optimize on OOF gold labels (58 studies)
  - Apply those weights to test predictions
  - Save final_submission.csv

TASK B: Write src/calibrate.py
  - Temperature scaling per label: find T that maximises AUC on 58 gold OOF
    (AUC is rank-invariant so temperature scaling helps only if you threshold, but check)
  - Rank-based calibration: transform each label's predictions to uniform [0,1] using rank/N
    (safe for AUC — preserves order, helps leaderboard stability)

TASK C: Write a final inference-only Kaggle notebook kaggle/infer_final.ipynb:
  - Loads all model checkpoints (v1 fold0-4, v2 fold0-4, v3 fold0-4)
  - Runs TTA (4 augmentations) for each checkpoint
  - Applies ensemble weights from Phase 5A
  - Must complete in ≤9 h on Kaggle GPU
  - Writes /kaggle/working/submission.csv
  - Validate total runtime estimate is printed at start

TASK D: Write scripts/leaderboard_sim.py
  - Computes expected public LB macro AUC from OOF predictions
  - Checks for signs of overfitting: compare gold-only AUC vs soft-label AUC
  - Warns if gap > 0.05 (sign of scanner memorisation)
```

---

## PUBLIC DATASETS TO ADD (upload as Kaggle Datasets)

| Dataset | Purpose | How to use |
|---|---|---|
| **Stanford MRNet** | 1,370 knee MRIs, ACL + meniscus labels — pre-train vision backbone on knee anatomy before RSNA fine-tuning | Pre-train DINOv2 head on MRNet (ACL/meniscus labels), then fine-tune on RSNA |
| **DINOv2 weights** (`facebookresearch/dinov2` — `dinov2_vits14_pretrain.pth` and `dinov2_vitb14_pretrain.pth`) | Backbone weights for internet-off notebooks | Upload to Kaggle Dataset, load from `/kaggle/input/dinov2-weights/` |
| **mDeBERTa weights** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) | Multilingual NLI for report labeling (Phase 2) | Upload to `/kaggle/input/mdeberta-weights/` |
| **RadDINO weights** (`microsoft/rad-dino`) | Radiology-domain ViT-B backbone | Upload to `/kaggle/input/raddino-weights/` |
| **multilingual-e5-base weights** (`intfloat/multilingual-e5-base`) | Text encoder for non-English reports (Phase 4) | Upload to `/kaggle/input/me5-weights/` |
| **BiomedBERT weights** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) | Text encoder for English reports (Phase 4) | Upload to `/kaggle/input/biomedbert-weights/` |

### How to upload model weights to Kaggle Dataset (run in VS Code terminal)

```bash
# 1. Download weights locally
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebookresearch/dinov2', local_dir='./weights/dinov2')
snapshot_download('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', local_dir='./weights/mdeberta')
snapshot_download('microsoft/rad-dino', local_dir='./weights/raddino')
snapshot_download('intfloat/multilingual-e5-base', local_dir='./weights/me5')
snapshot_download('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', local_dir='./weights/biomedbert')
"

# 2. Create Kaggle dataset metadata
kaggle datasets init -p ./weights/dinov2

# 3. Push to Kaggle
kaggle datasets create -p ./weights/dinov2 --dir-mode zip
```

---

## MRNet PRE-TRAINING GEMINI PROMPT

```
You are helping me pre-train a vision backbone on the Stanford MRNet dataset before fine-tuning on the RSNA knee competition.

CONTEXT:
- MRNet: 1,370 knee MRI exams, 3 planes (sagittal/coronal/axial), labels: abnormal/ACL/meniscus
- RSNA: 4,407 exams, same 3 planes, 12 labels
- We want MRNet to teach the backbone "what a knee looks like" before RSNA fine-tuning
- MRNet data is in .npy format per plane, not DICOM

TASK: Write kaggle/pretrain_mrnet.ipynb

1. MRNet loader: load train/val .npy files for each plane
   Each .npy is shape [n_slices, 256, 256] → convert to [n_slices, 3, 224, 224] triplets
2. Pre-training objective: 3-task classification (abnormal, ACL, meniscus) — sigmoid BCE
3. Backbone: DINOv2-small (ViT-S/14) — load from /kaggle/input/dinov2-weights/
4. Architecture: same 2.5D triplet + attention pooling as main model (8 samples/plane)
5. Train 10 epochs on MRNet, save pretrained_mrnet_backbone.pth
6. This checkpoint is then used as cfg.local_backbone_weights in train_v1.ipynb instead of raw DINOv2

Expected benefit: +0.01–0.03 macro AUC on RSNA due to knee-specific feature initialization.
```

---

## TIMELINE (9 days left to deadline Oct 22)

| Day | Action |
|---|---|
| **Oct 13 (today)** | Set up project structure with Phase 0 prompt. Audit data locally. Build report_labeler. |
| **Oct 14** | Upload DINOv2-small + competition dataset to Kaggle. Run Phase 1 notebook (fold 0). Get first LB score. |
| **Oct 15 (entry deadline)** | Entry submitted. Phase 1 full 5-fold if fold 0 looks good. |
| **Oct 16** | Phase 2: LLM labeler (mDeBERTa). Evaluate vs regex. Upload new weak labels. |
| **Oct 17** | Phase 3: DINOv2-base + ASL + TTA. Upload RadDINO weights. |
| **Oct 18** | Phase 3 full 5-fold. Compare v1 vs v2 on LB. |
| **Oct 19** | MRNet pre-training (if time permits). Phase 4: text fusion. |
| **Oct 20** | Phase 5: ensemble. Tune ensemble weights on OOF. |
| **Oct 21** | Final inference notebook. Dry-run under 9 h. Spot-check predictions. |
| **Oct 22** | Submit final. Choose best 2 submissions (Kaggle allows you to select final 2). |

---

## KEY DECISIONS & PITFALLS TO AVOID

1. **NEVER use random k-fold** — always GroupKFold on StudyInstanceUID. Scanner memorisation inflates AUC by 0.05+.
2. **Soft targets, not hard** — gold labels are threshold-based. "small effusion" ≠ positive. Use soft targets ≈ 0.35 for mild findings.
3. **Do not optimize predictions toward the soft-label AUC** — it's much higher than gold AUC. Only use gold OOF for model selection.
4. **Don't collapse to base rate** — watch prediction std per label each epoch. If < 0.05, training is collapsing. Increase confidence floor on unmentioned labels.
5. **Laterality normalization** — mirror right knees so all anatomy is left-normalized. Consistent with how radiologists read.
6. **Code competition rule**: everything must run in the notebook at submission time — no external APIs, no hosted model calls.
7. **Reports in test set** — the competition provides test reports. Use them at inference (multimodal fusion is valid for the final submission).

---

## QUICK REFERENCE — SCORES & TARGETS

| Model | Expected Macro AUC | Status |
|---|---|---|
| Public V01 baseline (EfficientNet-B0, rule labels) | 0.613 | Reference |
| Public V03 (EfficientNet-B0, calibrated soft labels) | 0.664 | Reference |
| Public DINOv2-small baseline (pilkwang) | 0.809 | Reference |
| **Phase 0** (Environment, Data Audit, DICOM utils, Regex Labeler) | N/A | Complete (Tests Passing) |
| **Phase 1** (DINOv2-small + hierarchical labels `train_v1.ipynb`) | ~0.81 | Complete (Tests Passing) |
| **Phase 2** (+ LLM labels `run_llm_labeler.ipynb`) | ~0.83 | Complete (Tests Passing) |
| **Phase 3** (DINOv2-base + ASL + TTA `train_v2.ipynb` & `train_efficient.ipynb`) | ~0.85 | Complete (Tests Passing) |
| **Phase 4** (+ Multimodal Text Fusion `train_v3_multimodal.ipynb`) | ~0.87 | Complete (Tests Passing) |
| **Phase 5** (Full Ensemble, Calibration, Infer Final `infer_final.ipynb`) | ~0.88–0.89+ | Complete (Tests Passing) |
| **Current User Top Submission** (`rsna-knee-tri-specialist-superstack`) | 0.945 | Submitted & Verified |
| **Top 2% target** | ~0.87+ (estimate) | Achieved (0.945 PB) |
