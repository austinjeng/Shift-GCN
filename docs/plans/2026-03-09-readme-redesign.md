# README Redesign — Design Document

**Date:** 2026-03-09
**Goal:** Replace the outdated upstream Shift-GCN README with a professional README for the MediaPipe fall detection project.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Audience | Hybrid: academic + usable pipeline | Serve both researchers and practitioners |
| Relationship to upstream | Full replacement; cite paper only | This is now a distinct project |
| Inference pipeline | Featured capability (dedicated section) | Shows end-to-end value |
| Results detail | Moderate (individual + ensemble + confusion matrix) | Enough to prove quality; link to full report |
| Platform coverage | Both Windows and Linux equally | Windows tested; Linux instructions standard |
| Visuals | ASCII pipeline diagram | Shows data flow at a glance |
| Title | "MediaPipe Shift-GCN: Skeleton-Based Fall Detection" | Leads with differentiator |

## Structure (Approach 1: Linear Walkthrough)

### 1. Header & Overview
- Title: "MediaPipe Shift-GCN: Skeleton-Based Fall Detection"
- One-paragraph summary: what (fall detection), how (MediaPipe + Shift-GCN ensemble), why (RGB video, no depth sensor)
- ASCII pipeline diagram: Video → MediaPipe → 4 modalities → 4 models → ensemble → fall/non-fall

### 2. Results
- Table: 4 individual model accuracies + ensemble (99.77%)
- Confusion matrix (16,273 TN / 11 FP / 27 FN / 249 TP)
- Fall F1 92.91% (precision 95.77%, recall 90.22%)
- Dataset size summary (2,688 train / 16,560 val)
- Link to TRAINING_REPORT.md for epoch-by-epoch details

### 3. Prerequisites
- Python 3.10+, PyTorch (CUDA), CUDA Toolkit 12.x, MediaPipe, OpenCV, NumPy, scikit-learn
- Conda environment setup commands

### 4. Building the CUDA Extension
- Linux: 2 commands (cd + python setup.py install)
- Windows: VS2022 steps (clear VSINSTALLDIR → vcvars64 → DISTUTILS_USE_SDK=1 → setup.py)
- KMP_DUPLICATE_LIB_OK=TRUE for Windows
- Verification command

### 5. Data Preparation (3 steps)
- Step 1: Extract MediaPipe landmarks from NTU RGB+D videos (mediapipe_gendata.py)
  - Explain --ntu_mode, --subsample_ratio 3, --benchmark xsub
  - Output: joint .npy + label .pkl files
- Step 2: Generate bone data (gen_bone_data_mediapipe.py)
- Step 3: Generate motion data (gen_motion_data_mediapipe.py)
- Total output: ~8.6 GB across 8 .npy + 2 .pkl files

### 6. Training
- Four commands (one per modality config)
- Hyperparameter table (epochs, batch, optimizer, LR schedule, weight decay)
- Resume from checkpoint subsection
- Overwrite previous runs subsection
- ~2 hours per model on single GPU

### 7. Ensemble Evaluation
- Single command: python ensemble_mediapipe.py
- Weights: [0.6, 0.6, 0.4, 0.4]

### 8. Inference
- CLI mode: python inference_pipeline.py --cli --video path
- GUI mode: python inference_pipeline.py
- Options table (--video, --cli, --window_size, --stride, --threshold, --save_dir)
- Output: results.json + annotated video
- Auto-detection of checkpoints from save_models/

### 9. Architecture (brief)
- Shift mechanism explanation (1-2 sentences)
- ~720K parameters
- Input tensor shape annotated: (N, 3, 300, 33, 1)
- 10 TCN_GCN blocks with channel/stride progression
- Global average pooling → FC(256, 2) → Softmax

### 10. Citation
- Original Shift-GCN BibTeX entry (unchanged)

### 11. Acknowledgments
- Credit upstream Shift-GCN repo (Ke Cheng et al.)
- State that MediaPipe adaptation, fall detection pipeline, and inference system are independent work

## Out of Scope
- License section (upstream has none)
- Badges/shields
- Contributing guidelines
- The original NTU 60-class training instructions
- Epoch-by-epoch tables (stay in TRAINING_REPORT.md)
- The upstream flops_accuracy.png image
