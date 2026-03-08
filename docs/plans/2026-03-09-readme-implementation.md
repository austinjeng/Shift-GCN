# README Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite README.md from the outdated upstream Shift-GCN content to a professional MediaPipe fall detection project README.

**Architecture:** Single-file rewrite of `README.md` following the approved design in `docs/plans/2026-03-09-readme-redesign.md`. The README has 11 sections in linear walkthrough order. No code changes — documentation only.

**Tech Stack:** Markdown

**Reference files (read-only, for accuracy):**
- `TRAINING_REPORT.md` — results data, hyperparameters, confusion matrix
- `inference_pipeline.py:42-54` — actual CLI argument names and defaults
- `config/mediapipe/train_joint.yaml` — training config reference
- `CLAUDE.md` — CUDA build details, Windows quirks
- `data_gen/mediapipe_gendata.py` — data gen argument names

---

### Task 1: Header, Overview, and Pipeline Diagram

**Files:**
- Modify: `README.md` (full rewrite — replace all content)

**Step 1: Write the complete README.md**

Replace the entire contents of `README.md` with the new version. The full content follows below. All data has been verified against source files.

````markdown
# MediaPipe Shift-GCN: Skeleton-Based Fall Detection

A skeleton-based fall detection system built on [Shift-GCN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.pdf) (CVPR 2020). Uses [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (33 landmarks) instead of depth-sensor skeletons, enabling fall detection from ordinary RGB video.

Trains a 4-model ensemble (joint, bone, joint motion, bone motion) on [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and runs inference on arbitrary video files via CLI or Tkinter GUI.

```
                         ┌──────────────────┐
                         │   RGB Video      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  MediaPipe Pose   │
                         │  (33 landmarks)   │
                         └────────┬─────────┘
                                  │
              ┌───────────┬───────┴───────┬────────────┐
              ▼           ▼               ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌──────────────┐
         │  Joint  │ │  Bone   │ │Joint Motion │ │ Bone Motion  │
         └────┬────┘ └────┬────┘ └──────┬──────┘ └──────┬───────┘
              │           │             │               │
         ┌────▼────┐ ┌────▼────┐ ┌──────▼──────┐ ┌──────▼───────┐
         │Shift-GCN│ │Shift-GCN│ │  Shift-GCN  │ │  Shift-GCN   │
         │ (×0.6)  │ │ (×0.6)  │ │   (×0.4)    │ │   (×0.4)     │
         └────┬────┘ └────┬────┘ └──────┬──────┘ └──────┬───────┘
              │           │             │               │
              └───────────┴───────┬─────┴───────────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Weighted Ensemble │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Fall / Non-Fall  │
                         └──────────────────┘
```

## Results

Trained on NTU RGB+D (cross-subject split) with MediaPipe Pose landmarks.

| Model | Top-1 Accuracy |
|-------|---------------|
| Joint | 99.49% |
| Bone | 99.51% |
| Joint Motion | 99.46% |
| Bone Motion | 99.64% |
| **Ensemble** | **99.77%** |

**Confusion Matrix (Ensemble)**

|  | Predicted Non-Fall | Predicted Fall |
|--|-------------------|----------------|
| **Actual Non-Fall** | 16,273 | 11 |
| **Actual Fall** | 27 | 249 |

Fall F1-score: **92.91%** (precision 95.77%, recall 90.22%).

Dataset: 2,688 train samples (1:3 fall:non-fall ratio) / 16,560 validation samples (natural distribution). See [TRAINING_REPORT.md](TRAINING_REPORT.md) for epoch-by-epoch results and full analysis.

## Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- CUDA Toolkit 12.x (or compatible version)
- A C++ compiler (g++ on Linux, Visual Studio on Windows)
- MediaPipe, OpenCV, NumPy, scikit-learn

### Environment Setup

```bash
conda create -n shiftgcn python=3.12
conda activate shiftgcn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python numpy scikit-learn
```

Adjust the PyTorch index URL to match your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for options.

## Building the CUDA Extension

The shift operation uses a custom CUDA kernel that must be compiled before training or inference.

### Linux

```bash
cd model/Temporal_shift/cuda
python setup.py install
```

### Windows

Requires Visual Studio 2022 (or 2019) with the **"Desktop development with C++"** workload installed.

```bat
:: Clear stale environment, then activate VS build tools
set "VSINSTALLDIR="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Build the extension
set DISTUTILS_USE_SDK=1
cd model\Temporal_shift\cuda
python setup.py install
```

Add the following to your conda activation script to prevent an OpenMP conflict:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
```

### Verify the Build

```bash
python -c "from cuda.shift import Shift; print('CUDA extension OK')"
```

## Data Preparation

Preparing training data is a three-step process. All scripts are in `data_gen/`.

### Step 1: Extract MediaPipe Landmarks

Download the [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) RGB videos and extract 33-landmark skeletons:

```bash
python data_gen/mediapipe_gendata.py \
  --video_dir /path/to/nturgb+d_rgb \
  --out_dir ./data/mediapipe/ \
  --ntu_mode \
  --subsample_ratio 3 \
  --benchmark xsub
```

| Flag | Purpose |
|------|---------|
| `--ntu_mode` | Parse NTU filenames to determine subject/action splits |
| `--subsample_ratio 3` | Balance training set to 1:3 fall:non-fall ratio |
| `--benchmark xsub` | Use the cross-subject evaluation protocol |

This produces `train_data_joint.npy` (305 MB), `val_data_joint.npy` (1.9 GB), and label files.

### Step 2: Generate Bone Data

```bash
python data_gen/gen_bone_data_mediapipe.py
```

Computes joint-to-parent difference vectors along the skeleton tree (32 bones).

### Step 3: Generate Motion Data

```bash
python data_gen/gen_motion_data_mediapipe.py
```

Computes frame-to-frame differences for both joint and bone modalities.

After all three steps, `./data/mediapipe/` contains 8 `.npy` data files + 2 `.pkl` label files (~8.6 GB total).

## Training

Train four models — one per modality. Each takes approximately 2 hours on a single GPU.

```bash
python main.py --config ./config/mediapipe/train_joint.yaml
python main.py --config ./config/mediapipe/train_bone.yaml
python main.py --config ./config/mediapipe/train_joint_motion.yaml
python main.py --config ./config/mediapipe/train_bone_motion.yaml
```

All four configs share identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Epochs | 140 |
| Batch size | 64 |
| Optimizer | SGD + Nesterov momentum |
| Learning rate | 0.1 (step decay ×0.1 at epochs 60, 80, 100) |
| Weight decay | 0.0001 |

Checkpoints are saved every 2 epochs to `./save_models/`.

### Resuming from a Checkpoint

If training is interrupted, resume from the latest checkpoint:

```bash
python main.py --config ./config/mediapipe/train_joint.yaml \
  --resume ./save_models/mediapipe_ShiftGCN_joint-60-2520.pt
```

### Overwriting a Previous Run

To clean old checkpoints and start fresh:

```bash
python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True
```

## Ensemble Evaluation

After training all four models, evaluate the weighted ensemble:

```bash
python ensemble_mediapipe.py
```

This combines predictions from the best checkpoint of each modality with weights `[0.6, 0.6, 0.4, 0.4]` (joint, bone, joint motion, bone motion).

## Inference

Run fall detection on any RGB video. The pipeline extracts MediaPipe landmarks, runs the 4-model ensemble over sliding windows, and produces an annotated output video with a JSON report.

### CLI Mode

```bash
python inference_pipeline.py --cli --video /path/to/video.mp4
```

### GUI Mode

```bash
python inference_pipeline.py
```

Opens a Tkinter window for selecting video files and viewing results.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Input video path (required in CLI mode) |
| `--cli` | `False` | Run in CLI mode without GUI |
| `--window_size` | `300` | Frames per sliding window |
| `--stride` | `150` | Window step size (frames) |
| `--threshold` | `0.5` | Fall confidence threshold |
| `--output_dir` | `./inference_output` | Directory for output files |
| `--ensemble_weights` | `0.6,0.6,0.4,0.4` | Comma-separated model weights |
| `--save_dir` | `./save_models` | Directory containing trained checkpoints |

### Output

- **`results.json`** — Per-window scores, per-frame aggregated confidence, and detected fall intervals.
- **Annotated video** — Skeleton overlay with a real-time confidence bar.

Checkpoints are auto-detected from `--save_dir` (selects the highest-epoch checkpoint per modality).

## Architecture

Shift-GCN replaces traditional graph convolutions with channel-wise circular shifts across joints, eliminating learnable adjacency matrices. This yields a lightweight model (~720K parameters) with strong accuracy.

**Input shape:** `(N, C, T, V, M)` = (batch, 3 coordinates, 300 frames, 33 landmarks, 1 person)

```
data_bn → 10 TCN_GCN blocks → Global Average Pooling → FC(256, 2)

Block layout:
  Blocks 1–4:   64 channels, stride 1
  Block 5:     128 channels, stride 2  (temporal downsampling)
  Blocks 6–7:  128 channels, stride 1
  Block 8:     256 channels, stride 2  (temporal downsampling)
  Blocks 9–10: 256 channels, stride 1
```

Each `TCN_GCN_unit` applies a spatial shift convolution (`Shift_gcn`) followed by a temporal shift convolution (`Shift_tcn`) with a residual connection.

For full details, see the [original paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.pdf).

## Citation

If you use this work, please cite the original Shift-GCN paper:

```bibtex
@inproceedings{cheng2020shiftgcn,
  title     = {Skeleton-Based Action Recognition with Shift Graph Convolutional Network},
  author    = {Ke Cheng and Yifan Zhang and Xiangyu He and Weihan Chen and Jian Cheng and Hanqing Lu},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
}
```

## Acknowledgments

This project builds on the [Shift-GCN](https://github.com/kchengiva/Shift-GCN) codebase by Ke Cheng et al. The MediaPipe Pose adaptation, binary fall detection training pipeline, and video inference system were developed independently.
````

**Step 2: Review the rendered README**

Verify:
- Pipeline diagram renders correctly in a fixed-width font
- All links resolve (TRAINING_REPORT.md, paper PDF, NTU dataset, PyTorch, MediaPipe docs)
- CLI flags match `inference_pipeline.py:42-54` exactly
- Hyperparameters match `config/mediapipe/train_joint.yaml`
- Results match `TRAINING_REPORT.md` sections 5 and 6

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for MediaPipe fall detection project"
```

---

### Task 2: Clean Up Stale Assets

**Files:**
- Delete: `flops_accuracy.png` (upstream chart, no longer referenced)

**Step 1: Verify the image is no longer referenced**

```bash
grep -r "flops_accuracy" . --include="*.md" --include="*.py"
```

Expected: No matches (after Task 1 removes the old README reference).

**Step 2: Remove and commit**

```bash
git rm flops_accuracy.png
git commit -m "chore: remove upstream FLOPs chart (no longer referenced)"
```

---

### Task 3: Final Verification

**Step 1: Check for broken internal links**

Verify these files exist:
- `TRAINING_REPORT.md`
- `data_gen/mediapipe_gendata.py`
- `data_gen/gen_bone_data_mediapipe.py`
- `data_gen/gen_motion_data_mediapipe.py`
- `config/mediapipe/train_joint.yaml` (and bone, joint_motion, bone_motion variants)
- `ensemble_mediapipe.py`
- `inference_pipeline.py`
- `model/Temporal_shift/cuda/setup.py`

**Step 2: Read through the final README end-to-end**

Read `README.md` and check for:
- Consistent formatting
- No orphaned references to upstream content (NTU 60-class, original prerequisites)
- Accurate flag names and defaults
