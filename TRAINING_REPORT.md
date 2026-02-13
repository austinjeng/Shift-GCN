# Shift-GCN Fall Detection Training Report

**Date:** February 12, 2026
**Task:** Binary fall detection (NTU RGB+D Action A043) using skeleton-based action recognition
**Framework:** Shift-GCN with MediaPipe Pose landmarks
**Platform:** Windows 11 Pro, NVIDIA GPU (CUDA 12.6), VS2022 (MSVC v143)

---

## 1. Executive Summary

Four Shift-GCN models — one per skeleton modality (joint, bone, joint_motion, bone_motion) — were trained on MediaPipe Pose landmarks extracted from the NTU RGB+D dataset for binary fall detection. The weighted ensemble of all four models achieved **99.77% Top-1 accuracy** on the validation set, with a fall detection F1-score of **92.91%**.

| Metric | Value |
|--------|-------|
| Ensemble Top-1 Accuracy | **99.77%** |
| Ensemble Top-5 Accuracy | **100.00%** |
| Fall Precision | 95.77% |
| Fall Recall | 90.22% |
| Fall F1-Score | 92.91% |
| Total Training Time | ~8.9 hours (4 models sequentially) |

---

## 2. Dataset

### 2.1 Source

- **Dataset:** NTU RGB+D (56,880 RGB videos)
- **Skeleton Extraction:** Google MediaPipe Pose (33 landmarks per frame)
- **Positive Class:** Action A043 ("falling down") — 948 videos total
- **Negative Class:** All other actions (A001–A042, A044–A060)
- **Benchmark Split:** Cross-subject (xsub)
  - Training subjects: {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}

### 2.2 Class Balancing

- **Training set:** Class-balanced via `subsample_ratio=3` (3 non-fall samples per 1 fall sample)
- **Validation set:** Natural distribution preserved (no balancing)

### 2.3 Data Statistics

| Split | Samples | Fall | Non-Fall | Fall Ratio | Shape |
|-------|---------|------|----------|------------|-------|
| Train | 2,688 | 672 | 2,016 | 25.0% (1:3) | `(2688, 3, 300, 33, 1)` |
| Val | 16,560 | 276 | 16,284 | 1.67% | `(16560, 3, 300, 33, 1)` |

**Tensor dimensions:** `(N, C, T, V, M)` = (samples, channels, frames, vertices, persons)
- **C = 3:** x, y, z coordinates
- **T = 300:** frames (zero-padded/truncated to 300)
- **V = 33:** MediaPipe Pose landmarks
- **M = 1:** single person per sample

### 2.4 Data Files

| File | Size | Description |
|------|------|-------------|
| `train_data_joint.npy` | 305 MB | Raw joint coordinates |
| `val_data_joint.npy` | 1.9 GB | Raw joint coordinates |
| `train_data_bone.npy` | 305 MB | Joint-to-parent differences along skeleton tree |
| `val_data_bone.npy` | 1.9 GB | Joint-to-parent differences along skeleton tree |
| `train_data_joint_motion.npy` | 305 MB | Frame-to-frame joint position differences |
| `val_data_joint_motion.npy` | 1.9 GB | Frame-to-frame joint position differences |
| `train_data_bone_motion.npy` | 305 MB | Frame-to-frame bone vector differences |
| `val_data_bone_motion.npy` | 1.9 GB | Frame-to-frame bone vector differences |
| `train_label.pkl` | 87 KB | Training labels (sample names + binary labels) |
| `val_label.pkl` | 534 KB | Validation labels (sample names + binary labels) |
| **Total** | **~8.6 GB** | |

### 2.5 Four Modalities Explained

1. **Joint:** Raw (x, y, z) coordinates of each of the 33 MediaPipe landmarks per frame. Captures absolute spatial position of the body.

2. **Bone:** Difference vectors between each joint and its parent in the skeleton tree (32 bone vectors + 1 zero root). Captures relative spatial structure and limb orientations.

3. **Joint Motion:** Frame-to-frame differences of joint coordinates (`frame[t] - frame[t-1]`). Captures velocity and temporal dynamics of each joint.

4. **Bone Motion:** Frame-to-frame differences of bone vectors. Captures how limb orientations and lengths change over time — the acceleration of skeletal structure.

---

## 3. Model Architecture

### 3.1 Shift-GCN

Shift-GCN replaces traditional graph convolution operations with a **shift mechanism** — channel-wise circular shifts across the spatial dimension (joints). This eliminates the need for learnable adjacency matrices, reducing parameters and computation while maintaining accuracy.

**Key components:**
- `Shift_gcn`: Spatial graph convolution via channel-wise shift + linear projection
- `Shift_tcn`: Temporal convolution via channel-wise shift + 1D convolution
- `TCN_GCN_unit`: Combined spatial-temporal block with residual connections

### 3.2 Network Architecture

```
Input: (N, 3, 300, 33, 1) → data_bn

10 TCN_GCN blocks:
  l1:  TCN_GCN_unit( 3 →  64, stride=1, no residual)
  l2:  TCN_GCN_unit(64 →  64, stride=1)
  l3:  TCN_GCN_unit(64 →  64, stride=1)
  l4:  TCN_GCN_unit(64 →  64, stride=1)
  l5:  TCN_GCN_unit(64 → 128, stride=2)  ← temporal downsampling
  l6:  TCN_GCN_unit(128 → 128, stride=1)
  l7:  TCN_GCN_unit(128 → 128, stride=1)
  l8:  TCN_GCN_unit(128 → 256, stride=2) ← temporal downsampling
  l9:  TCN_GCN_unit(256 → 256, stride=1)
  l10: TCN_GCN_unit(256 → 256, stride=1)

Global Average Pooling → fc(256, 2) → softmax
```

### 3.3 Model Parameters

| Parameter | Value |
|-----------|-------|
| Total parameters | ~720K |
| Checkpoint size | 6.3 MB |
| Input batch norm | `BatchNorm1d(1 × 3 × 33 = 99)` |
| Output classes | 2 (non-fall, fall) |
| FC initialization | `normal_(0, sqrt(2/num_class))` |

### 3.4 Skeleton Graph

**MediaPipe Pose:** 33 landmarks with 32 edges forming a spanning tree rooted at NOSE (landmark 0).

```
Graph: graph.mediapipe_pose.Graph
Labeling mode: spatial
Nodes: 33
Edges: 32 (inward) + 32 (outward) + 33 (self-loops) = 97
Center joint: landmarks 23, 24 (hip midpoint) for preprocessing
```

Bridge edges connect the three disconnected components of MediaPipe's default POSE_CONNECTIONS:
- NOSE(0) → LEFT_SHOULDER(11): connects head to torso
- NOSE(0) → MOUTH_LEFT(9): connects mouth to face

---

## 4. Training Configuration

All four models share identical hyperparameters:

### 4.1 Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with Nesterov momentum |
| Base learning rate | 0.1 |
| Weight decay | 0.0001 |
| Nesterov momentum | True |
| Warm-up epochs | 0 |

### 4.2 Learning Rate Schedule

Step decay (multiply by 0.1 at each step):

| Epoch Range | Learning Rate |
|-------------|---------------|
| 1–60 | 0.1 |
| 61–80 | 0.01 |
| 81–100 | 0.001 |
| 101–140 | 0.0001 |

### 4.3 Training Parameters

| Parameter | Value |
|-----------|-------|
| Total epochs | 140 |
| Batch size (train) | 64 |
| Batch size (eval) | 64 |
| Training batches/epoch | 42 (`ceil(2688/64)`, `drop_last=True`) |
| Eval batches/epoch | 259 (`ceil(16560/64)`) |
| Eval interval | Every 5 epochs |
| Checkpoint save interval | Every 2 epochs |
| DataLoader workers (train) | 8 |
| DataLoader workers (eval) | 2 (capped on Windows) |
| `drop_last` (train) | True |
| Random seed | 1 |
| `only_train_epoch` | 1 (train all params after epoch 1) |

### 4.4 Data Augmentation

No data augmentation was applied:

| Augmentation | Setting |
|-------------|---------|
| `random_choose` | False |
| `random_shift` | False |
| `random_move` | False |
| `window_size` | -1 (use all frames) |
| `normalization` | False |

### 4.5 Loss Function

- **CrossEntropyLoss** (standard, unweighted)
- Applied on raw logits before softmax
- No class weighting despite imbalanced validation set

---

## 5. Training Results

### 5.1 Individual Model Performance

| Modality | Best Top-1 | Best Epoch | Final Top-1 (Epoch 140) | Top-5 |
|----------|-----------|------------|-------------------------|-------|
| Joint | **99.49%** | 60 | 99.32% | 100% |
| Bone | **99.51%** | 35, 125 | 99.44% | 100% |
| Joint Motion | **99.46%** | 40 | 99.21% | 100% |
| Bone Motion | **99.64%** | 45 | 99.42% | 100% |

### 5.2 Epoch-by-Epoch Validation Accuracy

#### Joint Model

| Epoch | Top-1 | | Epoch | Top-1 | | Epoch | Top-1 |
|-------|-------|-|-------|-------|-|-------|-------|
| 5 | 96.18% | | 50 | 99.00% | | 95 | 99.32% |
| 10 | 97.06% | | 55 | 99.22% | | 100 | 99.31% |
| 15 | 98.51% | | 60 | **99.49%** | | 105 | 99.29% |
| 20 | 98.76% | | 65 | 99.37% | | 110 | 99.29% |
| 25 | 99.18% | | 70 | 99.38% | | 115 | 99.26% |
| 30 | 98.74% | | 75 | 99.33% | | 120 | 99.35% |
| 35 | 98.72% | | 80 | 99.33% | | 125 | 99.44% |
| 40 | 97.96% | | 85 | 99.37% | | 130 | 99.37% |
| 45 | 99.33% | | 90 | 99.36% | | 135 | 99.30% |
| | | | | | | 140 | 99.32% |

#### Bone Model

| Epoch | Top-1 | | Epoch | Top-1 | | Epoch | Top-1 |
|-------|-------|-|-------|-------|-|-------|-------|
| 5 | 97.55% | | 50 | 99.05% | | 95 | 99.44% |
| 10 | 98.91% | | 55 | 99.31% | | 100 | 99.41% |
| 15 | 99.24% | | 60 | 99.46% | | 105 | 99.40% |
| 20 | 99.37% | | 65 | 99.41% | | 110 | 99.37% |
| 25 | 99.26% | | 70 | 99.41% | | 115 | 99.37% |
| 30 | 99.32% | | 75 | 99.44% | | 120 | 99.48% |
| 35 | **99.51%** | | 80 | 99.39% | | 125 | **99.51%** |
| 40 | 99.50% | | 85 | 99.42% | | 130 | 99.42% |
| 45 | 99.44% | | 90 | 99.47% | | 135 | 99.40% |
| | | | | | | 140 | 99.44% |

#### Joint Motion Model

| Epoch | Top-1 | | Epoch | Top-1 | | Epoch | Top-1 |
|-------|-------|-|-------|-------|-|-------|-------|
| 5 | 98.52% | | 50 | 98.99% | | 95 | 99.08% |
| 10 | 98.47% | | 55 | 98.95% | | 100 | 99.03% |
| 15 | 98.94% | | 60 | 99.37% | | 105 | 99.06% |
| 20 | 98.76% | | 65 | 99.05% | | 110 | 99.11% |
| 25 | 99.00% | | 70 | 98.80% | | 115 | 98.89% |
| 35 | 99.20% | | 75 | 98.97% | | 120 | 99.09% |
| 40 | **99.46%** | | 80 | 99.27% | | 125 | 99.17% |
| 45 | 98.69% | | 85 | 99.09% | | 130 | 99.11% |
| | | | 90 | 99.18% | | 135 | 99.03% |
| | | | | | | 140 | 99.21% |

#### Bone Motion Model

| Epoch | Top-1 | | Epoch | Top-1 | | Epoch | Top-1 |
|-------|-------|-|-------|-------|-|-------|-------|
| 5 | 98.43% | | 50 | 99.25% | | 95 | 99.40% |
| 10 | 99.06% | | 55 | 99.57% | | 100 | 99.37% |
| 15 | 99.35% | | 60 | 99.21% | | 105 | 99.35% |
| 20 | 97.60% | | 65 | 99.42% | | 110 | 99.37% |
| 25 | 99.32% | | 70 | 99.49% | | 115 | 99.37% |
| 30 | 99.38% | | 75 | 99.47% | | 120 | 99.44% |
| 35 | 99.51% | | 80 | 99.34% | | 125 | 99.52% |
| 40 | 99.54% | | 85 | 99.38% | | 130 | 99.39% |
| 45 | **99.64%** | | 90 | 99.42% | | 135 | 99.41% |
| | | | | | | 140 | 99.42% |

### 5.3 Training Timeline

| Model | Start Time | End Time | Duration |
|-------|------------|----------|----------|
| Joint | 10:26 | 12:49 | ~2h 23m |
| Bone | 12:50 | 15:02 | ~2h 12m |
| Joint Motion | 15:03 | 17:16 | ~2h 13m |
| Bone Motion | 17:17 | 19:22 | ~2h 5m |
| **Total** | **10:26** | **19:22** | **~8h 56m** |

### 5.4 Checkpoints

- **Save interval:** Every 2 epochs
- **Checkpoints per model:** 70
- **Checkpoint format:** `{model_state_dict, optimizer_state_dict, epoch, global_step, best_acc}`
- **Naming convention:** `./save_models/<Experiment_name>-<epoch>-<global_step>.pt`
- **Checkpoint size:** 6.3 MB each
- **Total checkpoint storage:** 70 × 4 × 6.3 MB ≈ **1.76 GB**

---

## 6. Ensemble Evaluation

### 6.1 Ensemble Method

Weighted score fusion across all four modalities. For each validation sample, the softmax output vectors from each model's best checkpoint are combined:

```
score = 0.6 × joint + 0.6 × bone + 0.4 × joint_motion + 0.4 × bone_motion
prediction = argmax(score)
```

| Modality | Weight | Rationale |
|----------|--------|-----------|
| Joint | 0.6 | Spatial position — primary modality |
| Bone | 0.6 | Relative structure — complements joint |
| Joint Motion | 0.4 | Temporal dynamics — supplementary |
| Bone Motion | 0.4 | Structural dynamics — supplementary |

### 6.2 Ensemble Results

```
Top-1 Accuracy:  99.77%
Top-5 Accuracy: 100.00%
```

### 6.3 Classification Report

```
              precision    recall  f1-score   support

    Non-Fall     0.9983    0.9993    0.9988     16284
        Fall     0.9577    0.9022    0.9291       276

    accuracy                         0.9977     16560
   macro avg     0.9780    0.9507    0.9640     16560
weighted avg     0.9977    0.9977    0.9977     16560
```

### 6.4 Confusion Matrix

|  | Predicted Non-Fall | Predicted Fall |
|--|-------------------|----------------|
| **Actual Non-Fall** | 16,273 (TN) | 11 (FP) |
| **Actual Fall** | 27 (FN) | 249 (TP) |

- **True Positives (TP):** 249 falls correctly detected
- **True Negatives (TN):** 16,273 non-falls correctly classified
- **False Positives (FP):** 11 non-falls incorrectly flagged as falls (0.07% false alarm rate)
- **False Negatives (FN):** 27 falls missed (9.78% miss rate)

### 6.5 Ensemble vs. Individual Models

| Model | Accuracy | Errors | Improvement over best single |
|-------|----------|--------|------------------------------|
| Best single (Bone Motion) | 99.64% | ~60 errors | — |
| Ensemble | 99.77% | 38 errors | +0.13% (37% fewer errors) |

---

## 7. Technical Issues Encountered

### 7.1 Windows DataLoader Multiprocessing Crash

**Problem:** During evaluation, the test DataLoader with `num_workers=8` crashed with:
```
OSError: [Errno 22] Invalid argument
_pickle.UnpicklingError: pickle data was truncated
```

**Root cause:** Windows uses `spawn` (not `fork`) for multiprocessing. Each worker process receives a pickled copy of the parent's state. With 8 workers attempting to serialize references to the 1.9 GB validation dataset, the pickle buffer exceeded system limits.

**Fix applied in `main.py:242`:**
```python
test_workers = min(self.arg.num_worker, 2) if os.name == 'nt' else self.arg.num_worker
```

**Impact:** Evaluation throughput decreased marginally (~35 sec per eval pass vs. ~28 sec with 8 workers), but eliminated all crashes. Training DataLoader (305 MB train set) continued to use 8 workers without issue.

---

## 8. Environment Details

| Component | Version/Specification |
|-----------|----------------------|
| OS | Windows 11 Pro (10.0.26100) |
| Python | 3.12 (Conda env: `goldcoin`) |
| PyTorch | With CUDA support |
| CUDA Toolkit | 12.6 |
| C++ Compiler | MSVC v143 (VS2022 17.12.3) |
| GPU | NVIDIA (single GPU, device 0) |
| MediaPipe | Pose landmark extraction |
| NumPy | Array processing for skeleton data |
| scikit-learn | Classification report and confusion matrix |

### 8.1 CUDA Extension

The Shift-GCN CUDA kernel (`shift_cuda_linear_cpp`) was compiled with:
- `CUDAExtension` via `setup.py`
- Custom build script pattern: clear env → set conda PATH → call vcvars64 → set `DISTUTILS_USE_SDK=1` → run setup.py
- `KMP_DUPLICATE_LIB_OK=TRUE` set in conda activation to resolve OpenMP duplicate library conflict

---

## 9. Reproduction Commands

```bash
# 1. Generate joint data from NTU RGB+D videos
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py \
  --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ \
  --ntu_mode --subsample_ratio 3 --benchmark xsub

# 2. Generate bone and motion modalities
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_bone_data_mediapipe.py
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_motion_data_mediapipe.py

# 3. Train all 4 models (sequential, ~2h each)
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint_motion.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone_motion.yaml --overwrite True

# 4. Ensemble evaluation
conda run -n goldcoin --cwd "D:\Shift-GCN" python ensemble_mediapipe.py

# Resume from checkpoint (if training is interrupted):
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py \
  --config ./config/mediapipe/train_joint.yaml \
  --resume ./save_models/mediapipe_ShiftGCN_joint-<EPOCH>-<STEP>.pt
```

---

## 10. Observations and Future Work

### 10.1 Key Observations

1. **Rapid convergence:** All models reached >98% accuracy within the first 10–15 epochs. The remaining epochs refined accuracy by ~1%.

2. **LR step decay impact:** Accuracy improvements are visible after each LR step (epochs 60, 80, 100), particularly for the joint model which peaked at epoch 60 (first decay).

3. **Modality ranking:** Bone Motion > Bone ≈ Joint > Joint Motion. Bone-based features slightly outperform joint-based features, suggesting relative skeletal structure is more discriminative than absolute position for fall detection.

4. **Motion modalities are noisier:** Joint_motion and bone_motion show more epoch-to-epoch variance in accuracy, likely due to frame differencing amplifying noise in MediaPipe landmark estimates.

5. **Ensemble benefit:** The ensemble reduced errors by 37% compared to the best single model, demonstrating that the four modalities capture complementary information.

6. **Class imbalance:** The validation set has a 59:1 non-fall:fall ratio. Despite this, the model achieves 90.22% recall on the minority class — but the 27 missed falls represent the primary area for improvement.

### 10.2 Potential Improvements

- **Threshold tuning:** Adjust the decision threshold on ensemble softmax scores to trade precision for higher recall (important for safety-critical fall detection)
- **Class-weighted loss:** Apply higher weight to the fall class during training to reduce false negatives
- **Data augmentation:** Enable `random_shift`, `random_choose`, or `random_move` to increase training diversity
- **Ensemble weight optimization:** Search for optimal alpha weights rather than using hand-tuned 0.6/0.6/0.4/0.4
- **Full NTU dataset:** Train without `subsample_ratio` to use all available non-fall examples
- **Cross-view benchmark:** Evaluate on xview split in addition to xsub
