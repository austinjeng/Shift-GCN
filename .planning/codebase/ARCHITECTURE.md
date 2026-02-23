# Architecture

**Analysis Date:** 2025-02-24

## Pattern Overview

**Overall:** Skeleton-based action recognition using Shift Graph Convolution Networks (GCN)

**Key Characteristics:**
- Shift mechanism replaces traditional spatial GCN — does NOT use adjacency matrix `A` for convolution
- Graph topology used only for structural information in `data_bn` preprocessing and consistency
- Multi-modality ensemble: supports joint, bone, motion, and bone-motion representations
- Parameterizable skeleton formats: NTU RGB-D (25 joints) and MediaPipe Pose (33 landmarks)
- Two-stream temporal-spatial processing: alternating shift operations and temporal convolutions

## Layers

**Input Processing (Data Layer):**
- Purpose: Load skeleton data, apply batch normalization across spatial and person dimensions
- Location: `main.py:231-251` (DataLoader), `feeders/feeder.py` (Feeder class)
- Contains: Data loading, normalization, augmentation (random shift, motion, etc.)
- Depends on: Numpy .npy files, pickle label files
- Used by: Model training/evaluation loops

**Graph Topology Layer:**
- Purpose: Define skeleton structure (vertex connectivity) and compute spatial adjacency matrices
- Location: `graph/ntu_rgb_d.py` (25 joints), `graph/mediapipe_pose.py` (33 landmarks)
- Contains: Joint definitions, parent-child links, adjacency matrix computation
- Depends on: `graph/tools.py` for graph operations
- Used by: Model initialization to configure `num_point` parameter

**Feature Shift Layer (Shift_gcn):**
- Purpose: Apply learned shift patterns to feature maps for spatial-channel interaction
- Location: `model/shift_gcn.py:77-142` (Shift_gcn class)
- Contains: Feature masking (`Feature_Mask`), linear weight transformation (`Linear_weight`), index-based shifting
- Depends on: `shift_in`, `shift_out` index arrays (spatial and channel shifts)
- Used by: TCN_GCN_unit for spatial feature extraction

**Temporal Shift Layer (Shift_tcn):**
- Purpose: Apply temporal shifting across frames with learnable transformation
- Location: `model/shift_gcn.py:48-74` (Shift_tcn class)
- Contains: Input/output shifts via CUDA kernel (`Shift` from `cuda/shift.py`), temporal linear transform
- Depends on: CUDA shift operation (requires NVIDIA GPU + CUDA toolkit)
- Used by: TCN_GCN_unit for temporal feature extraction

**Residual Unit Layer (TCN_GCN_unit):**
- Purpose: Combine spatial (GCN) and temporal (TCN) feature extraction with residual connections
- Location: `model/shift_gcn.py:145-162` (TCN_GCN_unit class)
- Contains: Shift_gcn → Shift_tcn pipeline with residual bypass
- Depends on: Shift_gcn, Shift_tcn
- Used by: Model stacked layers (l1-l10)

**Classification Head:**
- Purpose: Pool temporal-spatial features and predict action class
- Location: `model/shift_gcn.py:189-216` (Model.forward, lines 211-216)
- Contains: Global average pooling (temporal and person dims), fully connected layer
- Depends on: Output from l10 (256 channels)
- Used by: Training loss computation and inference

## Data Flow

**Training Forward Pass:**

1. Input: `(N, C, T, V, M)` = (batch, channels=3, frames, joints, persons)
2. `data_bn`: Flatten to `(N, M*V*C, T)`, apply batch normalization across spatial dimensions
3. Reshape back to `(N*M, C, T, V)` (separate each person-skeleton as batch item)
4. **L1-L10**: 10 stacked TCN_GCN_units with increasing channels: 3→64→64→64→64→128→128→128→256→256
5. Each unit applies: `Shift_gcn` (spatial) → `Shift_tcn` (temporal) + residual connection
6. Stride-2 downsampling at L5 (64→128) and L8 (128→256) reduces temporal resolution
7. Global pooling: average over time (T) and person (M) dimensions → `(N, 256)`
8. Fully connected: 256 → num_class (typically 2 for fall detection)
9. Output: `(N, num_class)` logits

**Data Preprocessing Pipeline (mediapipe_gendata.py):**

1. Extract MediaPipe landmarks from video → `(C=3, T, V=33, M=1)` coordinate arrays
2. Pad null frames with previous valid frames (`preprocess.py`)
3. Center data to hip midpoint (joints [23,24] for MediaPipe, joint 1 for NTU)
4. Rotate skeleton to align z-axis (bottom-top spine alignment)
5. Rotate skeleton to align x-axis (right-left shoulder alignment)
6. Flatten to `(N, C, T, V, M)` and save as .npy files
7. Generate derived modalities:
   - **Bone data**: Joint-to-parent differences along spanning tree (`gen_bone_data_mediapipe.py`)
   - **Motion data**: Frame differencing (current - previous frame, `gen_motion_data_mediapipe.py`)
   - **Bone-motion data**: Bone data frame differencing

**Inference & Ensemble:**

1. Single-model inference: Load best checkpoint, run forward pass, store output logits
2. 4-modality ensemble (`ensemble_mediapipe.py`):
   - Weighted average: 0.6×joint + 0.6×bone + 0.4×joint_motion + 0.4×bone_motion
   - Argmax to get final prediction
   - Compute accuracy, F1, confusion matrix vs. validation labels

## Key Abstractions

**Shift Mechanism:**
- Purpose: Learned spatial-temporal feature shifting without explicit adjacency matrix operations
- Examples: `Shift_gcn.shift_in`, `Shift_gcn.shift_out` (index permutation arrays)
- Pattern: Pre-compute shift indices as cyclic permutations of feature dimensions; apply via `torch.index_select` during forward pass
- Location: `model/shift_gcn.py:108-118`

**Feature Mask:**
- Purpose: Learnable per-joint per-channel gating applied before linear transformation
- Examples: `Feature_Mask` parameter in `Shift_gcn:96`
- Pattern: `tanh(Feature_Mask)+1` produces values in [0,2]; element-wise multiply with features
- Location: `model/shift_gcn.py:129`

**Graph Representation:**
- Purpose: Define skeleton topology (joints and connectivity rules)
- Examples: `graph.ntu_rgb_d.Graph`, `graph.mediapipe_pose.Graph`
- Pattern: Static adjacency matrix `A` (3, V, V) with self-loops, inward, outward edges; computed once at model init, never modified
- Location: `graph/ntu_rgb_d.py`, `graph/mediapipe_pose.py`

**Data Augmentation:**
- Purpose: Increase training robustness via random transformations
- Examples: `random_shift`, `random_choose`, `random_move` in `feeders/tools.py`
- Pattern: Apply during training if flags set in config; used during `__getitem__` in Feeder
- Location: `feeders/tools.py:32-101`

## Entry Points

**Training Entry Point:**
- Location: `main.py:566-584` (main script, lines 567-584)
- Triggers: `python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True`
- Responsibilities:
  1. Parse YAML config to get model args, data paths, hyperparams
  2. Instantiate `Processor` (training orchestrator)
  3. Call `processor.start()` → loop over epochs, train, eval, save checkpoints

**Training Loop:**
- Location: `main.py:377-449` (Processor.train method)
- Triggers: Called once per epoch from `Processor.start()`
- Responsibilities:
  1. Load batches from training DataLoader
  2. Forward pass: `model(data)` → logits
  3. Compute loss, backward, optimizer.step()
  4. Save checkpoint dict (state_dict, optimizer, epoch, global_step, best_acc) every N epochs

**Evaluation Loop:**
- Location: `main.py:450-515` (Processor.eval method)
- Triggers: Called every N epochs or after final epoch (configurable `eval_interval`)
- Responsibilities:
  1. Load batches from validation DataLoader
  2. Forward pass with `torch.no_grad()`
  3. Accumulate logits, compute top-k accuracy
  4. Save best logits to `.pkl` if accuracy improves
  5. Log metrics to file

**Data Generation Entry Points:**
- `mediapipe_gendata.py`: Extract landmarks from NTU videos, apply class balancing (subsample non-fall)
- `gen_bone_data_mediapipe.py`: Compute bone differences from joint data
- `gen_motion_data_mediapipe.py`: Compute frame differences for motion representation

**Ensemble Entry Point:**
- Location: `ensemble_mediapipe.py:1-51`
- Triggers: `python ensemble_mediapipe.py`
- Responsibilities: Load 4 best model outputs, compute weighted average, report top-1/top-5 accuracy and F1

## Error Handling

**Strategy:** Mixed approach — try-except for data loading, assertions for shape mismatches, early returns for missing files

**Patterns:**
- `feeders/feeder.py:44-50`: Try standard pickle loading, fallback to Python2 pickle encoding
- `data_gen/mediapipe_gendata.py:54-56`: Check if video file can be opened before processing
- `main.py:283-292`: Warn on missing weights, load partial state dict if shapes mismatch
- `main.py:215-229`: Handle legacy checkpoint format (bare state_dict) vs. new dict format with optimizer state

## Cross-Cutting Concerns

**Logging:**
- Approach: Timestamp-prefixed to stdout and `.../work_dir/{exp_name}/log.txt`
- Implementation: `Processor.print_log()` in `main.py:359-366` writes to file with `exist_ok=True`

**Validation:**
- Approach: Input shape assertions in forward methods; label range checks in data loading
- Examples: `Shift_gcn.forward:122` unpacks `(n, c, t, v)` from input

**Determinism & Reproducibility:**
- Approach: Fixed random seed via `init_seed()` in `main.py:24-31`
- Covers: torch, numpy, random, cudnn (deterministic=True, benchmark=False)

**Device Management:**
- Approach: CUDA device specified in config; model and loss moved to device in `load_model()` and `load_optimizer()`
- Pattern: Single GPU (device_ids=[0]) or multi-GPU DataParallel (device_ids=[0,1,...])
- Hardcoded: `Shift_gcn.__init__` initializes weight tensors with `device='cuda'` (line 90, 93, 96)

**Checkpoint Serialization:**
- Format: Dictionary with keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `global_step`, `best_acc`
- Location: `main.py:441-448` (train), `main.py:215-223` (resume)
- Auto-detection: Distinguishes new dict format from legacy bare state_dict on load

---

*Architecture analysis: 2025-02-24*
