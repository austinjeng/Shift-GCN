# External Integrations

**Analysis Date:** 2025-02-24

## APIs & External Services

**MediaPipe Pose:**
- Service: Google MediaPipe Pose landmark detection
- What it's used for: Extract 33-joint skeleton landmarks from video frames
  - Implementation: `data_gen/mediapipe_gendata.py` lines 52-60
  - Uses: `mp.solutions.pose.Pose` for real-time multi-person pose estimation
  - Processes: NTU RGB+D video files to generate training/validation skeleton data
  - Output format: (N, 3, T, V=33, M=1) numpy arrays where C=coordinates (x,y,confidence)

**OpenCV (cv2):**
- Service: OpenCV computer vision library
- What it's used for: Video file reading and frame decoding
  - Implementation: `data_gen/mediapipe_gendata.py` line 53
  - SDK/Client: `cv2.VideoCapture()` for frame-by-frame video processing
  - Used in data generation pipeline to read NTU RGB+D .avi video files

## Data Storage

**Local Filesystem:**
- All data stored locally, no cloud services
- Locations:
  - Raw video input: User-specified via `--video_dir` (e.g., `E:\nturgb+d_rgb`)
  - Processed data: `data/mediapipe/` directory
    - `train_data_joint.npy`, `val_data_joint.npy` - Skeleton coordinates
    - `train_data_bone.npy`, `val_data_bone.npy` - Joint differences (spanning tree)
    - `train_data_joint_motion.npy`, `val_data_joint_motion.npy` - Frame differencing
    - `train_data_bone_motion.npy`, `val_data_bone_motion.npy` - Bone motion
    - `train_label.pkl`, `val_label.pkl` - Labels (binary: 0=non-fall, 1=fall A043)
  - Model checkpoints: `save_models/` directory
    - Pattern: `mediapipe_ShiftGCN_{modality}-{epoch}-{global_step}.pt`
    - Format: Python pickle dict with keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `global_step`, `best_acc`
  - Training logs: `work_dir/mediapipe_ShiftGCN_{modality}/log.txt`
  - Evaluation results: `work_dir/mediapipe_ShiftGCN_{modality}/eval_results/best_acc.pkl`

**Data Format:**
- NumPy .npy files (binary format) - skeleton data tensors
- Python pickle .pkl files - labels and evaluation scores
- PyTorch .pt files - model checkpoints (dict serialization)

**File Storage:**
- Local filesystem only
- No S3, blob storage, or remote file systems

**Caching:**
- None detected
- Data loaded fresh each epoch via DataLoader
- Memory-mapped numpy arrays optional for large datasets (use_mmap=True in feeder)

## Authentication & Identity

**Auth Provider:**
- Not applicable - no authentication required
- All integrations are open-source libraries

**Custom Authentication:**
- Not used

## Monitoring & Observability

**Error Tracking:**
- Not detected
- No external error tracking service (Sentry, DataDog, etc.)

**Logging:**
- Custom file-based logging in `main.py`
  - Output: `work_dir/{experiment_name}/log.txt`
  - Console logging via `--print-log` flag
- Metrics tracked: training loss, accuracy, Top-1/Top-5, F1 score
- Checkpoints auto-saved at `--save-interval` (default: 2 epochs)

**Observability:**
- Manual checkpoint inspection
- Command-line argument `--show-topk` for metrics visualization
- Classification report and confusion matrix computed in `ensemble_mediapipe.py`

## CI/CD & Deployment

**Hosting:**
- Local machine development (Windows 11 Pro)
- No cloud deployment detected

**CI Pipeline:**
- Not detected
- Manual training: `conda run -n goldcoin python main.py --config ...`

**Model Artifacts:**
- Checkpoints stored in `save_models/` directory
- Evaluation results in `work_dir/mediapipe_ShiftGCN_{modality}/eval_results/`

## Environment Configuration

**Required env vars (Python/Conda):**
- Build-time only:
  - `DISTUTILS_USE_SDK=1` - For CUDA extension compilation on Windows
  - `KMP_DUPLICATE_LIB_OK=TRUE` - OpenMP duplicate lib fix in conda activation
  - `VSINSTALLDIR` - Must be cleared before calling vcvars64 to avoid path corruption

**Training-time:**
- No environment variables required for training
- All configuration via YAML files and CLI arguments

**Key YAML config parameters:**
- `base_lr` - Learning rate (typically 0.1)
- `step` - LR decay schedule steps (typically [60, 80, 100])
- `batch_size`, `test_batch_size` - Batch sizes (64 typical)
- `num_epoch` - Training epochs (140 typical)
- `num_worker` - DataLoader workers (8 for training, capped to 2 on Windows test)
- `eval_interval` - Validation frequency (5 epochs typical)

**Secrets location:**
- No secrets management
- No .env files, credential files, or sensitive data in codebase

## Webhooks & Callbacks

**Incoming:**
- None detected
- No external webhook endpoints

**Outgoing:**
- None detected
- No external service calls or callbacks

**Ensemble Callback:**
- `ensemble_mediapipe.py` loads pre-computed evaluation results from pickle files
- Weighted averaging: joint(0.6) + bone(0.6) + joint_motion(0.4) + bone_motion(0.4)
- Computes final binary classification predictions post-training

## External Datasets

**NTU RGB+D Dataset:**
- Source: External user-provided dataset (not included in repo)
- Format: Video files (SsssCcccPpppRrrrAaaa.avi naming convention)
- Processing: Extracted to MediaPipe landmarks via `data_gen/mediapipe_gendata.py`
- Benchmark: `xsub` (cross-subject) evaluation protocol
- Training subjects: 20 specific subject IDs
- Class distribution: A043 (fall) vs all others (binary classification)
- Preprocessing: Subsampling with `--subsample_ratio 3` for class balance (3:1 non-fall:fall)

## Summary of External Dependencies

| Category | Service | Version | Purpose |
|----------|---------|---------|---------|
| Pose Detection | MediaPipe Pose | (latest in goldcoin env) | Extract 33-joint landmarks from video |
| Video Reading | OpenCV (cv2) | (latest in goldcoin env) | Read NTU RGB+D .avi video files |
| ML Framework | PyTorch | 1.x/2.x | Neural network training |
| Data Proc | NumPy | (latest in goldcoin env) | Numerical operations |
| Metrics | scikit-learn | (latest in goldcoin env) | Compute F1, precision, recall, confusion matrix |
| Config | PyYAML | (latest in goldcoin env) | Parse .yaml training configs |
| Utils | tqdm | (latest in goldcoin env) | Progress bars |
| Build | CUDA 12.6 | 12.6 | GPU acceleration for shift kernel |

---

*Integration audit: 2025-02-24*
