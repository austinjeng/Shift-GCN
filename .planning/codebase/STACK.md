# Technology Stack

**Analysis Date:** 2025-02-24

## Languages

**Primary:**
- Python 3.x - Main training and data processing scripts in `main.py`, `data_gen/*.py`, `feeders/feeder.py`
- CUDA C/C++ - Custom shift operation kernel at `model/Temporal_shift/cuda/shift_cuda_kernel.cu` and wrapper `model/Temporal_shift/cuda/shift_cuda.cpp`

**Secondary:**
- NumPy - Numerical operations and array processing

## Runtime

**Environment:**
- Python 3.12 (inferred from cuda build outputs: `lib.win-amd64-cpython-312/`)
- NVIDIA CUDA 12.6 (as noted in CLAUDE.md)
- NVIDIA GPU with CUDA support required for training

**Package Manager:**
- Conda (uses `goldcoin` conda environment)
- pip (for Python package installation)

## Frameworks

**Core ML:**
- PyTorch 1.x/2.x - Deep learning framework used in `main.py`, `model/shift_gcn.py`, `feeders/feeder.py`
  - torch.nn - Neural network modules
  - torch.optim - Optimization algorithms (SGD with Nesterov)
  - torch.backends.cudnn - GPU acceleration and determinism
  - torch.utils.data.Dataset - Data loading interface

**Computer Vision:**
- MediaPipe Pose - Pose landmark detection in `data_gen/mediapipe_gendata.py`
  - Extracts 33 landmarks from video frames
- OpenCV (cv2) - Video reading and processing in `data_gen/mediapipe_gendata.py`

**Machine Learning:**
- scikit-learn - Evaluation metrics in `ensemble_mediapipe.py`
  - sklearn.metrics.classification_report
  - sklearn.metrics.confusion_matrix

**Testing/Utilities:**
- tqdm - Progress bars for data processing and training loops

**Build System:**
- setuptools - Python package building
- torch.utils.cpp_extension.CUDAExtension - CUDA extension compilation in `model/Temporal_shift/cuda/setup.py`
- torch.utils.cpp_extension.BuildExtension - Build system integration

## Key Dependencies

**Critical:**
- torch - DNN training, tensor operations, parameter management
- numpy - Numerical array operations, data preprocessing
- mediapipe - Pose estimation (33 landmarks)
- opencv-python (cv2) - Video codec support
- pyyaml - Configuration file parsing (train_*.yaml files)

**Infrastructure:**
- tqdm - Progress visualization during data loading and training
- pickle - Serialization of labels and evaluation results
- scikit-learn - Metrics computation for model evaluation

## Configuration

**Environment:**
- Conda environment: `goldcoin` with pre-installed dependencies
- No .env file usage (all config via YAML files and command-line arguments)
- Critical env vars for build: `DISTUTILS_USE_SDK=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `VSINSTALLDIR` (cleared before build)

**Build:**
- CUDA extension build: `cd model/Temporal_shift/cuda && python setup.py install`
  - Windows-specific: Requires VS2022 (MSVC v143) compiler
  - Conda compiler vars patched to support VS2022 instead of hardcoded VS2017
- Config files: `config/mediapipe/*.yaml` and `config/nturgbd-*/`.yaml for dataset-specific training

**YAML Configuration Examples:**
- `config/mediapipe/train_joint.yaml` - 4 modality configs (joint, bone, joint_motion, bone_motion)
- Settings: num_class=2, num_point=33, batch_size=64, num_epoch=140, num_worker=8
- Optimizer: SGD with Nesterov, base_lr=0.1, step decay [60,80,100], weight_decay=0.0001

## Platform Requirements

**Development:**
- OS: Windows 11 Pro (tested and working)
- GPU: NVIDIA GPU with CUDA 12.6 support
- Compiler: Visual Studio 2022 (MSVC v143) with C++ workload
- Storage: ~2GB for training data (joint, bone, joint_motion, bone_motion)

**Production/Training:**
- GPU: NVIDIA GPU with sufficient VRAM (tested on single GPU)
- GPU Memory: ~4-8GB estimated for batch_size=64 with full model
- Storage: ~305MB training data + model checkpoints (~100MB each)

**Data Requirements:**
- NTU RGB+D dataset (video files required for MediaPipe extraction)
- Output: `data/mediapipe/*.npy` files (~700MB-1GB total for train/val splits)

---

*Stack analysis: 2025-02-24*
