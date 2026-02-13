# Shift-GCN

## Status (2026-02-12)

**Full-dataset training COMPLETE** — 4 models trained, ensemble **99.77% Top-1** accuracy.
See `TRAINING_REPORT.md` for detailed results, epoch-by-epoch accuracy, and confusion matrix.
All code fixes applied (C1-C7, M1-M8, P1-P3, D1-D2) — see git log for details.

---

## Architecture
- Skeleton-based action recognition using shift graph convolutions
- `Shift_gcn` does NOT use adjacency matrix `A` — shift mechanism replaces spatial GCN entirely
- Graph topology (ntu_rgb_d, mediapipe_pose) is structural only, used for `data_bn` and consistency
- `num_point` is parameterized through `Model` → `TCN_GCN_unit` → `Shift_gcn`
- Supports NTU RGB-D (25 joints) and MediaPipe Pose (33 landmarks)

## Key Files
- `model/shift_gcn.py` — Model, TCN_GCN_unit, Shift_gcn, Shift_tcn classes
- `model/Temporal_shift/cuda/shift.py` — CUDA shift operation Python wrapper
- `graph/mediapipe_pose.py` — MediaPipe 33-landmark graph
- `data_gen/mediapipe_gendata.py` — NTU→MediaPipe extraction + class balancing
- `data_gen/gen_bone_data_mediapipe.py` — Bone data (joint-to-parent differences)
- `data_gen/gen_motion_data_mediapipe.py` — Motion data (frame differencing)
- `config/mediapipe/*.yaml` — 4 modality training configs (identical hyperparams)
- `ensemble_mediapipe.py` — 4-modality weighted ensemble evaluation
- `main.py` — Training entry point

## Environment
- Requires NVIDIA GPU + CUDA toolkit for training (`shift_cuda` kernel)
- Build CUDA extension: `cd model/Temporal_shift/cuda && python setup.py install`
- Model hardcodes `device='cuda'` in `Shift_gcn` parameter initialization
- To test on CPU: mock `cuda.shift.Shift` (with stride support) and patch `torch.zeros`/`torch.ones` to drop `device` kwarg

## CUDA Extension Build (Windows-Specific)
- Built with VS2022 (MSVC v143) + CUDA 12.6 on Windows
- Conda's `vs2017_compiler_vars.bat` and `vs2017_get_vsinstall_dir.bat` were patched:
  - Removed `-vcvars_ver=14.16` pin → allows VS2022 toolset
  - Changed vswhere filter to `-latest` → finds any VS version
- **Critical**: Must clear `VSINSTALLDIR` before calling vcvars64 to avoid path corruption
- Build pattern: clear env → set conda PATH → call vcvars64 → set `DISTUTILS_USE_SDK=1` → run setup.py
- `KMP_DUPLICATE_LIB_OK=TRUE` set in conda activation to fix OpenMP duplicate lib error

## Conda Environment
- Use `goldcoin` conda env: has mediapipe, opencv, numpy, torch, sklearn
- Run scripts via: `conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python script.py`
- `conda run` does NOT support multiline `-c` scripts — write a temp .py file instead
- `conda run` buffers all stdout until process completes — check `work_dir/*/log.txt` for real-time progress
- On Windows/git-bash: use forward slashes in Python string paths to avoid `\t`/`\n` escape issues

## Windows Caveats
- Test DataLoader capped to `min(num_worker, 2)` on `os.name == 'nt'` to avoid multiprocessing pickle crash with large val datasets (1.9GB+)
- Training DataLoader (305MB train set) works fine with 8 workers
- `conda run` may fail with UnicodeEncodeError on zh_TW locale (cp950 codec) — bypass with direct Python invocation for CUDA builds

## Checkpoint Format
- Checkpoints are dicts: `{model_state_dict, optimizer_state_dict, epoch, global_step, best_acc}`
- `--weights` auto-detects new dict format vs legacy bare state_dict
- `--resume <path.pt>` restores full training state (model + optimizer + epoch + best_acc)
- Checkpoint files: `./save_models/<Experiment_name>-<epoch>-<global_step>.pt`
- `--overwrite True` cleans old `.pt` checkpoints AND `eval_results/*.pkl`; safe with `--resume`

## Data Pipeline
- Joint data: `mediapipe_gendata.py` → `(N, 3, T, V, M)` .npy files
- Generated data lives in `data/mediapipe/` — do NOT commit (large binary, reproducible)
- Bone data: `gen_bone_data_mediapipe.py` — joint differences along spanning tree
- Motion data: `gen_motion_data_mediapipe.py` — frame differencing
- Preprocessing: `pre_normalization(data, zaxis, xaxis, center_joint)` — center_joint accepts int or list
- MediaPipe center: `center_joint=[23, 24]` (hip midpoint); NTU center: `center_joint=1` (spine)

## NTU RGB+D Dataset
- Filename pattern: `SsssCcccPpppRrrrAaaa.ext` (setup, camera, subject, replication, action)
- Training subjects (xsub): `{1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38}`
- A043 = "falling down" — positive class for binary fall detection
- `--ntu_mode` extracts MediaPipe landmarks; `--subsample_ratio 3` balances train set (3:1 non-fall:fall)
- Binary labels: A043 = 1, everything else = 0; val set keeps natural distribution

## Training Results (Full Dataset)
- Dataset: 2,688 train (672 fall + 2,016 non-fall) / 16,560 val (276 fall + 16,284 non-fall)
- Joint: 99.49% | Bone: 99.51% | Joint Motion: 99.46% | Bone Motion: 99.64%
- **Ensemble: 99.77% Top-1** | Fall F1: 92.91% (precision 95.77%, recall 90.22%)
- Training: ~2h per model, 140 epochs, batch 64, SGD+Nesterov, LR 0.1 step decay [60,80,100]
- `drop_last=True` on training DataLoader

## Commands
```bash
# Data generation
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ --ntu_mode --subsample_ratio 3 --benchmark xsub
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_bone_data_mediapipe.py
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_motion_data_mediapipe.py

# Training (4 models, ~2h each)
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint_motion.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone_motion.yaml --overwrite True

# Resume after crash
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --resume ./save_models/mediapipe_ShiftGCN_joint-EPOCH-STEP.pt

# Ensemble (weights: 0.6 joint, 0.6 bone, 0.4 joint_motion, 0.4 bone_motion)
conda run -n goldcoin --cwd "D:\Shift-GCN" python ensemble_mediapipe.py
```

## Documentation
- Training report: `TRAINING_REPORT.md` (full results, epoch-by-epoch accuracy, confusion matrix)
- Adaptation report: `MEDIAPIPE_ADAPTATION_REPORT_zh-TW.md` (non-technical background)
