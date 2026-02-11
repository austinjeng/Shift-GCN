# Shift-GCN

## Current Progress (2026-02-09)

**目標**：使用 MediaPipe Pose 從 NTU RGB+D 影片進行二元跌倒偵測

| 階段 | 狀態 | 備註 |
|------|------|------|
| MediaPipe 圖結構 (`graph/mediapipe_pose.py`) | ✅ 完成 | 33 landmarks, 橋接邊連通 |
| 模型參數化 (`model/shift_gcn.py`) | ✅ 完成 | `num_point` 可設定 |
| 前處理修改 (`data_gen/preprocess.py`) | ✅ 完成 | 支援多關節中心平均 |
| 資料生成腳本 (`data_gen/mediapipe_gendata.py`) | ✅ 完成 | NTU 模式 + 類別平衡 |
| 設定檔 (`config/mediapipe/*.yaml`) | ✅ 完成 | 4 種模態 (production: 140 epochs) |
| POC Joint 資料生成 | ✅ 完成 | train: 6, val: 4 |
| Bone 資料生成 | ✅ 完成 | `gen_bone_data_mediapipe.py` |
| Motion 資料生成 | ✅ 完成 | `gen_motion_data_mediapipe.py` (4 files) |
| CUDA 擴展編譯 | ✅ 完成 | VS2022 + CUDA 12.6 (Windows) |
| 模型訓練 (POC) | ✅ 完成 | 4 模態 x 10 epochs, Top-1: 50% |
| 集成評估 (POC) | ✅ 完成 | Top-1: 50%, Top-5: 100% |
| 程式碼審查 | ✅ 完成 | 發現關鍵問題待修 (見下方) |
| 關鍵修復 C1-C7 | ✅ 完成 | 全部 7 個問題已修復 |
| ML-Ops 修復 M1-M8 | ✅ 完成 | interval/resume/cleanup/ensemble 等 8 項 |
| 穩健性修復 P1-P3 | ✅ 完成 | resume 安全、記憶體優化、檔案處理 (6 項) |
| 技術債清理 D1-D2 | ✅ 完成 | 目錄建立冪等化、移除棄用 Variable |

### Critical Fixes Applied (C1-C7)

All issues identified in code review have been **fixed**:

| ID | Severity | File | Fix Applied |
|----|----------|------|-------------|
| C1 | Critical | `main.py` | Removed `Variable(volatile=True)`, moved tensors inside `torch.no_grad()` |
| C2 | Critical | `main.py` | Added `--overwrite` CLI flag, removed interactive `input()` prompts |
| C3 | Critical | `config/mediapipe/train_*.yaml` | Restored `num_epoch: 140`, `num_worker: 8`, `eval_interval: 5` |
| C4 | Important | `model/shift_gcn.py` | Added `_` suffix to all 8 `nn.init` calls |
| C5 | Consider | `data_gen/mediapipe_gendata.py` | Chunk-based processing (5000 videos/chunk) |
| C6 | Consider | `data_gen/preprocess.py` | Vectorized rotation with `np.dot(person[mask], matrix.T)` |
| C7 | Consider | `ensemble_mediapipe.py` | Added `classification_report`, confusion matrix |

### ML-Ops Fixes Applied (M1-M8)

| ID | Priority | File | Fix Applied |
|----|----------|------|-------------|
| M1 | HIGH | `main.py` | Honor `eval_interval` and `save_interval` in training loop |
| M2 | HIGH | `mediapipe_gendata.py` | Stop class-balancing validation split (train only) |
| M3 | HIGH | `main.py` | Resume-safe checkpoints (`--resume`, full state dict) |
| M4 | MEDIUM | `main.py` | Fix `KeyError` on missing `debug` key (`.get()`) |
| M5 | MEDIUM | `ensemble_mediapipe.py` | Dict-based sample alignment (not index-based) |
| M6 | LOW | `main.py` | Glob-based checkpoint cleanup (not `shutil.rmtree`) |
| M7 | LOW | `main.py` | Removed dead `ReduceLROnPlateau` scheduler |
| M8 | LOW | `main.py` | Replaced global `arg` with `self.arg` in eval |

### Robustness Fixes Applied (P1-P3)

| ID | Priority | File | Fix Applied |
|----|----------|------|-------------|
| P1a | HIGH | `main.py` | `--overwrite` skips resume target via `os.path.abspath()` comparison |
| P1b | HIGH | `main.py` | Removed wrong `global_step` formula in `start()`; trusts `__init__` value |
| P1c | HIGH | `mediapipe_gendata.py` | Two-pass mmap chunk concat (preallocate + sequential copy) |
| P2a | MEDIUM | `main.py` | `--overwrite` also cleans `eval_results/*.pkl` |
| P3a | LOW | `main.py` | Eval file handles wrapped in `try/finally` via `_eval_inner` |
| P3b | LOW | `main.py` | `.pkl` weights opened in `'rb'` (binary mode) |

### Tech Debt Cleanup (D1-D2)

| ID | Priority | File | Fix Applied |
|----|----------|------|-------------|
| D1 | LOW | `main.py` | `eval_results/` dir creation uses `exist_ok=True` (idempotent) |
| D2 | LOW | `main.py` | Removed deprecated `torch.autograd.Variable` wrapper from training loop |

### Next Steps
```bash
# 1. Generate full dataset (remove --video_list, set --subsample_ratio appropriately)
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ --ntu_mode --subsample_ratio 3 --benchmark xsub
# 2. Regenerate bone + motion data
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_bone_data_mediapipe.py
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python gen_motion_data_mediapipe.py
# 3. Train all 4 models (140 epochs each, use --overwrite True to auto-remove old checkpoints)
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint_motion.yaml --overwrite True
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_bone_motion.yaml --overwrite True
# To resume after crash: --resume ./save_models/mediapipe_ShiftGCN_joint-EPOCH-STEP.pt
# 4. Ensemble evaluation (now includes classification_report + confusion matrix)
conda run -n goldcoin --cwd "D:\Shift-GCN" python ensemble_mediapipe.py
```

### Documentation
- 完整適配報告：`MEDIAPIPE_ADAPTATION_REPORT_zh-TW.md`（含非技術背景說明）

---

## Architecture
- Skeleton-based action recognition using shift graph convolutions
- `Shift_gcn` does NOT use adjacency matrix `A` — shift mechanism replaces spatial GCN entirely
- Graph topology (ntu_rgb_d, mediapipe_pose) is structural only, used for `data_bn` and consistency
- `num_point` is parameterized through `Model` → `TCN_GCN_unit` → `Shift_gcn`
- Supports NTU RGB-D (25 joints) and MediaPipe Pose (33 landmarks)

## Key Files
- `model/shift_gcn.py` — Model, TCN_GCN_unit, Shift_gcn, Shift_tcn classes
- `model/Temporal_shift/cuda/shift.py` — CUDA shift operation Python wrapper (patched: contiguous + saved_tensors)
- `model/Temporal_shift/cuda/setup.py` — CUDA extension build config (CUDAExtension)
- `graph/ntu_rgb_d.py` — NTU 25-joint graph (default)
- `graph/mediapipe_pose.py` — MediaPipe 33-landmark graph
- `data_gen/preprocess.py` — Skeleton normalization (center joint, axis alignment)
- `data_gen/gen_bone_data_mediapipe.py` — Bone data generation (joint-to-parent differences)
- `data_gen/gen_motion_data_mediapipe.py` — Motion data generation (frame differencing)
- `config/` — YAML configs per dataset (ntu, mediapipe)
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
  - Removed `-vcvars_ver=14.16` pin (VS2017 toolset) → allows VS2022 toolset
  - Changed vswhere filter from `-version [15.0,16.0)` to `-latest` → finds any VS version
- **Critical build pattern**: Must clear `VSINSTALLDIR` before calling vcvars64 to avoid path corruption (`CommunityVC\` instead of `Community\VC\`)
- Build script pattern: clear env → set conda PATH → call vcvars64 → set `DISTUTILS_USE_SDK=1` → run setup.py
- `KMP_DUPLICATE_LIB_OK=TRUE` set in conda activation to fix OpenMP duplicate lib error (numpy + torch both link libiomp5)
- Patched conda activation scripts at: `C:\Anaconda\envs\goldcoin\etc\conda\activate.d\`
- DLL loading: torch lib and CUDA bin directories must be on PATH (conda activation handles this)

## Conda Environment
- Use `goldcoin` conda env: has mediapipe, opencv, numpy, torch, sklearn
- Run scripts via: `conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python script.py`
- `conda run` does NOT support multiline `-c` scripts — write a temp .py file instead
- `conda run` may fail with UnicodeEncodeError on zh_TW locale (cp950 codec) — bypass with direct Python invocation for CUDA builds
- On Windows/git-bash: use forward slashes in Python string paths (`D:/Shift-GCN/...`) to avoid `\t`/`\n` escape issues

## Code Fixes Applied
- `shift.py`: `input.contiguous()` in forward + `ctx.saved_tensors` (deprecated API)
- C1: eval loop uses `torch.no_grad()` instead of `Variable(volatile=True)`
- C2: `--overwrite` CLI flag replaces interactive `input()` prompts
- C4: all `nn.init` calls use `_` suffix
- C5: `_extract_and_save` processes in chunks (default 5000 videos/chunk)
- C6: rotation loops vectorized with `np.dot(person[mask], matrix.T)`
- C7: `ensemble_mediapipe.py` requires `sklearn` for classification_report
- M1: Training loop honors `eval_interval` and `save_interval` (no more every-epoch saves)
- M2: Validation split no longer class-balanced (preserves natural distribution)
- M3: `--resume <checkpoint.pt>` restores model, optimizer, epoch, global_step, best_acc
- M4: `debug` key accessed via `.get('debug', False)` to avoid KeyError
- M5: Ensemble uses dict-based sample lookup (robust to ordering differences)
- M6: Checkpoint cleanup uses `glob.glob()` instead of `shutil.rmtree()` (matches flat .pt files)
- M7: Removed unused `ReduceLROnPlateau` scheduler
- M8: Eval methods use `self.arg` instead of module-level `arg`
- P1a: `--overwrite` protects `--resume` target from deletion (abspath comparison)
- P1b: Removed `global_step` recalculation in `start()` (was wrong units + overwrote checkpoint value)
- P1c: Chunk concat uses two-pass mmap + preallocate instead of loading all chunks at once
- P2a: `--overwrite` cleans `eval_results/*.pkl` alongside checkpoint `.pt` files
- P3a: Eval `wrong_file`/`result_file` handles closed in `finally` block
- P3b: `.pkl` weight file opened with `'rb'` instead of `'r'`
- D1: `eval_results/` dir creation uses `os.makedirs(..., exist_ok=True)` instead of conditional check
- D2: Removed `from torch.autograd import Variable`; training loop uses plain tensors directly

## Checkpoint Format
- Checkpoints are dicts: `{model_state_dict, optimizer_state_dict, epoch, global_step, best_acc}`
- `--weights` auto-detects new dict format vs legacy bare state_dict
- `--resume <path.pt>` restores full training state (model + optimizer + epoch + best_acc)
- Checkpoint files: `./save_models/<Experiment_name>-<epoch>-<global_step>.pt` (flat files, not directories)
- `--overwrite True` cleans old `.pt` checkpoints AND `eval_results/*.pkl`; safe to combine with `--resume` (resume target is protected)

## Known Deprecations (all fixed)
- `np.int` removed in numpy — use `int` (fixed in Shift_gcn)
- `yaml.load(f)` needs `Loader=yaml.FullLoader` (fixed in main.py)
- `nn.init.constant` / `nn.init.kaiming_normal` → use `_` suffix variants (fixed: C4)
- `volatile=True` in `Variable()` — deprecated since PyTorch 0.4 (fixed: C1)
- `Variable()` wrapper itself — unnecessary since PyTorch 0.4 (fixed: D2)

## Data Pipeline
- Joint data: `*_gendata.py` → `(N, 3, T, V, M)` .npy files
- Generated data lives in `data/mediapipe/` — do NOT commit (large binary, reproducible from source)
- Bone data: `gen_bone_data*.py` — joint differences along spanning tree
- Motion data: `gen_motion_data*.py` — frame differencing
- Preprocessing: `pre_normalization(data, zaxis, xaxis, center_joint)` — center_joint accepts int or list of ints for averaging
- MediaPipe center: `center_joint=[23, 24]` (hip midpoint); NTU center: `center_joint=1` (spine)
- POC data shapes: train `(6, 3, 300, 33, 1)`, val `(4, 3, 300, 33, 1)` — all 8 files (joint/bone × motion × train/val) verified

## NTU RGB+D Filename Convention
- Pattern: `SsssCcccPpppRrrrAaaa.ext` (setup, camera, subject, replication, action)
- Parsed via `parse_ntu_filename()` in `data_gen/mediapipe_gendata.py`
- Training subjects (xsub): `{1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38}`
- Training cameras (xview): `{2, 3}`
- A043 = "falling down" — used as positive class for binary fall detection

## NTU-to-MediaPipe Pipeline
- `mediapipe_gendata.py --ntu_mode` extracts MediaPipe landmarks from NTU RGB+D videos
- `--video_list <file>` filters to a subset of video basenames (one per line); omit to process all
- POC test list: `data_gen/poc_videos.txt` (10 videos, balanced fall/non-fall across train/val)
- Binary labels: A043 (falling) = 1, everything else = 0
- Cross-subject or cross-view split via `--benchmark xsub|xview`
- Class balancing via `--subsample_ratio` (train only; val keeps natural distribution)
- Output: `{train,val}_data_joint.npy` + `{train,val}_label.pkl` — feeds directly into existing config/mediapipe pipeline
- Without `--ntu_mode`, original generic label_map behavior is preserved

## Training Performance Notes
- With `num_worker: 32` and 6 samples: ~1.5 min/epoch (DataLoader overhead dominates)
- With `num_worker: 2` and 6 samples: ~5 sec/epoch — 18x faster for POC
- For full dataset: `num_worker: 8-12` is recommended (match to CPU cores, not exceed)
- `main.py` uses `drop_last=True` on training DataLoader — watch for data loss with small datasets

## POC Results (10 videos, 10 epochs)
- All 4 modalities: Top-1 50%, Top-5 100% — expected random performance for tiny dataset
- Loss constant at 52.71 across all epochs — model not learning (expected: too few samples)
- Pipeline validated end-to-end: data gen → bone → motion → CUDA build → train × 4 → ensemble

## Testing
- POC (10-video subset): `conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ --ntu_mode --video_list poc_videos.txt --subsample_ratio 0 --benchmark xsub`
- Expected POC output: train (6, 3, 300, 33, 1) with labels {0,1}, val (4, 3, 300, 33, 1) with labels {0,1}
- `python main.py --config ./config/mediapipe/train_joint.yaml` — train MediaPipe joint model
- Ensemble: `python ensemble_mediapipe.py` (weights: 0.6 joint, 0.6 bone, 0.4 joint_motion, 0.4 bone_motion)
- Resume training: `python main.py --config ./config/mediapipe/train_joint.yaml --resume ./save_models/mediapipe_ShiftGCN_joint-EPOCH-STEP.pt`
