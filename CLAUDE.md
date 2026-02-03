# Shift-GCN

## Architecture
- Skeleton-based action recognition using shift graph convolutions
- `Shift_gcn` does NOT use adjacency matrix `A` — shift mechanism replaces spatial GCN entirely
- Graph topology (ntu_rgb_d, mediapipe_pose) is structural only, used for `data_bn` and consistency
- `num_point` is parameterized through `Model` → `TCN_GCN_unit` → `Shift_gcn`
- Supports NTU RGB-D (25 joints) and MediaPipe Pose (33 landmarks)

## Key Files
- `model/shift_gcn.py` — Model, TCN_GCN_unit, Shift_gcn, Shift_tcn classes
- `graph/ntu_rgb_d.py` — NTU 25-joint graph (default)
- `graph/mediapipe_pose.py` — MediaPipe 33-landmark graph
- `data_gen/preprocess.py` — Skeleton normalization (center joint, axis alignment)
- `config/` — YAML configs per dataset (ntu, mediapipe)
- `model/Temporal_shift/cuda/` — Custom CUDA kernel for shift operation

## Environment
- Requires NVIDIA GPU + CUDA toolkit for training (`shift_cuda` kernel)
- Build CUDA extension: `cd model/Temporal_shift/cuda && python setup.py install`
- Model hardcodes `device='cuda'` in `Shift_gcn` parameter initialization
- To test on CPU: mock `cuda.shift.Shift` (with stride support) and patch `torch.zeros`/`torch.ones` to drop `device` kwarg

## Conda Environment
- Use `goldcoin` conda env: has mediapipe, opencv, numpy, torch
- Run scripts via: `conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python script.py`
- `conda run` does NOT support multiline `-c` scripts — write a temp .py file instead
- On Windows/git-bash: use forward slashes in Python string paths (`D:/Shift-GCN/...`) to avoid `\t`/`\n` escape issues

## Known Deprecations (upstream, not yet fixed)
- `np.int` removed in numpy — use `int` (fixed in Shift_gcn)
- `yaml.load(f)` needs `Loader=yaml.FullLoader` (fixed in main.py)
- `nn.init.constant` / `nn.init.kaiming_normal` → use `_` suffix variants (unfixed upstream)

## Data Pipeline
- Joint data: `*_gendata.py` → `(N, 3, T, V, M)` .npy files
- Generated data lives in `data/mediapipe/` — do NOT commit (large binary, reproducible from source)
- Bone data: `gen_bone_data*.py` — joint differences along spanning tree
- Motion data: `gen_motion_data*.py` — frame differencing
- Preprocessing: `pre_normalization(data, zaxis, xaxis, center_joint)` — center_joint accepts int or list of ints for averaging
- MediaPipe center: `center_joint=[23, 24]` (hip midpoint); NTU center: `center_joint=1` (spine)

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
- Class balancing via `--subsample_ratio` (negatives subsampled to ratio * positives)
- Output: `{train,val}_data_joint.npy` + `{train,val}_label.pkl` — feeds directly into existing config/mediapipe pipeline
- Without `--ntu_mode`, original generic label_map behavior is preserved

## Testing
- POC (10-video subset): `conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ --ntu_mode --video_list poc_videos.txt --subsample_ratio 0 --benchmark xsub`
- Expected POC output: train (6, 3, 300, 33, 1) with labels {0,1}, val (4, 3, 300, 33, 1) with labels {0,1}
- `python main.py --config ./config/mediapipe/train_joint.yaml` — train MediaPipe joint model
- Ensemble: `python ensemble_mediapipe.py` (weights: 0.6 joint, 0.6 bone, 0.4 joint_motion, 0.4 bone_motion)
