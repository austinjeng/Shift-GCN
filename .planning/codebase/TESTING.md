# Testing Patterns

**Analysis Date:** 2025-02-24

## Test Framework

**Status:** No automated test suite detected

- **Test runner:** None — no `pytest.ini`, `unittest` config, or `.py` test files found
- **Test files:** Zero test files (`*_test.py`, `*test.py`, `tests/` directory absent)
- **Assertion library:** Not applicable
- **Test command:** Not applicable

## Manual Testing Approach

**Validation occurs in three places:**

1. **Data Pipeline Validation** — Scripts validate I/O, not assertions:
   - `data_gen/mediapipe_gendata.py` — prints warnings and returns None on failure
   - `feeders/feeder.py:test()` — visualization function for manual inspection
   - `graph/mediapipe_pose.py:__main__` — plots adjacency matrix

2. **Training/Eval Script Execution** — Empirical validation:
   - Training run logs to `work_dir/{Experiment_name}/log.txt`
   - Eval metrics printed to stdout and logged to `.pkl` files
   - Checkpoint integrity verified by resume mechanism

3. **Interactive Testing** — Single utility function:
   ```python
   # feeders/feeder.py:106-150
   def test(data_path, label_path, vid=None, graph=None, is_3d=False):
       '''Vis the samples using matplotlib'''
       # Loads data and plots skeleton poses frame-by-frame
   ```

## Test File Organization

**Not applicable** — no test files exist. However, validation scripts follow this pattern:

- **Data generation**: `data_gen/mediapipe_gendata.py`, `data_gen/gen_bone_data_mediapipe.py`
  - Located in `data_gen/` directory
  - Standalone executable via: `conda run -n goldcoin ... python script.py`
  - Output logged to stdout, saved to `data/mediapipe/*.npy` + `*.pkl`

- **Model evaluation**: `ensemble_mediapipe.py` at project root
  - Runs full ensemble evaluation
  - Prints metrics to stdout
  - No output file validation

- **Visualization**: `feeders/feeder.py:test()`
  - Interactive matplotlib display
  - Manual inspection of skeleton poses

## Evaluation Mechanism (Pseudo-Testing)

**Top-K Accuracy Evaluation:**
```python
# feeders/feeder.py:92-95
def top_k(self, score, top_k):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
    return sum(hit_top_k) * 1.0 / len(hit_top_k)
```

Used in training loop validation. During evaluation (main.py:450-515):
```python
def eval(self, epoch, save_score=False, ...):
    # Calls _eval_inner which:
    # 1. Disables gradients: with torch.no_grad():
    # 2. Computes predictions
    # 3. Saves results to .pkl
    # 4. Prints top-1 and top-5 accuracy
    # 5. Saves confusion metrics if weights provided
```

## Checkpoint Validation Pattern

**Test resume mechanism by training → crash → resume:**

```python
# main.py:215-229 — Resume checkpoint loading
if self.arg.resume and os.path.isfile(self.arg.resume):
    self.print_log('Resuming from checkpoint: {}'.format(self.arg.resume))
    checkpoint = torch.load(self.arg.resume)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.arg.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_acc = checkpoint['best_acc']
        # Validates checkpoint format and integrity
    else:
        # Legacy format fallback
        self.model.load_state_dict(checkpoint)
```

Checkpoint dict structure verified by isinstance check and key presence validation.

## Metrics & Validation

**Classification Metrics (ensemble_mediapipe.py:42-50):**
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(all_labels, all_preds,
                            target_names=['Non-Fall', 'Fall'], digits=4))
cm = confusion_matrix(all_labels, all_preds)
print('--- Confusion Matrix ---')
print(f'              Pred Non-Fall  Pred Fall')
print(f'  Non-Fall    {cm[0,0]:>12}  {cm[0,1]:>9}')
print(f'  Fall        {cm[1,0]:>12}  {cm[1,1]:>9}')
```

Used for final ensemble evaluation but not integrated into training loop.

## Data Validation

**Feeder class validates dataset shapes on load:**

```python
# feeders/feeder.py:41-60
def load_data(self):
    # Load label pickle
    try:
        with open(self.label_path) as f:
            self.sample_name, self.label = pickle.load(f)
    except:
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f, encoding='latin1')

    # Load data with optional mmap
    if self.use_mmap:
        self.data = np.load(self.data_path, mmap_mode='r')
    else:
        self.data = np.load(self.data_path)

    if self.debug:
        self.label = self.label[0:100]
        self.data = self.data[0:100]
        self.sample_name = self.sample_name[0:100]
```

- Checks file existence and pickle format
- Supports debug mode for small-scale testing (100 samples)
- Falls back to different pickle encoding (Python 2 compatibility)

## Model Output Validation

**Training loop validates network output:**

```python
# main.py:418-420
value, predict_label = torch.max(output.data, 1)
acc = torch.mean((predict_label == label.data).float())
```

- Computes batch accuracy in-place during training
- Prints per-log-interval: `log_interval=100` iterations

**Evaluation validates full dataset:**

```python
# main.py:482-486
_, predict_label = torch.max(output, 1)
# ... accumulate predictions
if wrong_file is not None or result_file is not None:
    predict = list(predict_label.cpu().numpy())
    true = list(label.cpu().numpy())
    for i, x in enumerate(predict):
        if result_file is not None:
            f_r.write(str(x) + ',' + str(true[i]) + '\n')
        if x != true[i] and wrong_file is not None:
            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
```

## Data Generation Tests

**Landmark extraction validates MediaPipe output:**

```python
# data_gen/mediapipe_gendata.py:46-90
def extract_landmarks(video_path, max_frame=300):
    """Extract MediaPipe Pose landmarks from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Warning: cannot open {video_path}')
        return None

    landmarks_seq = []
    with mp_pose.Pose(...) as pose:
        while cap.isOpened() and len(landmarks_seq) < max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # ... extract landmarks
            if results.pose_world_landmarks:
                frame_joints = np.array([...], dtype=np.float32)
            else:
                frame_joints = np.zeros((33, 3), dtype=np.float32)
            landmarks_seq.append(frame_joints)

    if len(landmarks_seq) == 0:
        return None  # No poses detected

    data = np.stack(landmarks_seq, axis=0)
    return data
```

**Returns None on:**
- Video file cannot open
- No frames extracted
- MediaPipe fails to detect pose (returns all-zeros frame)

**Preprocessing validates shapes:**

```python
# data_gen/preprocess.py:18-24
def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4], center_joint=1):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])

    # Fills zero frames with forward-fill
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')  # Reports empty samples
```

## Configuration Validation

**YAML config validated during arg parsing:**

```python
# main.py:569-579
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)  # Fails if config key not in parser
    parser.set_defaults(**default_arg)
```

Checks that all YAML keys are recognized argparse arguments.

## Coverage

**Requirements:** None enforced

**What IS tested (empirically):**
- Dataset loading and normalization (run via data_gen scripts)
- Model forward pass and backward pass (full training run)
- Checkpoint save/load (via `--resume` flag)
- Ensemble evaluation (standalone script)
- Top-K accuracy computation (in Feeder.top_k())

**What IS NOT tested (no unit tests):**
- Individual layer (tcn, Shift_tcn, Shift_gcn) in isolation
- Error handling edge cases (e.g., corrupted .npy files, missing labels)
- Multi-GPU DataParallel behavior
- Windows-specific DataLoader fixes (only tested empirically during full training)

## Testing Commands

No formal test runner. Manual validation via:

```bash
# 1. Generate training data
conda run -n goldcoin --cwd "D:\Shift-GCN\data_gen" python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --out_dir ../data/mediapipe/ --ntu_mode --subsample_ratio 3 --benchmark xsub

# 2. Train model (implicit testing of forward/backward passes)
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True

# 3. Resume from checkpoint (tests save/load)
conda run -n goldcoin --cwd "D:\Shift-GCN" python main.py --config ./config/mediapipe/train_joint.yaml --resume ./save_models/mediapipe_ShiftGCN_joint-50-80000.pt

# 4. Ensemble evaluation (tests aggregation and metrics)
conda run -n goldcoin --cwd "D:\Shift-GCN" python ensemble_mediapipe.py

# 5. Visualize dataset (manual inspection)
# In Python: from feeders.feeder import test; test('./data/mediapipe/val_data_joint.npy', './data/mediapipe/val_label.pkl', vid='S001C002P001R001A001', is_3d=False)
```

## Debug Mode

**Lightweight debug via `debug: False/True` flag in config:**

```python
# feeders/feeder.py:57-60
if self.debug:
    self.label = self.label[0:100]
    self.data = self.data[0:100]
    self.sample_name = self.sample_name[0:100]
```

And in main.py initialization:

```python
# main.py:184, 535
if not arg.train_feeder_args.get('debug', False):
    # Clean checkpoints logic
if not arg.test_feeder_args.get('debug', False):
    # Write wrong/result files
```

Set `debug: True` in YAML to test with 100 samples instead of full dataset.

## Conclusion

**This is a research codebase** with no formal test framework. Quality assurance relies on:
1. **Empirical validation** — Run full training pipeline and inspect output
2. **Checkpoint integrity** — Resume mechanism validates save/load
3. **Metrics publication** — Classification reports and confusion matrices
4. **Manual inspection** — Visualization functions for pose verification
5. **Debug mode** — Optional 100-sample test run

Adding pytest would be beneficial for CI/CD integration, but currently all validation is implicit in the training loop.

---

*Testing analysis: 2025-02-24*
