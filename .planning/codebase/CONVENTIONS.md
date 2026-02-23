# Coding Conventions

**Analysis Date:** 2025-02-24

## Naming Patterns

**Files:**
- Lowercase with underscores: `mediapipe_gendata.py`, `shift_gcn.py`, `feeder.py`
- Configuration files in `.yaml`: `train_joint.yaml`, `train_bone.yaml`
- Graph definition files: `ntu_rgb_d.py`, `mediapipe_pose.py`

**Functions:**
- Lowercase with underscores: `parse_ntu_filename()`, `extract_landmarks()`, `pre_normalization()`
- Utility functions in module-level functions (not class methods where possible)
- Constructor/initialization functions: `__init__()`, `load_data()`, `load_model()`

**Variables:**
- Lowercase with underscores for regular variables: `sample_name`, `data_path`, `batch_size`
- CamelCase for class instances: `Feeder`, `Model`, `Processor`, `Graph`
- Uppercase for constants: `NTU_TRAINING_SUBJECTS`, `max_body_true`, `max_frame`
- Parameter dict keys use underscores: `train_feeder_args`, `model_args`, `graph_args`

**Types & Classes:**
- PascalCase: `Feeder`, `Model`, `Processor`, `Graph`, `Shift_gcn`, `TCN_GCN_unit`
- Functional classes mixed case: `ShiftFunction`, `Shift_tcn`, `Shift`
- Module classes inherit from `nn.Module` or `Dataset`

## Code Style

**Formatting:**
- No explicit linter/formatter config detected (`.flake8`, `.pylintrc` absent)
- Convention by observation:
  - 4-space indentation (PEP 8 standard)
  - Line wrapping for long argument lists (seen in `main.py:234-250`)
  - Dict/argument formatting on multiple lines for readability

**Linting:**
- No linting rules enforced in codebase
- Some violations observed (e.g., bare `except:` at `main.py:285`, unused imports)

## Import Organization

**Order:**
1. Standard library imports: `import argparse`, `import os`, `import time`, `import sys`
2. Third-party scientific/ML: `import numpy`, `import torch`, `import cv2`, `import mediapipe`
3. Third-party utilities: `from tqdm import tqdm`, `import yaml`, `import pickle`
4. Local imports: `sys.path.extend(['../'])`, `from feeders import tools`

**Path Aliases:**
- Relative imports with `sys.path` manipulation: `sys.path.extend(['../'])` (seen in `feeders/feeder.py:7`)
- No standardized alias system; full module paths used in config: `feeder: feeders.feeder.Feeder`
- Dynamic class loading via `import_class()` utility (replicated in `main.py:558`, `model/shift_gcn.py:14`)

**Example (from main.py:1-21):**
```python
import argparse
import glob
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
```

## Error Handling

**Patterns:**
- Try-except with broad exception catching:
  ```python
  # main.py:264-270: Detect checkpoint format (new dict vs legacy state_dict)
  if '.pkl' in self.arg.weights:
      with open(self.arg.weights, 'rb') as f:
          weights = pickle.load(f)
  else:
      weights = torch.load(self.arg.weights)
  if isinstance(weights, dict) and 'model_state_dict' in weights:
      weights = weights['model_state_dict']
  ```

- Silent fallback with encoding handling:
  ```python
  # feeders/feeder.py:44-50: Handle Python 2/3 pickle compatibility
  try:
      with open(self.label_path) as f:
          self.sample_name, self.label = pickle.load(f)
  except:
      with open(self.label_path, 'rb') as f:
          self.sample_name, self.label = pickle.load(f, encoding='latin1')
  ```

- Data validation with warnings (not exceptions):
  ```python
  # data_gen/mediapipe_gendata.py:54-56: Warn instead of raise
  if not cap.isOpened():
      print(f'Warning: cannot open {video_path}')
      return None
  ```

- Bare exception handling (anti-pattern, seen in main.py:285):
  ```python
  try:
      self.model.load_state_dict(weights)
  except:  # Bare except - catches all exceptions
      state = self.model.state_dict()
      diff = list(set(state.keys()).difference(set(weights.keys())))
  ```

**No custom exceptions defined** — relies on built-in exceptions (ValueError, IndexError, argparse.ArgumentTypeError).

## Logging

**Framework:** `print()` directly to stdout; no logging module

**Patterns:**
- Print with timestamp in `Processor.print_log()`:
  ```python
  # main.py:359-366
  def print_log(self, str, print_time=True):
      if print_time:
          localtime = time.asctime(time.localtime(time.time()))
          str = "[ " + localtime + ' ] ' + str
      print(str)
      if self.arg.print_log:
          with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
              print(str, file=f)
  ```

- Progress bars with `tqdm()`:
  ```python
  # main.py:385: Training progress
  process = tqdm(loader)
  ```

- Manual timing with `record_time()` and `split_time()`:
  ```python
  # main.py:368-375
  def record_time(self):
      self.cur_time = time.time()
      return self.cur_time

  def split_time(self):
      split_time = time.time() - self.cur_time
      self.record_time()
      return split_time
  ```

- Info printed mid-training:
  ```python
  # main.py:425-427
  self.print_log(
      '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}  network_time: {:.4f}'.format(
          batch_idx, len(loader), loss.data, self.lr, network_time))
  ```

## Comments

**When to Comment:**
- Sparse; most code is uncommented
- Comments used for:
  - Section markers with hyphens: `# data: N C V T M` (feeder.py:42)
  - Algorithm explanation: `# xuanzhuan juzhen` (rotation, feeders/tools.py:92)
  - Non-obvious transformations: `# N, C, T, V, M  to  N, M, T, V, C` (preprocess.py:19)

**JSDoc/TSDoc:**
- Used in `mediapipe_gendata.py` for public functions:
  ```python
  # data_gen/mediapipe_gendata.py:26-43
  def parse_ntu_filename(filename):
      """Parse NTU RGB+D filename (SsssCcccPpppRrrrAaaa.ext) to extract metadata.

      Returns:
          dict with keys: setup, camera, subject, replication, action, or None on failure.
      """
  ```

- Docstrings in feeder constructor:
  ```python
  # feeders/feeder.py:12-26
  def __init__(self, data_path, label_path, ...):
      """
      :param data_path:
      :param label_path:
      :param random_choose: If true, randomly choose a portion of the input sequence
      ...
      """
  ```

- Inline docstrings for pre_normalization args:
  ```python
  # data_gen/preprocess.py:8-16
  def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4], center_joint=1):
      """Normalize skeleton data.

      Args:
          data: (N, C, T, V, M) skeleton data.
          zaxis: [joint_bottom, joint_top] for z-axis alignment.
          ...
      """
  ```

## Function Design

**Size:** Mix of short utility functions (10-20 lines) and longer coordinate/training loops (50-100 lines)
- Utility: `conv_init()` (3 lines), `edge2mat()` (6 lines)
- Medium: `extract_landmarks()` (45 lines), `auto_pading()` (9 lines)
- Large: `train()` (80 lines), `_eval_inner()` (70 lines)

**Parameters:**
- Keyword args with defaults for data loading: `Feeder(..., debug=False, use_mmap=True)`
- Dict unpacking for nested config: `Feeder(**self.arg.train_feeder_args)`
- Mixed positional/keyword in methods: `top_k(self, score, top_k)`

**Return Values:**
- Single return value (numpy array, tensor, bool)
- Tuple return for multi-value operations: `pickle.load()` returns `(sample_name, label)`
- None for void operations: `load_data()`, `load_model()` (side-effect heavy)
- Early returns for validation:
  ```python
  # data_gen/mediapipe_gendata.py:54-56
  if not cap.isOpened():
      return None
  ```

## Module Design

**Exports:**
- `Feeder` class primary export from `feeders/feeder.py`
- Graph classes as module-level exports: `graph.ntu_rgb_d.Graph`, `graph.mediapipe_pose.Graph`
- Config-driven imports via dynamic `import_class()`:
  ```python
  # main.py:256
  Model = import_class(self.arg.model)  # e.g., 'model.shift_gcn.Model'
  ```

**Barrel Files:**
- `__init__.py` files exist but minimal: `feeders/__init__.py`, `graph/__init__.py`, `model/__init__.py`
- No re-exports observed; mostly empty

**Example (main.py:558-563):**
```python
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
```

This pattern allows fully qualified imports from config files without hardcoding module paths.

## YAML Configuration Pattern

Configuration drives training/testing behavior. Key structure:
```yaml
Experiment_name: mediapipe_ShiftGCN_joint
feeder: feeders.feeder.Feeder
train_feeder_args:
  data_path: ./data/mediapipe/train_data_joint.npy
  label_path: ./data/mediapipe/train_label.pkl
  debug: False
  random_choose: False
  window_size: -1
  normalization: False
model: model.shift_gcn.Model
model_args:
  num_class: 2
  num_point: 33
  graph: graph.mediapipe_pose.Graph
  graph_args:
    labeling_mode: 'spatial'
```

Keys use snake_case for consistency with Python naming.

## Tensor Operations

**Framework:** PyTorch exclusively

**Conventions:**
- Shape notation in comments: `(N, C, T, V, M)` for `(batch, channels, time, vertices, persons)`
- In-place operations: `.permute()`, `.contiguous()` for tensor layout optimization
- `.cuda()` explicit device placement (hardcoded in `Shift_gcn` at `model/shift_gcn.py:90`)
- No `.to()` device abstraction used
- `.float()` and `.long()` explicit dtype casting:
  ```python
  # main.py:400-401
  data = data.float().cuda(self.output_device)
  label = label.long().cuda(self.output_device)
  ```

## Parameter Initialization

**Pattern (model/shift_gcn.py):**
- `nn.init.kaiming_normal_()` for conv weights (seen in `shift_gcn.py:22, 63`)
- `nn.init.constant_()` for bias and batch norm scales
- `nn.init.normal_()` for learnable linear weights (shift_gcn.py:91)
- Uniform for shift parameters: `.uniform_(-1e-8, 1e-8)` and `.uniform_(-init_scale, init_scale)` (cuda/shift.py:42-43)

---

*Convention analysis: 2025-02-24*
