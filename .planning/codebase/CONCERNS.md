# Codebase Concerns

**Analysis Date:** 2026-02-24

## Tech Debt

**Hardcoded CUDA Device in Model Parameters:**
- Issue: Model parameters are initialized with `device='cuda'` hardcoded, preventing CPU execution or flexible multi-GPU deployment
- Files: `model/shift_gcn.py` (lines 90, 93, 96), `model/Temporal_shift/cuda/shift.py` (lines 39-40)
- Impact: Cannot run inference on CPU, forces single GPU usage, breaks portability across environments
- Fix approach: Accept device as a constructor parameter and pass to `torch.zeros()`, `torch.ones()`, `Parameter()` calls. Update `Shift_gcn.__init__()` and `Shift.__init__()` signatures to accept `device` kwarg from model instantiation

**Bare Except Clauses:**
- Issue: Multiple bare `except:` statements catch all exceptions including `KeyboardInterrupt`, `SystemExit`, making debugging difficult
- Files:
  - `main.py:285` — weight loading fallback silently absorbs any load error
  - `feeders/feeder.py:47` — pickle loading accepts both text and binary modes but generic handler masks real errors
- Impact: Hard to diagnose actual errors; may hide data corruption or malformed pickles
- Fix approach: Replace with specific exception types (`pickle.UnpicklingError`, `RuntimeError`, etc.) and log the actual error before fallback

**Bare Except in Weight Loading:**
- Issue: `main.py:285` has `except:` that attempts state dict merging without knowing actual failure reason
- Files: `main.py:283-292`
- Impact: If weights are corrupted or incompatible, the partial merge could silently load a hybrid model with undefined behavior
- Fix approach: Catch `RuntimeError` specifically, inspect and report missing/mismatched keys, and fail explicitly if incompatibility is severe

## Known Bugs

**Missing Pickle File Handle Closure:**
- Symptoms: Memory leaks in long-running pipelines or batch data generation
- Files: `feeders/feeder.py:45-50` — pickle.load() called in context manager but feeder is never closed
- Trigger: Create many Feeder instances in loops without cleanup
- Workaround: Feeder instances are typically created once, but large-scale data processing loops should explicitly call `del feeder` to release file handles

**Ensemble Score Dictionary Hardcoded Paths:**
- Symptoms: `ensemble_mediapipe.py` fails if `best_acc.pkl` files are missing or in unexpected locations
- Files: `ensemble_mediapipe.py:8-13` — hardcoded paths to 4 `.pkl` files with no existence checks
- Trigger: If training crashes before best checkpoint is saved, or if work_dir is reorganized
- Workaround: Manually verify all 4 pkl files exist before running ensemble; add error checking to script

## Security Considerations

**Dynamic Module Import via String:**
- Risk: `__import__()` + `getattr()` pattern allows arbitrary module loading if input not validated
- Files: `main.py:558-563`, `feeders/feeder.py:98-103`, `model/shift_gcn.py:14-19`
- Current mitigation: Module names come from config files (YAML) which are version-controlled, not user input
- Recommendations: Add whitelist of allowed modules (`feeder.feeder`, `model.shift_gcn`, `graph.*`) and validate against it before importing; document that config files are trusted input only

**Pickle Deserialization from Untrusted Sources:**
- Risk: `pickle.load()` can execute arbitrary code if pickle file is malicious
- Files: `ensemble_mediapipe.py:9-13`, `feeders/feeder.py:46-50`, `main.py:265-266`
- Current mitigation: Pickle files generated internally during training; no external input
- Recommendations: Add validation that pickle file size is reasonable (< 1GB); consider using `unsafe=False` option if available; never load pickles from user uploads

**YAML Deserialization:**
- Risk: `yaml.load()` without safe loader can execute code
- Files: `main.py:573`
- Current mitigation: Config files are checked into git, not user-uploaded
- Recommendations: Already using `yaml.FullLoader` (safe); document that only dev team edits configs; consider switching to `yaml.SafeLoader` if no advanced YAML features needed

## Performance Bottlenecks

**Windows DataLoader with Large Validation Set:**
- Problem: Test DataLoader capped to 2 workers on Windows (vs. 8 on Linux) due to multiprocessing pickle serialization limits with 1.9 GB dataset
- Files: `main.py:242-244`
- Cause: Windows uses `spawn` (copies parent state to each worker), causing pickle buffer overflow with 8 workers; Linux uses `fork` (shares memory)
- Improvement path:
  1. Reduce worker count to 1 and use `pin_memory=True` + increase batch size to maintain throughput
  2. Use memory-mapped loading (already done with `use_mmap=True` in Feeder) but ensure all data processing avoids copies
  3. Separate large dataset into shards to reduce per-worker memory footprint
  4. Consider switching to `torch.utils.data.distributed.DistributedSampler` for formal multi-process handling

**Ensemble Evaluation with Hardcoded Paths:**
- Problem: `ensemble_mediapipe.py:8-13` loads all 4 pickle files sequentially from disk; no batching or memory-mapped access
- Files: `ensemble_mediapipe.py`
- Cause: 4 × 16,560 samples × 2 classes × 8 bytes = ~2.1 MB per pickle, loaded entirely into RAM
- Improvement path: Stream pickle data in batches, or use numpy memmap for on-disk random access without full load

## Fragile Areas

**Graph Dependency in Model:**
- Files: `model/shift_gcn.py:172-173`, `main.py:256-257`
- Why fragile: Model graph topology is passed as string identifier, imported at runtime; if graph module changes structure (edges, nodes), model behavior silently breaks
- Safe modification: Add assertions in `Shift_gcn.__init__()` to check `A.shape` matches expected `num_point`; validate graph connectivity before forward pass
- Test coverage: No unit tests for different graph topologies; only integration tests via full training

**Shift Operation CUDA Kernel:**
- Files: `model/Temporal_shift/cuda/shift.py:12-23`
- Why fragile: Custom autograd function depends on C++ CUDA extension (`shift_cuda`); if extension not built or version mismatch, forward pass crashes silently
- Safe modification: Add `hasattr(torch, 'cuda')` and try/except on first ShiftFunction call to fail early with clear error; add version marker to extension
- Test coverage: No unit tests for shift operation; only tested indirectly through full forward pass

**Parameter Initialization Device Mismatch:**
- Files: `model/shift_gcn.py:90-96`, `model/Temporal_shift/cuda/shift.py:39-40`
- Why fragile: If model is moved to different device after init (e.g., `model.to('cuda:1')`), hardcoded `device='cuda'` parameters stay on default GPU
- Safe modification: Initialize parameters on CPU by default, then let `.to(device)` move them; or accept device argument
- Test coverage: No multi-GPU tests; assumes single GPU allocation

**File I/O Without Error Handling:**
- Files: `main.py:339, 365, 501, 514` — `open()` calls in data saving without try/except
- Why fragile: If work_dir doesn't exist or is read-only, saving config/logs/pkl files will crash with no cleanup
- Safe modification: Create directory with `os.makedirs(exist_ok=True)` before any save operation (already done for work_dir, but not checked before file open); wrap file I/O in try/except
- Test coverage: No tests for disk-full or permission-denied scenarios

## Scaling Limits

**Validation Dataset Size vs. DataLoader Workers:**
- Current capacity: 1.9 GB val set with 2 workers on Windows; 8 workers on Linux
- Limit: Adding more workers on Windows will trigger multiprocessing pickle overflow; larger datasets (> 2.5 GB) may fail even with 1 worker
- Scaling path:
  1. Use `torch.utils.data.DataLoader` with `prefetch_factor` and `persistent_workers=True` to reduce per-worker overhead
  2. Implement custom multi-process fetching with queue-based batching instead of DataLoader
  3. Switch to distributed data loading across multiple machines (DistributedDataParallel)

**Checkpoint File Storage:**
- Current capacity: 4 models × 70 checkpoints × 6.3 MB = 1.76 GB
- Limit: Training 10 models would generate ~4.4 GB; archive storage becomes expensive
- Scaling path: Implement checkpoint pruning (keep only best-K and latest-K); use checkpoint sharding to save only difference from previous epoch; compress checkpoints

**Model Parameter Initialization:**
- Current capacity: 720K parameters fit in GPU memory; forward/backward on batch of 64 works on single GPU
- Limit: Scaling to larger models (e.g., > 50M parameters) or bigger batch sizes may require gradient checkpointing or model parallelism
- Scaling path: Add `torch.utils.checkpoint.checkpoint()` to selectively recompute forward passes; implement model sharding for multi-GPU training

## Dependencies at Risk

**PyTorch Version Pinning:**
- Risk: No explicit PyTorch version constraint; may break on major updates (e.g., PyTorch 2.0+ changed tensor type conventions)
- Impact: `torch.autograd.Function` API is stable, but `Parameter(device=)` kwarg behavior varies
- Migration plan: Document minimum PyTorch version in README; add `torch.__version__` check at startup; test on both PyTorch 1.13 and 2.0+

**MediaPipe Dependency:**
- Risk: MediaPipe 33-landmark model is Google-managed; if Google discontinues or significantly changes the model, all extracted data becomes incompatible
- Impact: Data generation pipeline (`mediapipe_gendata.py`) depends on specific landmark indices (0-32); model update could re-index or rename landmarks
- Migration plan: Lock MediaPipe version in conda env; document landmark index mapping in `TRAINING_REPORT.md`; add fallback pose estimator (e.g., OpenPose)

**CUDA Toolkit Version Mismatch:**
- Risk: CUDA extension built with CUDA 12.6; if environment has CUDA 11.8 or 12.0, extension load fails with cryptic symbol mismatch error
- Impact: Training cannot start; binary is not compatible across CUDA minor versions
- Migration plan: Distribute pre-built wheels for common CUDA versions; or implement CPU fallback for shift operation (slower but runnable)

## Missing Critical Features

**No Validation Metric Logging:**
- Problem: Best accuracy saved to `best_acc.pkl` but no historical log of validation curve
- Blocks: Cannot easily plot training progress or detect overfitting without manual epoch-by-epoch grep
- Improvement: Add CSV logging of (epoch, train_loss, val_acc, val_f1) to work_dir

**No Early Stopping:**
- Problem: Training runs fixed 140 epochs regardless of validation plateau
- Blocks: Training wastes time if model converges at epoch 50; no way to stop early without manual intervention
- Improvement: Add patience-based early stopping; save best checkpoint and restore at end

**No Hyperparameter Search:**
- Problem: Learning rate schedule, optimizer, batch size are hardcoded in config YAML
- Blocks: Cannot easily search for better hyperparameters; ensemble weights are hand-tuned (0.6/0.6/0.4/0.4)
- Improvement: Implement random search or grid search over learning rate, weight decay, ensemble weights

**No Class Weighting:**
- Problem: CrossEntropyLoss is unweighted despite 59:1 imbalance in validation set
- Blocks: False negatives (missed falls) are not penalized during training; only mitigated by ensemble
- Improvement: Implement `loss_weight = [len(non_fall) / len(fall) for each class]` and pass to CrossEntropyLoss

## Test Coverage Gaps

**No Unit Tests:**
- What's not tested:
  - `Shift_gcn` forward/backward in isolation
  - `Shift_tcn` temporal dimensions
  - `Feeder` data loading and augmentation functions
  - graph topology (edge existence, connectivity)
  - rotation/normalization preprocessing functions
- Files: `model/shift_gcn.py`, `feeders/feeder.py`, `data_gen/preprocess.py`, `graph/*.py`
- Risk: Silent model behavior changes when modifying forward() or graph structure
- Priority: HIGH — Add pytest suite for all module classes

**No Integration Tests for Different Datasets:**
- What's not tested: Cross-dataset compatibility (xsub vs xview, NTU vs MediaPipe)
- Files: `config/nturgbd-cross-subject/`, `config/nturgbd-cross-view/`
- Risk: Config changes may break NTU RGB-D training without being caught
- Priority: MEDIUM — Add CI tests that train each config on a small data subset

**No Multi-GPU Tests:**
- What's not tested: DataParallel with `device=[0,1]`; distributed training
- Files: `main.py:294-299`
- Risk: Code path for multi-GPU is untested; may fail at scale
- Priority: MEDIUM — Add test that instantiates model with 2 GPUs and runs forward/backward

**No Robustness Tests:**
- What's not tested:
  - Corrupted pickle files (early EOF)
  - Missing data files (graceful error, not crash)
  - Malformed config YAML
  - Out-of-memory scenarios
  - Interrupted training recovery (`--resume`)
- Files: All of `main.py`
- Risk: Production failures when infrastructure is unreliable
- Priority: MEDIUM — Add tests for error scenarios and recovery paths

**No Inference Tests:**
- What's not tested: Loaded checkpoint inference behavior; numerical stability across different input ranges
- Files: `main.py:534-546` (test phase)
- Risk: Models may not generalize to real-world skeleton data if not tested on varied inputs
- Priority: LOW — Add synthetic test set with edge cases (missing joints, very short sequences)

---

*Concerns audit: 2026-02-24*
