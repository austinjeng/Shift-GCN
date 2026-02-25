# Inference Pipeline Design — Sliding Window Fall Detection

**Date:** 2026-02-25
**Status:** Approved

## Overview

A single Python script (`inference_pipeline.py`) with a Tkinter GUI that takes any video file, runs sliding-window fall detection using all 4 trained Shift-GCN models, and outputs a JSON report + annotated video with skeleton overlays and a confidence bar.

## Interface

### GUI (default)

Tkinter window with:
- File pickers for input video and output directory
- Parameter controls: window size (300), stride (150), threshold (0.5 slider)
- Checkpoint path fields (auto-detected from `save_models/`, with Browse buttons)
- Progress bar with stage-aware status label
- "Run Analysis" button; on completion, summary + option to open output folder

### CLI fallback

```bash
python inference_pipeline.py --cli --video path/to/video.mp4 \
    --output_dir ./inference_output \
    --window_size 300 --stride 150 --threshold 0.5
```

All GUI parameters available as CLI args. Checkpoint paths auto-detected if not specified.

## Processing Pipeline

### Step 1 — Extract Landmarks

- MediaPipe Pose on every frame (`model_complexity=1`, `min_detection_confidence=0.5`)
- `pose_world_landmarks` (3D metric) for model input
- `pose_landmarks` (pixel coords) stored separately for skeleton rendering
- Failed frames → zeros

### Step 2 — Pre-normalize

Same as training data generation (`data_gen/mediapipe_gendata.py`):
- Forward-fill zero frames
- Center at hip midpoint `[23, 24]`
- Z-axis alignment `[23→11]`
- X-axis alignment `[12→11]`

### Step 3 — Sliding Windows

- Slice `(3, T_total, 33, 1)` into 300-frame windows with configured stride
- Last window zero-padded if extends beyond video
- Videos < 300 frames → single padded window, flagged if < 150 real frames

### Step 4 — Derive Modalities (per window)

- **Joint:** normalized data as-is
- **Bone:** `bone[v] = joint[v] - joint[parent[v]]` along spanning tree
- **Joint motion:** `motion[t] = joint[t+1] - joint[t]`, last frame zeroed
- **Bone motion:** `motion[t] = bone[t+1] - bone[t]`, last frame zeroed

### Step 5 — Model Inference + Ensemble

- Run each window through all 4 models → softmax scores
- Ensemble: `0.6*joint + 0.6*bone + 0.4*joint_motion + 0.4*bone_motion`
- Produces one fall-class score per window

### Step 6 — Per-frame Aggregation

- Map each window's ensemble score to its frame range
- Average overlapping scores per frame → smooth confidence curve
- Threshold to find contiguous fall intervals

### Step 7 — Render Annotated Video

- Re-read original video
- Overlay: MediaPipe skeleton (pixel-space landmarks)
- Bottom confidence bar: green (<0.3) → yellow (0.3–0.5) → red (>0.5) with playhead
- Status strip: frame number, timestamp, fall/safe label, confidence
- Semi-transparent red tint on fall interval frames
- Output codec: H.264 (`mp4v`), same FPS as input

## Output Format

### JSON (`output_dir/results.json`)

```json
{
  "video_path": "path/to/video.mp4",
  "video_info": {
    "total_frames": 1200,
    "fps": 30.0,
    "duration_seconds": 40.0,
    "resolution": [1920, 1080]
  },
  "parameters": {
    "window_size": 300,
    "stride": 150,
    "threshold": 0.5,
    "ensemble_weights": [0.6, 0.6, 0.4, 0.4],
    "num_windows": 7
  },
  "detections": [
    {
      "start_frame": 156,
      "end_frame": 234,
      "start_time": "0:05.20",
      "end_time": "0:07.80",
      "mean_confidence": 0.87,
      "peak_confidence": 0.95,
      "peak_frame": 192
    }
  ],
  "per_frame_scores": [0.02, 0.03, ...],
  "flags": [],
  "summary": "1 fall detected at 0:05.20-0:07.80 (confidence: 0.87)"
}
```

### Annotated Video (`output_dir/annotated_video.mp4`)

Each frame: original + skeleton + confidence bar + status strip + fall tint.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Structure | Single script | Matches project style, no premature abstraction |
| Interface | Tkinter GUI + CLI | Zero extra dependencies, native Windows file pickers |
| Window size | 300 frames | Matches training exactly |
| Short video | Zero-pad, flag < 150 | Preserves model behavior, warns user |
| Aggregation | Per-frame averaging | Standard TAD approach, best temporal precision |
| Modalities | Derived on-the-fly | No intermediate files needed |
| Skeleton drawing | Pixel-space landmarks | World-space for model, pixel-space for rendering |

## Dependencies

All already in `goldcoin` conda env — no new installs:
- torch, mediapipe, opencv-python, numpy, tkinter (built-in)

## Files

| File | Purpose |
|---|---|
| `inference_pipeline.py` | Entire pipeline — GUI, extraction, inference, rendering |
