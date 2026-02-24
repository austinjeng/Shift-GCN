# Shift-GCN Fall Detection — Training Presentation

> **Slide content for PowerPoint preparation**
> 15–25 slides | Full pipeline overview | Mixed audience
> Date: February 2026

---

## Slide 1: Title Slide

**Fall Detection Using Shift-GCN with MediaPipe Pose**

Skeleton-Based Action Recognition on NTU RGB+D

February 2026

> *Visual: Project logo or a video frame with skeleton overlay*

---

## Slide 2: Agenda

1. Problem & Motivation
2. Dataset — NTU RGB+D
3. Skeleton Extraction — MediaPipe Pose
4. Data Pipeline — Four Modalities
5. Model Architecture — Shift-GCN
6. Training Configuration
7. Results — Individual Models
8. Results — Ensemble
9. Key Observations
10. Future Work

> *Visual: Simple numbered list, highlight current section as you progress*

---

## Slide 3: Problem Statement

**Goal:** Detect "falling down" events from RGB video

- Falls are a leading cause of injury, especially for elderly populations
- Real-time detection enables faster emergency response
- Challenge: Distinguish falls from similar actions (bending, sitting down, lying down)

**Our Approach:**
- Skeleton-based recognition (not raw video pixels)
- Binary classification: **Fall** vs. **Non-Fall**

> *Visual: Side-by-side images showing a fall vs. similar actions (bending, sitting)*

---

## Slide 4: Why Skeleton-Based?

| Approach | Pros | Cons |
|----------|------|------|
| **RGB/Video** | Rich appearance info | Privacy concerns, lighting-sensitive, large models |
| **Optical Flow** | Captures motion | Expensive to compute, background noise |
| **Skeleton-Based** | Privacy-preserving, compact, robust to appearance | Requires pose estimation |

**Skeleton advantages for fall detection:**
- Invariant to clothing, lighting, camera angle
- Compact representation (33 points vs. millions of pixels)
- Captures the structural change that defines a fall

> *Visual: Diagram showing RGB frame → skeleton extraction → compact representation*

---

## Slide 5: NTU RGB+D Dataset

**The largest skeleton-based action recognition dataset**

- **56,880** RGB video clips
- **60** action classes (A001–A060)
- **40** subjects performing actions
- **3** camera viewpoints per setup
- Captured with Microsoft Kinect v2

**Our target:** Action A043 — "Falling Down" (948 videos total)

**Benchmark split:** Cross-subject (xsub)
- 20 training subjects / 20 validation subjects
- Ensures model generalizes across different people

> *Visual: Grid of example action thumbnails from NTU dataset, A043 highlighted*

---

## Slide 6: MediaPipe Pose — Skeleton Extraction

**Google MediaPipe Pose: 33 landmarks per frame**

- Real-time pose estimation from single RGB camera
- No depth sensor required (unlike Kinect skeleton)
- 33 body landmarks covering face, arms, torso, and legs
- Outputs (x, y, z) coordinates per landmark per frame

**Key landmarks:**
- Head: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists, hands
- Lower body: hips, knees, ankles, feet

> *Visual: MediaPipe Pose landmark diagram (the official 33-point skeleton figure)*

---

## Slide 7: Skeleton Graph Topology

**33 nodes connected by 32 edges forming a spanning tree**

- Rooted at NOSE (landmark 0)
- Bridge edges connect head → torso → limbs
- Bidirectional edges (inward + outward) + self-loops = **97 total edges**

**Preprocessing:**
- Center joint: Hip midpoint (landmarks 23, 24)
- Frame length: Padded/truncated to **300 frames**
- Coordinate normalization along spine axis

> *Visual: Tree diagram of the 33-landmark skeleton graph with labeled body parts*

---

## Slide 8: Data Pipeline Overview

```
NTU RGB+D Videos (56,880 clips)
        │
        ▼
  MediaPipe Pose Extraction
  (33 landmarks × 300 frames × 3 coords)
        │
        ├──→ Joint Data (raw xyz positions)
        │        │
        │        ├──→ Joint Motion (frame differences)
        │        │
        ├──→ Bone Data (parent-child vectors)
        │        │
        │        ├──→ Bone Motion (frame differences)
        │
        ▼
  4 Modalities → 4 Models → Ensemble
```

> *Visual: Flowchart diagram of the data pipeline*

---

## Slide 9: Four Modalities Explained

| Modality | What It Captures | Intuition |
|----------|-----------------|-----------|
| **Joint** | Raw (x, y, z) positions | Where is each body part? |
| **Bone** | Parent-child difference vectors | How are limbs oriented? |
| **Joint Motion** | Frame-to-frame position change | How fast is each joint moving? |
| **Bone Motion** | Frame-to-frame bone change | How are limb orientations changing? |

**Data shape:** `(N, 3, 300, 33, 1)` — (samples, channels, frames, joints, persons)

**Why four modalities?**
- Each captures different aspects of motion
- Ensemble of four complementary views → stronger prediction

> *Visual: Four skeleton visualizations showing each modality (positions, bones, velocity arrows, acceleration arrows)*

---

## Slide 10: Dataset Statistics

### Training Set (Class-Balanced)

| Class | Samples | Ratio |
|-------|---------|-------|
| Fall | 672 | 25% |
| Non-Fall | 2,016 | 75% |
| **Total** | **2,688** | **1:3** |

### Validation Set (Natural Distribution)

| Class | Samples | Ratio |
|-------|---------|-------|
| Fall | 276 | 1.67% |
| Non-Fall | 16,284 | 98.33% |
| **Total** | **16,560** | **1:59** |

**Training balance strategy:** 3:1 subsampling ratio (non-fall:fall)
**Validation:** Unmodified natural distribution — tests real-world performance

> *Visual: Two pie charts comparing train vs. validation class distribution*

---

## Slide 11: Shift-GCN Architecture

**Key Innovation: Replace graph convolution with shift operations**

Traditional GCN:
- Learns adjacency matrix weights
- Matrix multiplication over graph structure
- Heavy computation

**Shift-GCN:**
- Channel-wise circular shifts across joints
- No learnable adjacency matrix needed
- **Fewer parameters, faster computation, comparable accuracy**

> *Visual: Diagram comparing traditional GCN message passing vs. Shift-GCN channel shifts*

---

## Slide 12: Network Architecture

```
Input: (N, 3, 300, 33) → Batch Normalization
                │
    ┌───────────┴───────────┐
    │  10 TCN-GCN Blocks    │
    │                       │
    │  Blocks 1–4:   64ch   │  ← spatial-temporal feature extraction
    │  Blocks 5–7:  128ch   │  ← temporal downsampling (stride=2)
    │  Blocks 8–10: 256ch   │  ← temporal downsampling (stride=2)
    └───────────┬───────────┘
                │
    Global Average Pooling
                │
    Fully Connected (256 → 2)
                │
         Fall / Non-Fall
```

**Model size:** ~720K parameters | 6.3 MB per checkpoint

> *Visual: Vertical block diagram of the 10-layer network with channel dimensions labeled*

---

## Slide 13: Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD + Nesterov Momentum |
| Learning Rate | 0.1 → 0.01 → 0.001 → 0.0001 |
| LR Schedule | Step decay at epochs 60, 80, 100 |
| Batch Size | 64 |
| Total Epochs | 140 |
| Weight Decay | 0.0001 |
| Loss Function | Cross-Entropy (unweighted) |
| Data Augmentation | None |

**Training per model:** ~2 hours on single NVIDIA GPU
**Total training time:** ~9 hours (4 models sequentially)

> *Visual: Learning rate schedule plot showing step decay at epochs 60, 80, 100*

---

## Slide 14: Training Convergence

**All models converge rapidly — >98% accuracy within 15 epochs**

| Epoch | Joint | Bone | Joint Motion | Bone Motion |
|-------|-------|------|--------------|-------------|
| 5 | 96.18% | 97.55% | 98.52% | 98.43% |
| 15 | 98.51% | 99.24% | 98.94% | 99.35% |
| 60 | **99.49%** | 99.46% | 99.37% | 99.21% |
| 140 | 99.32% | 99.44% | 99.21% | 99.42% |

Best accuracy per model is reached early (epochs 35–60), then plateaus.

> *Visual: Line chart — 4 curves (one per modality) showing validation accuracy over 140 epochs, with vertical dashed lines at LR decay epochs 60, 80, 100*

---

## Slide 15: Individual Model Results

| Modality | Best Accuracy | Best Epoch |
|----------|:------------:|:----------:|
| Joint | 99.49% | 60 |
| Bone | 99.51% | 35 |
| Joint Motion | 99.46% | 40 |
| **Bone Motion** | **99.64%** | **45** |

**Observation:** Bone-based features slightly outperform joint-based features
- Relative skeletal structure is more discriminative than absolute position for fall detection
- Motion modalities show more variance (frame differencing amplifies noise)

> *Visual: Horizontal bar chart of the four model accuracies*

---

## Slide 16: Ensemble Method

**Weighted Score Fusion**

```
Score = 0.6 × Joint + 0.6 × Bone + 0.4 × Joint Motion + 0.4 × Bone Motion

Prediction = argmax(Score)
```

| Modality | Weight | Role |
|----------|:------:|------|
| Joint | 0.6 | Primary — spatial position |
| Bone | 0.6 | Primary — relative structure |
| Joint Motion | 0.4 | Supplementary — temporal dynamics |
| Bone Motion | 0.4 | Supplementary — structural dynamics |

**Why these weights?**
- Spatial modalities (joint, bone) are more stable → higher weight
- Motion modalities add complementary temporal info → lower weight

> *Visual: Diagram showing 4 model outputs being combined with weighted sum into final prediction*

---

## Slide 17: Ensemble Results

### Headline Numbers

| Metric | Value |
|--------|:-----:|
| **Top-1 Accuracy** | **99.77%** |
| Top-5 Accuracy | 100.00% |
| Fall Precision | 95.77% |
| Fall Recall | 90.22% |
| **Fall F1-Score** | **92.91%** |

### Ensemble vs. Best Single Model

| | Accuracy | Errors (out of 16,560) |
|---|:---:|:---:|
| Best single (Bone Motion) | 99.64% | ~60 |
| **Ensemble** | **99.77%** | **38** |

**Ensemble reduces errors by 37%**

> *Visual: Large bold number "99.77%" centered, with comparison bar chart below*

---

## Slide 18: Confusion Matrix

|  | Predicted Non-Fall | Predicted Fall |
|:--|:---:|:---:|
| **Actual Non-Fall** | 16,273 (TN) | 11 (FP) |
| **Actual Fall** | 27 (FN) | 249 (TP) |

**Key metrics:**
- **249 / 276** falls correctly detected (90.2% recall)
- Only **11** false alarms out of 16,284 non-falls (0.07% false alarm rate)
- **27** falls missed (9.8% miss rate)

> *Visual: Color-coded 2×2 confusion matrix heatmap (green for TN/TP, red for FP/FN)*

---

## Slide 19: Error Analysis

**Where does the model fail?**

- **27 missed falls (FN):** Likely slow or partial falls that resemble sitting/bending
- **11 false alarms (FP):** Non-fall actions with similar skeletal trajectories

**The class imbalance challenge:**
- Validation set is 59:1 (non-fall:fall)
- Despite this extreme imbalance, 90% recall on the minority class
- The 27 missed falls are the primary concern for safety applications

> *Visual: Examples of misclassified samples if available, or a pie chart of error types*

---

## Slide 20: Training Timeline

| Model | Duration | Cumulative |
|-------|:--------:|:----------:|
| Joint | 2h 23m | 2h 23m |
| Bone | 2h 12m | 4h 35m |
| Joint Motion | 2h 13m | 6h 48m |
| Bone Motion | 2h 05m | 8h 53m |

**Total: ~9 hours** on single GPU (sequential training)

- 70 checkpoints saved per model (every 2 epochs)
- 6.3 MB per checkpoint → 1.76 GB total checkpoint storage

> *Visual: Gantt chart or stacked timeline showing the 4 training sessions*

---

## Slide 21: Key Observations

1. **Rapid convergence** — All models reach >98% within 15 epochs; remaining epochs refine by ~1%

2. **Bone > Joint** — Relative skeletal structure is more discriminative than absolute position for fall detection

3. **Ensemble works** — 37% error reduction by combining four complementary views

4. **No augmentation needed** — 99.77% achieved without any data augmentation, suggesting the skeleton representation is already robust

5. **LR decay matters** — Visible accuracy jumps after each learning rate step (epochs 60, 80, 100)

> *Visual: Icons or bullet points with emphasis styling*

---

## Slide 22: Pipeline Summary

| Stage | Input | Output | Tool |
|-------|-------|--------|------|
| Video Collection | RGB cameras | 56,880 clips | NTU RGB+D |
| Pose Extraction | RGB frames | 33 landmarks/frame | MediaPipe |
| Data Generation | Landmarks | 4 modality .npy files | Custom scripts |
| Model Training | .npy data | 4 trained models | Shift-GCN + CUDA |
| Ensemble Eval | 4 model outputs | Final prediction | Weighted fusion |

**End-to-end:** Video → Skeleton → Four modalities → Four models → Ensemble → Fall/Non-Fall

> *Visual: Horizontal pipeline flowchart with icons for each stage*

---

## Slide 23: Future Work

**Improving recall (reducing missed falls):**
- Threshold tuning — adjust decision boundary for higher recall at cost of some precision
- Class-weighted loss — penalize missed falls more heavily during training
- Data augmentation — enable random shift/move to increase training diversity

**Scaling the system:**
- Optimize ensemble weights — grid search or learned combination
- Cross-view evaluation — test generalization across camera angles (xview benchmark)
- Real-time deployment — integrate MediaPipe + Shift-GCN for live camera inference

> *Visual: Roadmap-style diagram with short-term and long-term items*

---

## Slide 24: Summary

| | |
|---|---|
| **Task** | Binary fall detection from skeleton data |
| **Dataset** | NTU RGB+D (2,688 train / 16,560 val) |
| **Model** | Shift-GCN (4 modalities, weighted ensemble) |
| **Accuracy** | **99.77%** Top-1 |
| **Fall F1** | **92.91%** |
| **Training** | ~9 hours total, single GPU |

**Key takeaway:** Skeleton-based fall detection achieves near-perfect accuracy with a lightweight model (~720K params), while preserving privacy and being robust to visual conditions.

> *Visual: Clean summary table, possibly with the headline accuracy number large and centered*

---

## Slide 25: Q&A

**Questions?**

**Resources:**
- Training Report: `TRAINING_REPORT.md`
- Model Code: `model/shift_gcn.py`
- Ensemble Script: `ensemble_mediapipe.py`

> *Visual: Clean slide with contact info or project repository link*

---

## Appendix Slides (Optional)

### A1: MediaPipe Landmark Index

| ID | Landmark | ID | Landmark |
|----|----------|----|----------|
| 0 | Nose | 17 | Left Pinky |
| 1–4 | Left/Right Eye | 18 | Right Pinky |
| 5–6 | Left/Right Ear | 19 | Left Index |
| 7–8 | Left/Right Mouth | 20 | Right Index |
| 9–10 | Mouth Left/Right | 21 | Left Thumb |
| 11 | Left Shoulder | 22 | Right Thumb |
| 12 | Right Shoulder | 23 | Left Hip |
| 13 | Left Elbow | 24 | Right Hip |
| 14 | Right Elbow | 25 | Left Knee |
| 15 | Left Wrist | 26 | Right Knee |
| 16 | Right Wrist | 27–32 | Ankles, Heels, Toes |

### A2: Full Epoch-by-Epoch Accuracy Table

*(Include the detailed tables from TRAINING_REPORT.md if asked)*

### A3: Tensor Shape Reference

```
(N, C, T, V, M) = (samples, channels=3, frames=300, joints=33, persons=1)
```
