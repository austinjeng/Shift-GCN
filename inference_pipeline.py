import argparse
import glob
import json
import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Bone pairs (0-indexed): (joint, parent). Root=0 (NOSE) self-references.
# Matches data_gen/gen_bone_data_mediapipe.py (converted from 1-indexed).
BONE_PAIRS = [
    (0, 0), (1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 3),
    (8, 6), (9, 0), (10, 9), (11, 0), (12, 11), (13, 11), (14, 12),
    (15, 13), (16, 14), (17, 15), (18, 16), (19, 15), (20, 16),
    (21, 15), (22, 16), (23, 11), (24, 12), (25, 23), (26, 24),
    (27, 25), (28, 26), (29, 27), (30, 28), (31, 27), (32, 28),
]

ENSEMBLE_WEIGHTS_DEFAULT = [0.6, 0.6, 0.4, 0.4]
MODALITIES = ['joint', 'bone', 'joint_motion', 'bone_motion']


def auto_detect_checkpoint(modality, save_dir='./save_models'):
    """Find the checkpoint with the highest epoch for a given modality."""
    pattern = os.path.join(save_dir, f'mediapipe_ShiftGCN_{modality}-*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    def epoch_of(f):
        base = os.path.splitext(os.path.basename(f))[0]
        parts = base.rsplit('-', 2)
        return int(parts[-2]) if len(parts) >= 3 else 0
    return max(files, key=epoch_of)


def parse_args():
    parser = argparse.ArgumentParser(description='Shift-GCN Fall Detection Inference Pipeline')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode (no GUI)')
    parser.add_argument('--video', type=str, default=None, help='Input video path')
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--stride', type=int, default=150)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--ensemble_weights', type=str, default='0.6,0.6,0.4,0.4')
    parser.add_argument('--weights_joint', type=str, default=None)
    parser.add_argument('--weights_bone', type=str, default=None)
    parser.add_argument('--weights_joint_motion', type=str, default=None)
    parser.add_argument('--weights_bone_motion', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./save_models')
    return parser.parse_args()


def extract_landmarks(video_path, progress_callback=None):
    """Extract MediaPipe Pose landmarks from all frames of a video.

    Returns dict with:
        'world': np.ndarray (3, T, 33, 1) -- world-space 3D coords
        'pixel': list[np.ndarray | None] -- per-frame (33, 3) pixel coords (x,y,visibility)
        'fps': float
        'total_frames': int
        'resolution': (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    world_seq = []
    pixel_seq = []

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks:
                world_joints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                world_joints = np.zeros((33, 3), dtype=np.float32)

            if results.pose_landmarks:
                pixel_joints = np.array(
                    [[lm.x * width, lm.y * height, lm.visibility]
                     for lm in results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                pixel_joints = None

            world_seq.append(world_joints)
            pixel_seq.append(pixel_joints)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    cap.release()

    if len(world_seq) == 0:
        raise RuntimeError(f'No frames read from video: {video_path}')

    # world_seq: list of (33, 3) -> stack to (T, 33, 3) -> transpose to (3, T, 33) -> add M dim
    world_data = np.stack(world_seq, axis=0).transpose(2, 0, 1)  # (3, T, 33)
    world_data = world_data[:, :, :, np.newaxis]  # (3, T, 33, 1)

    return {
        'world': world_data,
        'pixel': pixel_seq,
        'fps': fps,
        'total_frames': len(world_seq),
        'resolution': (width, height),
    }


# ---------------------------------------------------------------------------
# Pre-normalization (inlined from data_gen/rotation.py + data_gen/preprocess.py)
# Must match the training pipeline EXACTLY.
# MediaPipe args: zaxis=[23, 11], xaxis=[12, 11], center_joint=[23, 24]
# ---------------------------------------------------------------------------

def _rotation_matrix(axis, theta):
    """Rotation matrix for counterclockwise rotation about axis by theta radians."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _angle_between(v1, v2):
    """Angle in radians between vectors v1 and v2."""
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pre_normalize(data):
    """Normalize skeleton data -- matches data_gen/preprocess.py exactly.

    Uses MediaPipe parameters:
        zaxis=[23, 11] (LEFT_HIP -> LEFT_SHOULDER)
        xaxis=[12, 11] (RIGHT_SHOULDER -> LEFT_SHOULDER)
        center_joint=[23, 24] (hip midpoint)

    Args:
        data: np.ndarray (N, 3, T, 33, 1)
    Returns:
        Normalized data, same shape.
    """
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # (N, M, T, V, C)

    # 1. Pad null frames with previous frames
    for i_s, skeleton in enumerate(s):
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    # 2. Center at hip midpoint (joints 23, 24)
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = np.mean(
            [skeleton[0][:, j:j+1, :] for j in [23, 24]], axis=0
        ).copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    # 3. Z-axis alignment: LEFT_HIP(23) -> LEFT_SHOULDER(11)
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, 23]
        joint_top = skeleton[0, 0, 11]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = _angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = _rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0)
            s[i_s, i_p, mask] = np.dot(person[mask], matrix_z.T)

    # 4. X-axis alignment: RIGHT_SHOULDER(12) -> LEFT_SHOULDER(11)
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, 12]
        joint_lshoulder = skeleton[0, 0, 11]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = _angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = _rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0)
            s[i_s, i_p, mask] = np.dot(person[mask], matrix_x.T)

    return np.transpose(s, [0, 4, 2, 3, 1])


# ---------------------------------------------------------------------------
# Sliding window + modality derivation
# ---------------------------------------------------------------------------

def create_sliding_windows(data, window_size=300, stride=150):
    """Slice (3, T, 33, 1) into overlapping windows of (3, window_size, 33, 1).

    Returns list of (window_data, start_frame, end_frame, num_real_frames).
    """
    C, T, V, M = data.shape
    windows = []

    if T <= window_size:
        padded = np.zeros((C, window_size, V, M), dtype=np.float32)
        padded[:, :T, :, :] = data
        windows.append((padded, 0, T, T))
        return windows

    start = 0
    while start < T:
        end = start + window_size
        if end <= T:
            w = data[:, start:end, :, :].copy()
            windows.append((w, start, end, window_size))
        else:
            padded = np.zeros((C, window_size, V, M), dtype=np.float32)
            real = T - start
            padded[:, :real, :, :] = data[:, start:T, :, :]
            windows.append((padded, start, T, real))
        start += stride
        if end >= T:
            break

    return windows


def derive_modalities(joint_window):
    """Derive all 4 modalities from a joint window (3, T, 33, 1).

    Returns dict: {'joint', 'bone', 'joint_motion', 'bone_motion'}.
    """
    C, T, V, M = joint_window.shape

    # Bone: joint-to-parent differences (matches gen_bone_data_mediapipe.py)
    bone = np.zeros_like(joint_window)
    for v, parent in BONE_PAIRS:
        bone[:, :, v, :] = joint_window[:, :, v, :] - joint_window[:, :, parent, :]

    # Joint motion: frame differencing (matches gen_motion_data_mediapipe.py)
    joint_motion = np.zeros_like(joint_window)
    joint_motion[:, :T-1, :, :] = joint_window[:, 1:, :, :] - joint_window[:, :T-1, :, :]

    # Bone motion: frame differencing on bone data
    bone_motion = np.zeros_like(bone)
    bone_motion[:, :T-1, :, :] = bone[:, 1:, :, :] - bone[:, :T-1, :, :]

    return {
        'joint': joint_window,
        'bone': bone,
        'joint_motion': joint_motion,
        'bone_motion': bone_motion,
    }


# ---------------------------------------------------------------------------
# Model loading + ensemble inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path):
    """Load a Shift-GCN model from checkpoint."""
    from collections import OrderedDict
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.shift_gcn import Model

    model = Model(
        num_class=2,
        num_point=33,
        num_person=1,
        graph='graph.mediapipe_pose.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
    ).cuda()

    weights = torch.load(checkpoint_path, weights_only=False)
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    weights = OrderedDict(
        [[k.split('module.')[-1], v.cuda()] for k, v in weights.items()]
    )
    model.load_state_dict(weights)
    model.eval()
    return model


def run_ensemble_inference(windows, models, ensemble_weights, progress_callback=None):
    """Run 4-model ensemble inference on all windows.

    Args:
        windows: list of (window_data, start, end, num_real) from create_sliding_windows.
        models: dict mapping modality name -> loaded Model.
        ensemble_weights: list of 4 floats [joint, bone, jm, bm].
        progress_callback: Optional callable(current_window, total_windows).

    Returns:
        list of (fall_score, start_frame, end_frame, num_real_frames) per window.
    """
    results = []
    with torch.no_grad():
        for i, (window_data, start, end, num_real) in enumerate(windows):
            mods = derive_modalities(window_data)
            ensemble_logits = np.zeros(2, dtype=np.float64)
            for mod_name, alpha in zip(MODALITIES, ensemble_weights):
                x = torch.from_numpy(mods[mod_name]).unsqueeze(0).float().cuda()
                # x shape: (1, 3, 300, 33, 1)
                logits = models[mod_name](x).cpu().numpy()[0]  # raw logits, no softmax
                ensemble_logits += alpha * logits
            # Softmax on final ensemble logits to get confidence
            exp_scores = np.exp(ensemble_logits - ensemble_logits.max())
            fall_score = float(exp_scores[1] / exp_scores.sum())  # class 1 = fall
            results.append((fall_score, start, end, num_real))
            if progress_callback:
                progress_callback(i + 1, len(windows))
    return results


# ---------------------------------------------------------------------------
# Per-frame aggregation + fall detection
# ---------------------------------------------------------------------------

def aggregate_per_frame(window_results, total_frames):
    """Map window scores to per-frame scores by averaging overlapping windows."""
    score_sum = np.zeros(total_frames, dtype=np.float64)
    score_count = np.zeros(total_frames, dtype=np.float64)
    for fall_score, start, end, num_real in window_results:
        real_end = start + num_real
        score_sum[start:real_end] += fall_score
        score_count[start:real_end] += 1.0
    score_count = np.maximum(score_count, 1.0)
    return score_sum / score_count


def detect_fall_intervals(per_frame_scores, threshold, fps):
    """Find contiguous regions where per-frame score exceeds threshold."""
    above = per_frame_scores > threshold
    detections = []
    in_region = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_region:
            start = i
            in_region = True
        elif not above[i] and in_region:
            _add_detection(detections, per_frame_scores, start, i, fps)
            in_region = False
    if in_region:
        _add_detection(detections, per_frame_scores, start, len(above), fps)
    return detections


def _add_detection(detections, scores, start, end, fps):
    """Helper to create a detection dict for a contiguous fall region."""
    region_scores = scores[start:end]
    peak_idx = int(np.argmax(region_scores))
    def fmt_time(frame):
        secs = frame / fps
        mins = int(secs // 60)
        secs = secs % 60
        return f'{mins}:{secs:05.2f}'
    detections.append({
        'start_frame': int(start),
        'end_frame': int(end),
        'start_time': fmt_time(start),
        'end_time': fmt_time(end),
        'mean_confidence': float(np.mean(region_scores)),
        'peak_confidence': float(np.max(region_scores)),
        'peak_frame': int(start + peak_idx),
    })


# ---------------------------------------------------------------------------
# JSON report generation
# ---------------------------------------------------------------------------

def generate_report(video_path, video_info, params, per_frame_scores, detections, flags):
    """Generate the JSON results report."""
    n_det = len(detections)
    if n_det == 0:
        summary = 'No falls detected.'
    elif n_det == 1:
        d = detections[0]
        summary = (f'1 fall detected at {d["start_time"]}-{d["end_time"]} '
                   f'(confidence: {d["mean_confidence"]:.2f})')
    else:
        parts = [f'{d["start_time"]}-{d["end_time"]}' for d in detections]
        summary = f'{n_det} falls detected at: {", ".join(parts)}'
    return {
        'video_path': os.path.abspath(video_path),
        'video_info': video_info,
        'parameters': params,
        'detections': detections,
        'per_frame_scores': [round(float(s), 4) for s in per_frame_scores],
        'flags': flags,
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# Annotated video renderer
# ---------------------------------------------------------------------------

# MediaPipe Pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # torso + arms
    (11, 23), (12, 24), (23, 24),                        # torso to hips
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (27, 29), (27, 31), (28, 30), (28, 32),              # feet
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),     # face
    (3, 7), (6, 8), (9, 10),                              # ears + mouth
]


def confidence_color(score):
    """Map score [0,1] to BGR color: green(0) -> yellow(0.3) -> red(0.5+)."""
    if score < 0.3:
        # Green to yellow
        t = score / 0.3
        return (0, int(255 * (1 - t * 0.5)), int(255 * t))
    elif score < 0.5:
        # Yellow to red
        t = (score - 0.3) / 0.2
        return (0, int(255 * (1 - t)), int(255))
    else:
        # Red
        return (0, 0, 255)


def render_annotated_video(video_path, output_path, pixel_landmarks, per_frame_scores,
                           detections, fps, threshold, progress_callback=None):
    """Render annotated output video with skeleton, confidence bar, and fall tinting."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(per_frame_scores)

    bar_height = 40
    status_height = 30
    out_height = height + bar_height + status_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, out_height))

    # Precompute fall regions as set of frames for fast lookup
    fall_frames = set()
    for d in detections:
        for f in range(d['start_frame'], d['end_frame']):
            fall_frames.add(f)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        score = per_frame_scores[frame_idx]
        is_fall = frame_idx in fall_frames

        # 1. Draw skeleton
        pl = pixel_landmarks[frame_idx]
        if pl is not None:
            color = (0, 0, 255) if is_fall else (0, 255, 0)
            for (a, b) in POSE_CONNECTIONS:
                if pl[a][2] > 0.3 and pl[b][2] > 0.3:  # visibility threshold
                    pt1 = (int(pl[a][0]), int(pl[a][1]))
                    pt2 = (int(pl[b][0]), int(pl[b][1]))
                    cv2.line(frame, pt1, pt2, color, 2)
            for j in range(33):
                if pl[j][2] > 0.3:
                    cv2.circle(frame, (int(pl[j][0]), int(pl[j][1])), 3, color, -1)

        # 2. Fall tint (semi-transparent red overlay)
        if is_fall:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 3. Build output frame with bar and status
        out_frame = np.zeros((out_height, width, 3), dtype=np.uint8)
        out_frame[:height, :, :] = frame

        # 4. Confidence bar
        bar_y = height
        for x in range(width):
            bar_frame_idx = int(x / width * total_frames)
            bar_frame_idx = min(bar_frame_idx, total_frames - 1)
            s = per_frame_scores[bar_frame_idx]
            c = confidence_color(s)
            cv2.line(out_frame, (x, bar_y), (x, bar_y + bar_height), c, 1)

        # Playhead marker
        px = int(frame_idx / max(total_frames - 1, 1) * (width - 1))
        cv2.line(out_frame, (px, bar_y), (px, bar_y + bar_height), (255, 255, 255), 2)

        # 5. Status strip
        status_y = bar_y + bar_height
        cv2.rectangle(out_frame, (0, status_y), (width, out_height), (30, 30, 30), -1)
        secs = frame_idx / fps
        time_str = f'{int(secs // 60)}:{secs % 60:05.2f}'
        label = 'FALL' if is_fall else 'SAFE'
        label_color = (0, 0, 255) if is_fall else (0, 255, 0)
        text = f'Frame {frame_idx}/{total_frames}  |  {time_str}  |  {label}  conf: {score:.2f}'
        cv2.putText(out_frame, text, (10, status_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

        writer.write(out_frame)

        if progress_callback:
            progress_callback(frame_idx + 1, total_frames)

    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(video_path, output_dir, window_size, stride, threshold,
                 ensemble_weights, checkpoint_paths, progress_callback=None):
    """Run the full inference pipeline.

    Args:
        progress_callback: Optional callable(stage, current, total) for progress.
    Returns:
        report dict (same as JSON output).
    """
    os.makedirs(output_dir, exist_ok=True)

    def cb(stage):
        def inner(cur, tot):
            if progress_callback:
                progress_callback(stage, cur, tot)
            else:
                print(f'\r  [{stage}] {cur}/{tot}', end='', flush=True)
        return inner

    # Stage 1: Extract landmarks
    print('Stage 1/4: Extracting landmarks...')
    landmarks = extract_landmarks(video_path, progress_callback=cb('Extracting landmarks'))
    print()

    total_frames = landmarks['total_frames']
    fps = landmarks['fps']
    width, height = landmarks['resolution']

    # Stage 2: Sliding windows on RAW data (before normalization)
    print('Stage 2/4: Creating sliding windows...')
    world_data = landmarks['world']  # (3, T, 33, 1)
    windows = create_sliding_windows(world_data, window_size, stride)

    # Stage 3: Normalize each window independently + run inference
    # This matches training exactly: each sample is zero-padded to window_size,
    # then pre_normalized (which cyclically fills zero-padded frames).
    print('Stage 3/4: Normalizing windows + running ensemble inference...')
    num_windows = len(windows)
    window_batch = np.stack([w[0] for w in windows], axis=0)  # (N, 3, W, 33, 1)
    window_batch = pre_normalize(window_batch)  # normalize all windows as a batch
    windows = [(window_batch[i], w[1], w[2], w[3]) for i, w in enumerate(windows)]

    # Load models
    models = {}
    for mod_name in MODALITIES:
        cp = checkpoint_paths[mod_name]
        print(f'  Loading {mod_name} model from {os.path.basename(cp)}...')
        models[mod_name] = load_model(cp)

    window_results = run_ensemble_inference(windows, models, ensemble_weights, cb('Inference'))
    print()

    # Free GPU memory after inference
    del models
    torch.cuda.empty_cache()

    # Stage 4: Aggregate + detect + render
    print('Stage 4/4: Generating output...')
    per_frame_scores = aggregate_per_frame(window_results, total_frames)
    detections = detect_fall_intervals(per_frame_scores, threshold, fps)

    flags = []
    if total_frames < window_size // 2:
        flags.append('low_confidence_short_video')

    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'duration_seconds': round(total_frames / fps, 2),
        'resolution': [width, height],
    }
    params = {
        'window_size': window_size,
        'stride': stride,
        'threshold': threshold,
        'ensemble_weights': ensemble_weights,
        'num_windows': len(windows),
    }

    report = generate_report(video_path, video_info, params,
                             per_frame_scores, detections, flags)

    # Save JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  JSON report saved: {json_path}')

    # Render annotated video
    video_out_path = os.path.join(output_dir, 'annotated_video.mp4')
    render_annotated_video(video_path, video_out_path, landmarks['pixel'],
                           per_frame_scores, detections, fps, threshold,
                           progress_callback=cb('Rendering'))
    print(f'\n  Annotated video saved: {video_out_path}')

    print(f'\n{report["summary"]}')
    return report


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

def run_gui(args, checkpoint_paths, ensemble_weights):
    """Launch the Tkinter GUI for the inference pipeline."""
    import tkinter as tk
    from tkinter import filedialog, ttk
    import threading

    root = tk.Tk()
    root.title('Shift-GCN Fall Detection')
    root.resizable(False, False)

    # -- Variables --
    video_var = tk.StringVar(value=args.video or '')
    output_var = tk.StringVar(value=args.output_dir)
    window_var = tk.IntVar(value=args.window_size)
    stride_var = tk.IntVar(value=args.stride)
    threshold_var = tk.DoubleVar(value=args.threshold)
    cp_vars = {}
    for mod in MODALITIES:
        cp_vars[mod] = tk.StringVar(value=checkpoint_paths.get(mod, ''))

    # -- Layout --
    pad = dict(padx=8, pady=4)
    row = 0

    # Video input
    tk.Label(root, text='Video:').grid(row=row, column=0, sticky='e', **pad)
    tk.Entry(root, textvariable=video_var, width=50).grid(row=row, column=1, **pad)
    tk.Button(root, text='Browse...',
              command=lambda: video_var.set(
                  filedialog.askopenfilename(
                      filetypes=[('Video', '*.mp4 *.avi *.mkv *.mov'), ('All', '*.*')])
                  or video_var.get()
              )).grid(row=row, column=2, **pad)
    row += 1

    # Output dir
    tk.Label(root, text='Output:').grid(row=row, column=0, sticky='e', **pad)
    tk.Entry(root, textvariable=output_var, width=50).grid(row=row, column=1, **pad)
    tk.Button(root, text='Browse...',
              command=lambda: output_var.set(
                  filedialog.askdirectory() or output_var.get()
              )).grid(row=row, column=2, **pad)
    row += 1

    # Parameters section
    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=8)
    row += 1
    tk.Label(root, text='Parameters', font=('', 10, 'bold')).grid(row=row, column=0, columnspan=3)
    row += 1

    tk.Label(root, text='Window Size:').grid(row=row, column=0, sticky='e', **pad)
    tk.Spinbox(root, textvariable=window_var, from_=30, to=600, width=8).grid(row=row, column=1, sticky='w', **pad)
    row += 1

    tk.Label(root, text='Stride:').grid(row=row, column=0, sticky='e', **pad)
    tk.Spinbox(root, textvariable=stride_var, from_=1, to=600, width=8).grid(row=row, column=1, sticky='w', **pad)
    row += 1

    tk.Label(root, text='Threshold:').grid(row=row, column=0, sticky='e', **pad)
    threshold_label = tk.Label(root, text=f'{args.threshold:.2f}')
    threshold_label.grid(row=row, column=2, sticky='w', **pad)
    def on_threshold_change(val):
        threshold_var.set(round(float(val), 2))
        threshold_label.config(text=f'{float(val):.2f}')
    tk.Scale(root, from_=0.1, to=0.9, resolution=0.05, orient='horizontal',
             variable=threshold_var, command=on_threshold_change,
             showvalue=False, length=200).grid(row=row, column=1, sticky='w', **pad)
    row += 1

    # Checkpoints section
    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=8)
    row += 1
    tk.Label(root, text='Checkpoints (auto-detected)', font=('', 10, 'bold')).grid(row=row, column=0, columnspan=3)
    row += 1

    for mod in MODALITIES:
        label = mod.replace('_', ' ').title() + ':'
        tk.Label(root, text=label).grid(row=row, column=0, sticky='e', **pad)
        tk.Entry(root, textvariable=cp_vars[mod], width=50).grid(row=row, column=1, **pad)
        tk.Button(root, text='Browse...',
                  command=lambda m=mod: cp_vars[m].set(
                      filedialog.askopenfilename(filetypes=[('Checkpoint', '*.pt')]) or cp_vars[m].get()
                  )).grid(row=row, column=2, **pad)
        row += 1

    # Progress section
    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=8)
    row += 1

    progress = ttk.Progressbar(root, length=400, mode='determinate')
    progress.grid(row=row, column=0, columnspan=3, **pad)
    row += 1

    status_var = tk.StringVar(value='Ready')
    tk.Label(root, textvariable=status_var, anchor='w').grid(row=row, column=0, columnspan=3, sticky='w', **pad)
    row += 1

    # Run button
    run_btn = tk.Button(root, text='Run Analysis', font=('', 11, 'bold'))
    run_btn.grid(row=row, column=0, columnspan=3, pady=12)

    def on_run():
        video = video_var.get()
        if not video or not os.path.isfile(video):
            status_var.set('Error: Select a valid video file.')
            return

        # Validate checkpoints
        cps = {}
        for mod in MODALITIES:
            cp = cp_vars[mod].get()
            if not cp or not os.path.isfile(cp):
                status_var.set(f'Error: Checkpoint missing for {mod}.')
                return
            cps[mod] = cp

        run_btn.config(state='disabled')
        progress['value'] = 0

        def progress_cb(stage, cur, tot):
            pct = cur / tot * 100 if tot > 0 else 0
            root.after(0, lambda: progress.configure(value=pct))
            root.after(0, lambda: status_var.set(f'{stage}... {cur}/{tot}'))

        def task():
            try:
                report = run_pipeline(
                    video_path=video,
                    output_dir=output_var.get(),
                    window_size=window_var.get(),
                    stride=stride_var.get(),
                    threshold=threshold_var.get(),
                    ensemble_weights=ensemble_weights,
                    checkpoint_paths=cps,
                    progress_callback=progress_cb,
                )
                root.after(0, lambda: status_var.set(report['summary']))
                root.after(0, lambda: progress.configure(value=100))
            except Exception as e:
                root.after(0, lambda: status_var.set(f'Error: {e}'))
            finally:
                root.after(0, lambda: run_btn.config(state='normal'))

        threading.Thread(target=task, daemon=True).start()

    run_btn.config(command=on_run)
    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    # Resolve checkpoint paths
    checkpoint_paths = {}
    for mod_name, arg_name in zip(MODALITIES,
            ['weights_joint', 'weights_bone', 'weights_joint_motion', 'weights_bone_motion']):
        path = getattr(args, arg_name)
        if path is None:
            path = auto_detect_checkpoint(mod_name, args.save_dir)
        if path is None or not os.path.isfile(path):
            print(f'Warning: checkpoint not found for {mod_name}. '
                  f'Provide --{arg_name} or place checkpoints in {args.save_dir}/')
            checkpoint_paths[mod_name] = ''
        else:
            checkpoint_paths[mod_name] = path

    ensemble_weights = [float(w) for w in args.ensemble_weights.split(',')]

    if args.cli:
        if not args.video:
            print('Error: --video is required in CLI mode')
            sys.exit(1)
        # Validate all checkpoints exist for CLI mode
        for mod_name in MODALITIES:
            if not checkpoint_paths.get(mod_name) or not os.path.isfile(checkpoint_paths[mod_name]):
                print(f'Error: checkpoint not found for {mod_name}.')
                sys.exit(1)
        run_pipeline(
            video_path=args.video,
            output_dir=args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold,
            ensemble_weights=ensemble_weights,
            checkpoint_paths=checkpoint_paths,
        )
    else:
        run_gui(args, checkpoint_paths, ensemble_weights)
