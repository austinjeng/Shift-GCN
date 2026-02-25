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


if __name__ == '__main__':
    args = parse_args()
    if args.cli:
        if not args.video:
            print('Error: --video is required in CLI mode')
            sys.exit(1)
        print(f'Extracting landmarks from: {args.video}')
        result = extract_landmarks(args.video,
                                   progress_callback=lambda cur, tot: print(f'\r  Frame {cur}/{tot}', end=''))
        print(f'\n  World shape: {result["world"].shape}, FPS: {result["fps"]}')

        # Test normalization
        data_batch = result['world'][np.newaxis, ...]  # (1, 3, T, 33, 1)
        normalized = pre_normalize(data_batch)
        print(f'  Normalized shape: {normalized.shape}')

        # Test sliding windows
        windows = create_sliding_windows(normalized[0], args.window_size, args.stride)
        print(f'  Windows: {len(windows)} (window_size={args.window_size}, stride={args.stride})')

        # Test modality derivation
        mods = derive_modalities(windows[0][0])
        for name, arr in mods.items():
            print(f'  {name}: {arr.shape}')
    else:
        print('GUI mode not yet implemented. Use --cli.')
