import argparse
import os
import pickle
import glob

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

import sys
sys.path.extend(['../'])
from preprocess import pre_normalization

num_joint = 33
max_body_true = 1
max_frame = 300


def extract_landmarks(video_path, max_frame=300):
    """Extract MediaPipe Pose landmarks from a video file.

    Returns:
        np.ndarray of shape (C=3, T, V=33, M=1) or None if no pose detected.
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Warning: cannot open {video_path}')
        return None

    landmarks_seq = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened() and len(landmarks_seq) < max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_world_landmarks:
                frame_joints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
                    dtype=np.float32,
                )  # (33, 3)
            else:
                frame_joints = np.zeros((num_joint, 3), dtype=np.float32)
            landmarks_seq.append(frame_joints)

    cap.release()

    if len(landmarks_seq) == 0:
        return None

    # (T, V, C) -> (C, T, V)
    data = np.stack(landmarks_seq, axis=0)  # (T, 33, 3)
    data = data.transpose(2, 0, 1)  # (3, T, 33)
    # Add person dimension -> (3, T, 33, 1)
    data = data[:, :, :, np.newaxis]
    return data


def gendata(video_dir, out_path, label_map, split_file=None, max_frame=300):
    """Generate skeleton data from video files.

    Args:
        video_dir: Directory containing video files.
        out_path: Output directory for .npy and .pkl files.
        label_map: Dict mapping class name -> integer label.
        split_file: Optional text file listing video basenames for this split
                     (one per line). If None, all videos in video_dir are used.
        max_frame: Maximum number of frames to extract per video.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if split_file and os.path.exists(split_file):
        with open(split_file, 'r') as f:
            video_names = [line.strip() for line in f if line.strip()]
        video_paths = [os.path.join(video_dir, v) for v in video_names]
    else:
        video_paths = sorted(
            glob.glob(os.path.join(video_dir, '*'))
        )

    sample_name = []
    sample_label = []
    sample_data = []

    for vpath in tqdm(video_paths, desc='Extracting poses'):
        if not os.path.isfile(vpath):
            continue
        basename = os.path.basename(vpath)
        # Determine label from parent directory name or filename prefix
        parent_dir = os.path.basename(os.path.dirname(vpath))
        if parent_dir in label_map:
            label = label_map[parent_dir]
        else:
            # Try to infer from filename: expect "classname_*.ext" or "classname/*.ext"
            name_no_ext = os.path.splitext(basename)[0]
            parts = name_no_ext.split('_')
            class_name = parts[0] if parts else name_no_ext
            if class_name in label_map:
                label = label_map[class_name]
            else:
                print(f'Warning: cannot determine label for {vpath}, skipping')
                continue

        data = extract_landmarks(vpath, max_frame=max_frame)
        if data is None:
            continue

        sample_name.append(basename)
        sample_label.append(label)
        sample_data.append(data)

    if len(sample_data) == 0:
        print('No valid samples found.')
        return

    # Pad all samples to max_frame along time axis
    N = len(sample_data)
    fp = np.zeros((N, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    for i, d in enumerate(sample_data):
        T = min(d.shape[1], max_frame)
        fp[i, :, :T, :, :] = d[:, :T, :, :]

    fp = pre_normalization(fp, zaxis=[23, 11], xaxis=[12, 11], center_joint=[23, 24])

    np.save(os.path.join(out_path, 'data_joint.npy'), fp)
    with open(os.path.join(out_path, 'label.pkl'), 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    print(f'Saved {N} samples to {out_path}')
    print(f'  data shape: {fp.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MediaPipe Pose Data Generator for Shift-GCN.')
    parser.add_argument('--video_dir', required=True,
                        help='Directory containing video files (organized by class subdirs or with class prefixes)')
    parser.add_argument('--out_dir', default='../data/mediapipe/',
                        help='Output directory')
    parser.add_argument('--train_split', default=None,
                        help='Text file listing training video filenames')
    parser.add_argument('--val_split', default=None,
                        help='Text file listing validation video filenames')
    parser.add_argument('--label_map', required=True,
                        help='Comma-separated class mappings, e.g. "action:0,no_action:1"')
    parser.add_argument('--max_frame', type=int, default=300,
                        help='Maximum number of frames per video')

    args = parser.parse_args()

    # Parse label map
    label_map = {}
    for pair in args.label_map.split(','):
        k, v = pair.split(':')
        label_map[k.strip()] = int(v.strip())
    print(f'Label map: {label_map}')

    out_path = args.out_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if args.train_split:
        print('Generating training data...')
        gendata(args.video_dir, out_path, label_map,
                split_file=args.train_split, max_frame=args.max_frame)
        # Rename outputs with train_ prefix
        os.rename(os.path.join(out_path, 'data_joint.npy'),
                  os.path.join(out_path, 'train_data_joint.npy'))
        os.rename(os.path.join(out_path, 'label.pkl'),
                  os.path.join(out_path, 'train_label.pkl'))

    if args.val_split:
        print('Generating validation data...')
        gendata(args.video_dir, out_path, label_map,
                split_file=args.val_split, max_frame=args.max_frame)
        os.rename(os.path.join(out_path, 'data_joint.npy'),
                  os.path.join(out_path, 'val_data_joint.npy'))
        os.rename(os.path.join(out_path, 'label.pkl'),
                  os.path.join(out_path, 'val_label.pkl'))

    if not args.train_split and not args.val_split:
        print('Generating data (no split specified)...')
        gendata(args.video_dir, out_path, label_map, max_frame=args.max_frame)
