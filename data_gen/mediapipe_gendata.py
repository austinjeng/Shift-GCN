import argparse
import os
import pickle
import glob
import random

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

NTU_TRAINING_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}
NTU_TRAINING_CAMERAS = {2, 3}


def parse_ntu_filename(filename):
    """Parse NTU RGB+D filename (SsssCcccPpppRrrrAaaa.ext) to extract metadata.

    Returns:
        dict with keys: setup, camera, subject, replication, action, or None on failure.
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    try:
        action = int(name[name.find('A') + 1:name.find('A') + 4])
        subject = int(name[name.find('P') + 1:name.find('P') + 4])
        camera = int(name[name.find('C') + 1:name.find('C') + 4])
        setup = int(name[name.find('S') + 1:name.find('S') + 4])
        replication = int(name[name.find('R') + 1:name.find('R') + 4])
    except (ValueError, IndexError):
        return None
    return dict(setup=setup, camera=camera, subject=subject,
                replication=replication, action=action)


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


def _subsample_negatives(videos, ratio, seed):
    """Deterministically subsample negative (label=0) videos to balance classes.

    Args:
        videos: list of (video_path, label) tuples.
        ratio: number of negatives per positive (e.g. 1.0 = equal counts).
        seed: random seed for reproducibility.

    Returns:
        Shuffled list of (video_path, label) tuples with balanced classes.
    """
    positives = [v for v in videos if v[1] == 1]
    negatives = [v for v in videos if v[1] == 0]

    target_neg = int(len(positives) * ratio)
    rng = random.Random(seed)
    if target_neg < len(negatives):
        negatives = rng.sample(negatives, target_neg)

    combined = positives + negatives
    rng.shuffle(combined)
    return combined


def _extract_and_save(videos, out_path, part, max_frame=300):
    """Extract MediaPipe landmarks from videos and save as .npy + .pkl.

    Args:
        videos: list of (video_path, label) tuples.
        out_path: output directory.
        part: 'train' or 'val'.
        max_frame: maximum frames per video.
    """
    sample_name = []
    sample_label = []
    sample_data = []

    for vpath, label in tqdm(videos, desc=f'Extracting poses ({part})'):
        data = extract_landmarks(vpath, max_frame=max_frame)
        if data is None:
            continue
        sample_name.append(os.path.basename(vpath))
        sample_label.append(label)
        sample_data.append(data)

    if len(sample_data) == 0:
        print(f'No valid samples found for {part}.')
        return

    N = len(sample_data)
    fp = np.zeros((N, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    for i, d in enumerate(sample_data):
        T = min(d.shape[1], max_frame)
        fp[i, :, :T, :, :] = d[:, :T, :, :]

    fp = pre_normalization(fp, zaxis=[23, 11], xaxis=[12, 11], center_joint=[23, 24])

    np.save(os.path.join(out_path, f'{part}_data_joint.npy'), fp)
    with open(os.path.join(out_path, f'{part}_label.pkl'), 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    n_pos = sum(1 for l in sample_label if l == 1)
    n_neg = sum(1 for l in sample_label if l == 0)
    print(f'Saved {part}: {N} samples (falling={n_pos}, non-falling={n_neg})')
    print(f'  data shape: {fp.shape}')


def gendata_ntu(video_dir, out_path, falling_action=43, benchmark='xsub',
                subsample_ratio=1.0, max_frame=300, seed=42):
    """Generate binary fall-detection data from NTU RGB+D videos via MediaPipe.

    Args:
        video_dir: directory containing NTU .avi/.mp4 video files.
        out_path: output directory for .npy and .pkl files.
        falling_action: NTU action class for falling (default 43 = "falling down").
        benchmark: 'xsub' (cross-subject) or 'xview' (cross-view).
        subsample_ratio: ratio of negatives to positives for class balancing.
        max_frame: maximum frames per video.
        seed: random seed for subsampling reproducibility.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    video_extensions = {'.avi', '.mp4', '.mkv'}
    all_files = sorted(
        f for f in glob.glob(os.path.join(video_dir, '*'))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in video_extensions
    )

    train_videos = []
    val_videos = []

    for fpath in all_files:
        info = parse_ntu_filename(fpath)
        if info is None:
            print(f'Warning: cannot parse NTU filename {os.path.basename(fpath)}, skipping')
            continue

        label = 1 if info['action'] == falling_action else 0

        if benchmark == 'xsub':
            is_training = info['subject'] in NTU_TRAINING_SUBJECTS
        elif benchmark == 'xview':
            is_training = info['camera'] in NTU_TRAINING_CAMERAS
        else:
            raise ValueError(f'Unknown benchmark: {benchmark}')

        if is_training:
            train_videos.append((fpath, label))
        else:
            val_videos.append((fpath, label))

    print(f'Found {len(all_files)} videos, '
          f'train={len(train_videos)}, val={len(val_videos)}')
    print(f'Train falling: {sum(1 for _, l in train_videos if l == 1)}, '
          f'non-falling: {sum(1 for _, l in train_videos if l == 0)}')
    print(f'Val falling: {sum(1 for _, l in val_videos if l == 1)}, '
          f'non-falling: {sum(1 for _, l in val_videos if l == 0)}')

    if subsample_ratio > 0:
        train_videos = _subsample_negatives(train_videos, subsample_ratio, seed)
        val_videos = _subsample_negatives(val_videos, subsample_ratio, seed)
        print(f'After subsampling (ratio={subsample_ratio}):')
        print(f'  train={len(train_videos)}, val={len(val_videos)}')

    _extract_and_save(train_videos, out_path, 'train', max_frame=max_frame)
    _extract_and_save(val_videos, out_path, 'val', max_frame=max_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MediaPipe Pose Data Generator for Shift-GCN.')
    parser.add_argument('--video_dir', required=True,
                        help='Directory containing video files')
    parser.add_argument('--out_dir', default='../data/mediapipe/',
                        help='Output directory')
    parser.add_argument('--max_frame', type=int, default=300,
                        help='Maximum number of frames per video')

    # NTU mode arguments
    parser.add_argument('--ntu_mode', action='store_true',
                        help='Enable NTU RGB+D mode (binary fall detection)')
    parser.add_argument('--falling_action', type=int, default=43,
                        help='NTU action class for falling (default: 43)')
    parser.add_argument('--benchmark', default='xsub', choices=['xsub', 'xview'],
                        help='NTU split benchmark (default: xsub)')
    parser.add_argument('--subsample_ratio', type=float, default=1.0,
                        help='Ratio of negatives to positives for class balancing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling')

    # Generic mode arguments
    parser.add_argument('--train_split', default=None,
                        help='Text file listing training video filenames')
    parser.add_argument('--val_split', default=None,
                        help='Text file listing validation video filenames')
    parser.add_argument('--label_map', default=None,
                        help='Comma-separated class mappings, e.g. "action:0,no_action:1"')

    args = parser.parse_args()

    if args.ntu_mode:
        gendata_ntu(
            video_dir=args.video_dir,
            out_path=args.out_dir,
            falling_action=args.falling_action,
            benchmark=args.benchmark,
            subsample_ratio=args.subsample_ratio,
            max_frame=args.max_frame,
            seed=args.seed,
        )
    else:
        if not args.label_map:
            parser.error('--label_map is required when not using --ntu_mode')

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
