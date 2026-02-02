import os
import numpy as np
from numpy.lib.format import open_memmap

# 1-indexed bone pairs: (joint, parent). Root joint (NOSE=1) points to itself.
# Matches the spanning tree in graph/mediapipe_pose.py.
paris = {
    'mediapipe': (
        (1, 1),    # NOSE (root, self)
        (2, 1),    # LEFT_EYE_INNER -> NOSE
        (3, 2),    # LEFT_EYE -> LEFT_EYE_INNER
        (4, 3),    # LEFT_EYE_OUTER -> LEFT_EYE
        (5, 1),    # RIGHT_EYE_INNER -> NOSE
        (6, 5),    # RIGHT_EYE -> RIGHT_EYE_INNER
        (7, 6),    # RIGHT_EYE_OUTER -> RIGHT_EYE
        (8, 4),    # LEFT_EAR -> LEFT_EYE_OUTER
        (9, 7),    # RIGHT_EAR -> RIGHT_EYE_OUTER
        (10, 1),   # MOUTH_LEFT -> NOSE
        (11, 10),  # MOUTH_RIGHT -> MOUTH_LEFT
        (12, 1),   # LEFT_SHOULDER -> NOSE
        (13, 12),  # RIGHT_SHOULDER -> LEFT_SHOULDER
        (14, 12),  # LEFT_ELBOW -> LEFT_SHOULDER
        (15, 13),  # RIGHT_ELBOW -> RIGHT_SHOULDER
        (16, 14),  # LEFT_WRIST -> LEFT_ELBOW
        (17, 15),  # RIGHT_WRIST -> RIGHT_ELBOW
        (18, 16),  # LEFT_PINKY -> LEFT_WRIST
        (19, 17),  # RIGHT_PINKY -> RIGHT_WRIST
        (20, 16),  # LEFT_INDEX -> LEFT_WRIST
        (21, 17),  # RIGHT_INDEX -> RIGHT_WRIST
        (22, 16),  # LEFT_THUMB -> LEFT_WRIST
        (23, 17),  # RIGHT_THUMB -> RIGHT_WRIST
        (24, 12),  # LEFT_HIP -> LEFT_SHOULDER
        (25, 13),  # RIGHT_HIP -> RIGHT_SHOULDER
        (26, 24),  # LEFT_KNEE -> LEFT_HIP
        (27, 25),  # RIGHT_KNEE -> RIGHT_HIP
        (28, 26),  # LEFT_ANKLE -> LEFT_KNEE
        (29, 27),  # RIGHT_ANKLE -> RIGHT_KNEE
        (30, 28),  # LEFT_HEEL -> LEFT_ANKLE
        (31, 29),  # RIGHT_HEEL -> RIGHT_ANKLE
        (32, 28),  # LEFT_FOOT_INDEX -> LEFT_ANKLE
        (33, 29),  # RIGHT_FOOT_INDEX -> RIGHT_ANKLE
    ),
}

sets = {
    'train', 'val'
}

dataset = 'mediapipe'

from tqdm import tqdm

for set in sets:
    print(dataset, set)
    data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(
        '../data/{}/{}_data_bone.npy'.format(dataset, set),
        dtype='float32',
        mode='w+',
        shape=(N, 3, T, V, M))

    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(paris[dataset]):
        v1 -= 1
        v2 -= 1
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
