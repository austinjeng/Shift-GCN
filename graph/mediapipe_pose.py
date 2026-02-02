import sys

sys.path.extend(['../'])
from graph import tools

num_node = 33
self_link = [(i, i) for i in range(num_node)]

# 32 edges forming a spanning tree over 33 MediaPipe Pose landmarks.
# Edges are (child, parent), rooted at NOSE (0).
# Two bridge edges connect the 3 disconnected components of POSE_CONNECTIONS:
#   - NOSE(0) -> LEFT_SHOULDER(11): connects head to body
#   - NOSE(0) -> MOUTH_LEFT(9): connects mouth to head
inward = [
    (1, 0), (2, 1), (3, 2), (7, 3),           # left face
    (4, 0), (5, 4), (6, 5), (8, 6),           # right face
    (9, 0), (10, 9),                            # mouth (bridge: 9->0)
    (11, 0), (12, 11),                          # shoulders (bridge: 11->0)
    (13, 11), (15, 13), (17, 15), (19, 15), (21, 15),  # left arm
    (14, 12), (16, 14), (18, 16), (20, 16), (22, 16),  # right arm
    (23, 11), (24, 12),                         # hips
    (25, 23), (27, 25), (29, 27), (31, 27),    # left leg
    (26, 24), (28, 26), (30, 28), (32, 28),    # right leg
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
