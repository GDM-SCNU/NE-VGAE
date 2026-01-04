# coding=utf-8
# Author: Jung
# Time: 2025/7/17 23:13

import numpy as np
from collections import defaultdict


def compute_modularity_from_labels(adj, labels):
    """
    输入：
        adj: ndarray (N×N) 邻接矩阵
        labels: ndarray (N,) 社区标签
    输出：
        模块度 Q
    """
    m = np.sum(adj) / 2
    degrees = np.sum(adj, axis=1)
    Q = 0.0

    label_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        label_dict[label].append(idx)

    for nodes in label_dict.values():
        for i in nodes:
            for j in nodes:
                Q += adj[i, j] - (degrees[i] * degrees[j]) / (2 * m)

    return Q / (2 * m)


def compute_average_intra_density_from_labels(adj, labels):
    """
    输入：
        adj: ndarray (N×N) 邻接矩阵
        labels: ndarray (N,) 聚类标签
    输出：
        平均社区内部密度
    """
    from collections import defaultdict
    cluster_dict = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_dict[label].append(i)

    densities = []
    for nodes in cluster_dict.values():
        n = len(nodes)
        if n <= 1:
            continue
        sub_adj = adj[np.ix_(nodes, nodes)]
        e = np.sum(sub_adj) / 2
        max_edges = n * (n - 1) / 2
        densities.append(e / max_edges)

    return np.mean(densities) if densities else 0.0



if __name__ == "__main__":
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])

    labels = np.array([0, 0, 1, 1])

    mod = compute_modularity_from_labels(adj, labels)
    density = compute_average_intra_density_from_labels(adj, labels)

    print("模块度:", round(mod, 4))
    print("平均社区内密度:", round(density, 4))
