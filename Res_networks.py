# coding=utf-8
# Author: Jung
# Time: 2025/7/16 17:00

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pickle as pkl
from scipy.sparse import csc_matrix

df = pd.read_csv("anonymized_full_release_competition_dataset.csv")

col_name = 'AveResGaming'
# 选择用于计算相似度的特征列
feature_cols = [col_name]

# 对每个学生聚合平均（如果存在重复 studentId）
df_student = df.groupby('studentId')[feature_cols].mean().dropna()
df_student_status = df.groupby('studentId')["RES_GAMING"].mean().dropna()
df_student_confidence = df.groupby('studentId')["confidence(GAMING)"].mean().dropna()

prob = df_student_confidence.values * df_student_status.values
s = pd.Series(prob)
labels = pd.qcut(s, q = 2, labels=[0,1])
labels = labels.astype(int).to_numpy()

G = nx.Graph()
n = len(df_student.values)
print(n)
# X = np.array(df_student.values).reshape(-1, 1)
# knn = NearestNeighbors(n_neighbors= 20).fit(X)
# knn_graph = knn.kneighbors_graph(X, mode="connectivity").toarray()
# for i in range(n):
#     for j in range(n):
#         if knn_graph[i, j] and i != j:
#             G.add_edge(i, j)
threshold = 0.99
for i in range(n):
    G.add_edge(i, i)
    for j in range(i+1, n):
        xi, xj = df_student.values[i], df_student.values[j]
        dist = abs(xi - xj)
        sim = 1 / (1+dist)
        if sim > threshold:
            G.add_edge(i, j)
adj = nx.adjacency_matrix(G).A
adj = adj - np.eye(n)
adj = csc_matrix(adj)
data = {
    'name' : f"ASSISTments17-{col_name}",
    'adj' : adj,
    'labels': labels
}
with open(f"_ASSISTments17_{col_name}"+'.pkl', 'wb') as f:
    pkl.dump(data, f)


# threshold = 0.95
# for i in range(n):
#     for j in range(i+1, n):
#         xi, xj = df_student.values[i], df_student.values[j]
#         dist = abs(xi - xj)
#         sim = 1 / (1+dist)
#         if sim > threshold:
#             G.add_edge(i, j)


print("节点数：", G.number_of_nodes())
print("边数：", G.number_of_edges())
print("平均度数：", sum(dict(G.degree()).values()) / G.number_of_nodes())
print("图密度：：", nx.density(G))
#
# import community as community_louvain
# partition = community_louvain.best_partition(G)
# modularity = community_louvain.modularity(partition, G)
# print(modularity)