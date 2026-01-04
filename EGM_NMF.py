# coding=utf-8
# Author: Jung
# Time: 2025/7/17 23:20
import numpy as np
from sklearn.decomposition import NMF
import pickle as pkl
from EGM.tools import compute_modularity_from_labels, compute_average_intra_density_from_labels
from EGM.EGM_evaluation import end2end_eva
np.random.seed(826)


def load_data(name):
    with open(name+".pkl", 'rb') as f:
        data = pkl.load(f)
    adj = data['adj']
    labels = data['labels']
    return adj, labels
adj, labels = load_data("datasets/ASSISTments17_RES_FRUSTRATED")
model = NMF(n_components=3, init='nndsvda', max_iter=1000, random_state=826)
W = model.fit_transform(adj)
pred = np.argmax(W, axis = 1)

# Q = compute_modularity_from_labels(adj, pred)
# intra_density = compute_average_intra_density_from_labels(adj, pred)
acc, nmi, ari, f1 = end2end_eva(labels, pred)
print(f"acc = {acc}, nmi = {nmi}, ari = {ari}, f1 = {f1}")
# print(f"模块度：{Q}")
# print(f"平均社区内密度：{intra_density}")