# coding=utf-8
# Author: Jung
# Time: 2025/7/21 16:30
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import pickle as pkl
# ------------------- 1. 读取数据 -------------------
df = pd.read_csv("anonymized_full_release_competition_dataset.csv")  # 你自己的路径

col = 'RES_BORED'
col_avg = 'AveResGaming'
# 保留核心字段
df = df[['studentId', 'assistmentId', col, 'correct', col_avg]].dropna()
df['assistmentId'] = df['assistmentId'].astype(str)
df['studentId'] = df['studentId'].astype(str)

# ------------------- 2. 构建辅助映射 -------------------
students = df['studentId'].unique().tolist()

# 每位学生做过哪些题
questions_dict = df.groupby('studentId')['assistmentId'].apply(list).to_dict()

# 每位学生每题的无聊分数
bored_score_dict = {
    sid: dict(zip(d['assistmentId'], d[col]))
    for sid, d in df.groupby('studentId')
}

avgbored_score_dict = {
    sid: d[col_avg].unique()
    for sid, d in df.groupby('studentId')
}

# 每位学生每题是否做对
correctness_dict = {
    sid: dict(zip(d['assistmentId'], d['correct']))
    for sid, d in df.groupby('studentId')
}

# ------------------- 3. 构建图 -------------------
def build_student_graph(students, questions, bored_score, avgbored_score, sim_thres, jaccard_thres=0.4):
    G = nx.Graph()
    mask_G = nx.Graph()
    for stu in students:
        G.add_node(stu)
        mask_G.add_node(stu)

    for i, j in combinations(students, 2):
        mask_G.add_edge(i, j, weight = (avgbored_score[i][0] + avgbored_score[j][0]) /2)

        set_i = set(questions[i])
        set_j = set(questions[j])
        common_q = set_i & set_j
        union_q = set_i | set_j
        if len(common_q) == 0:
            continue
        jaccard = len(common_q) / len(union_q)
        if jaccard >= jaccard_thres: # 主要影响
            bored_i = np.array([bored_score[i][q] for q in common_q])
            bored_j = np.array([bored_score[j][q] for q in common_q])
            emotion_sim = 1 - np.mean(np.abs(bored_i - bored_j))
            if emotion_sim >= sim_thres:
                G.add_edge(i, j, weight=emotion_sim)
    return G, mask_G

sim_thres = np.mean(df[col].tolist())
G, mask_G = build_student_graph(students, questions_dict, bored_score_dict, avgbored_score_dict, sim_thres = 0.9)


# data = {
#     'mask' : nx.to_numpy_array(mask_G, nodelist=students)
# }
# with open(f"{col_avg}_mask"+'.pkl', 'wb') as f:
#     pkl.dump(data, f)

# ------------------- 4. 构建多级标签 -------------------
def build_multilevel_labels(correctness, students):
    labels = {}
    for stu in students:
        records = list(correctness[stu].values())
        if not records:
            labels[stu] = 0
        else:
            acc = np.mean(records)
            if acc < 0.8:
                labels[stu] = 0
            elif acc < 0.9:
                labels[stu] = 1
            else:
                labels[stu] = 2
    return labels

labels_dict = build_multilevel_labels(correctness_dict, students)

# ------------------- 5. 输出邻接矩阵与标签 -------------------
adj = nx.to_numpy_array(G, nodelist=students)
labels = np.array([labels_dict[s] for s in students])

import json
id_map = {i: nid for i, nid in enumerate(students)}
with open("id_map.json", "w") as f:
    json.dump(id_map, f)
# from scipy.sparse import csc_matrix
# adj = csc_matrix(adj)
# data = {
#     'adj' : adj,
#     'labels': labels
# }
# with open(f"cs_BORED"+'.pkl', 'wb') as f:
#     pkl.dump(data, f)
