import numpy as np
import networkx as nx
import pandas as pd
from typing import Union

def load_graph(data: Union[np.ndarray, str], directed: bool = None):
    """
    支持输入：
    - 邻接矩阵 (n, n)
    - 边列表 (m, 2) 或 (m, 3)
    - 或传入.npy文件路径
    """
    if isinstance(data, str):
        data = np.load(data)

    data = np.asarray(data)
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        # adjacency matrix
        if directed is None:
            directed = not np.allclose(data, data.T)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] != 0:
                    G.add_edge(i, j, weight=float(data[i, j]))
        return G
    elif data.ndim == 2 and data.shape[1] in (2, 3):
        # edge list
        if directed is None:
            directed = False
        G = nx.DiGraph() if directed else nx.Graph()
        if data.shape[1] == 2:
            for u, v in data:
                G.add_edge(int(u), int(v), weight=1.0)
        else:
            for u, v, w in data:
                G.add_edge(int(u), int(v), weight=float(w))
        return G
    else:
        raise ValueError("Input must be adjacency matrix or edge list.")

def compute_metrics(G: nx.Graph):
    """
    计算并返回六个指标：
    - 密度、平均度、平均最短路径、平均聚类系数、中介中心性（每节点）、模块度
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = np.mean(degs)

    density = nx.density(G)

    # 最短路径计算在最大连通分量上
    if nx.is_directed(G):
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    LCC = G.subgraph(max(components, key=len)).copy()
    try:
        avg_shortest_path = nx.average_shortest_path_length(LCC)
    except:
        avg_shortest_path = float('nan')

    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = float('nan')

    # 中介中心性
    k = None
    if n > 2000:
        k = int(np.sqrt(n))
    bc = nx.betweenness_centrality(G, k=k, normalized=True, weight=None, seed=42)
    bc_df = pd.DataFrame({'node': list(bc.keys()), 'betweenness': list(bc.values())})
    bc_df.sort_values('betweenness', ascending=False, inplace=True)
    bc_df.to_csv("betweenness_centrality.csv", index=False)

    # 模块度（仅适用于无向图）
    if not nx.is_directed(G):
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        from networkx.algorithms.community.quality import modularity
        try:
            Q = modularity(G, comms)
        except:
            Q = float('nan')
    else:
        Q = float('nan')

    summary = {
        "number_of_nodes": n,
        "number_of_edges": m,
        "density": density,
        "average_degree": avg_deg,
        "average_shortest_path": avg_shortest_path,
        "average_clustering": avg_clustering,
        "modularity": Q,
        "n_communities": len(comms) if not nx.is_directed(G) else 'N/A',
    }

    return summary, bc_df

import pickle as pkl

def load_data(name):
    r = f"D:\PyCharm_WORK\MyCode\EGM\datasets\ASSISTments17_{name}.pkl"
    with open(r, 'rb') as f:
        data = pkl.load(f)
    return data['adj'].A
if __name__ == "__main__":
    # 示例：随机生成一个无向稀疏图邻接矩阵

    A = load_data("RES_OFFTASK")
    G = load_graph(A, directed=False)
    metrics, bc_df = compute_metrics(G)

    print("\n=== 网络指标汇总 ===")
    for k, v in metrics.items():
        print(f"{k:<25}: {v}")
    # print("\n介数中心性已保存至：betweenness_centrality.csv")
