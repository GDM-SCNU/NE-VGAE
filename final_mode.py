# coding=utf-8
# Author: Jung
# Time: 2025/8/10 23:40


import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from Jung.utils import normalize_adj
import sys
import networkx as nx
import scipy.sparse as sp
from torch.optim import Adam, SGD
import numpy as np
from EGM.EGM_evaluation import eva
DID = 0
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed_all(826)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    BORED 1.5, 0.5, 1
"""
class TheoryAwareBernoulliDecoder(nn.Module): # 0.7 1.2 1
    def __init__(self, W, init_tau=(1.5, 0.5, 1)):
        """
        d: 潜在维度
        三类顺序: ('pp','nn','pn')
        """
        super().__init__()
        # 可学习参数（softplus 保证 tau>0）
        self.tau_raw  = nn.Parameter(torch.tensor(init_tau), requires_grad=False)   # 3,
        self.W = W
        self.S = self.polarity_spectral()

    def polarity_spectral(self, add_loops=False, loop_alpha=1e-6):
        # 做法2：只有 W，用谱二分得到 {-1,+1}
        W = (self.W + self.W.t()) / 2
        if add_loops:
            W = W + loop_alpha * torch.eye(W.size(0), device=W.device)
        d = torch.clamp(W.sum(1), min=1e-12)
        D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        L = torch.eye(W.size(0), device=W.device) - D_inv_sqrt @ W @ D_inv_sqrt
        evals, evecs = torch.linalg.eigh(L)  # 升序
        f = evecs[:, 1]
        s = torch.where(f >= 0, 1, -1).long()
        return s
    # 0.9
    def forward(self, Z, pos_pairs, neg_pairs, strong_q=0.9, mid_q=(0.4,0.6), eps=1e-8):
        """
        Z: [N,d] 潜在表示（来自 encoder 采样/均值）
        pos_pairs, neg_pairs: [M,2] 边索引（正边/负采样）
        s: [N] 极性 in {+1,-1} (int or long)
        W: [N,N] 情绪相似度（用于 η, g, b）
        返回：logit_pos, logit_neg
        """
        device = Z.device
        W = self.W
        s = self.S
        # 1) 规范性: η_i 与 g_i（基于 W 的强邻）
        Wsym = (W + W.t())/2
        thr_strong = torch.quantile(Wsym[Wsym>0], strong_q) if (Wsym>0).any() else torch.tensor(0., device=device)
        strong_mask = (Wsym >= thr_strong)
        deg = torch.clamp(Wsym.sum(dim=1), min=eps)
        eta = torch.clamp((Wsym * strong_mask).sum(dim=1) / deg, 0., 1.)          # [N]
        # g_i: 只用强邻的加权均值（避免除0）
        Wg = Wsym * strong_mask
        norm = torch.clamp(Wg.sum(dim=1, keepdim=True), min=eps)
        g = (Wg @ Z) / norm                                                       # [N,d]

        # 3) 计算 logit（按对）
        def logits_for(pairs):
            i, j = pairs[:,0], pairs[:,1]
            zi, zj = Z[i], Z[j]
            base = (zi * zj).mean(dim=1)                                           # z_i^T z_j
            # 情绪类别 c_ij
            si, sj = s[i], s[j]
            # 0:pp, 1:nn, 2:pn
            c = torch.where((si==1)&(sj==1), torch.zeros_like(si),
                torch.where((si==-1)&(sj==-1), torch.ones_like(si), torch.full_like(si, 2)))
            tau = self.tau_raw[c]
            # 规范性特征
            ai = eta[i] * (zi * g[i]).sum(dim=1)
            aj = eta[j] * (zj * g[j]).sum(dim=1)
            a = 0.5*(ai + aj)

            return  base /tau  +  0.01 *a

        logit_pos = logits_for(pos_pairs)
        logit_neg = logits_for(neg_pairs)
        return logit_pos, logit_neg



# ===================== Utilities =====================
def to_torch_sparse(mat: sp.csr_matrix, device):
    mat = mat.tocoo()
    indices = torch.tensor(np.vstack([mat.row, mat.col]), dtype=torch.long, device=device)
    values = torch.tensor(mat.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape), device=device).coalesce()

def sample_neg_edges(adj: sp.csr_matrix, num_samples: int, rng: np.random.RandomState):
    N = adj.shape[0]
    neg = []
    tried = set()
    while len(neg) < num_samples:
        i = rng.randint(0, N)
        j = rng.randint(0, N)
        if i == j:
            continue
        if (i, j) in tried:
            continue
        tried.add((i, j))
        if adj[i, j] == 0 and adj[j, i] == 0:
            neg.append((i, j))
    return np.array(neg, dtype=np.int64)


class Encoder_VGNAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_VGNAE, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)


    def forward(self, x):
        """
        大部分数据为0。5 | FRUSTRATED GAMING不需要normalize
        """
        x = F.normalize(x, p=2, dim=1) * 0.5
        x = self.linear(x)


        return x

class Encoder(nn.Module):
    def __init__(self, graph, hid2_dim):
        super().__init__()
        self.label = graph.ndata['label']
        self.clusters = len(torch.unique(self.label))
        self.feat = graph.ndata['feat'].to(torch.float32)
        self.feat_dim = self.feat.shape[1]
        self.hid2_dim = hid2_dim


        self.gcn_mean =  Encoder_VGNAE(self.feat_dim, hid2_dim)
        self.gcn_logstddev =  Encoder_VGNAE(self.feat_dim, hid2_dim)
        self.a_head = nn.Linear(self.feat_dim, 1)   # Gamma shape a>0
        self.b_head = nn.Linear(self.feat_dim, 1)   # Gamma rate  b>0



        self.act = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()         # ensure positivity

        self.adj = graph.adjacency_matrix().to_dense()# 无self-loop
        self.adj = self.adj + torch.eye(graph.num_nodes()) # with self-loop
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)
        self.pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()
        self.adj_1 = torch.from_numpy(normalize_adj(self.adj.numpy()).A)

        weight_mask = self.adj.view(-1) == 1
        self.weight_tensor = torch.ones(weight_mask.size(0))
        self.weight_tensor[weight_mask] = self.pos_weight

    def forward(self):


        self.mean = self.gcn_mean(self.feat)
        self.logstd = self.gcn_logstddev(self.feat)



        a = self.softplus(self.a_head(self.feat))
        b = self.softplus(self.b_head(self.feat))

        return self.mean, self.logstd, 0, a.squeeze(-1), b.squeeze(-1)  # a,b: [N]


class TPriorVGAE(nn.Module):
    def __init__(self, W, graph, hid2_dim, learnable_nu=True, init_nu=5.0):
        super().__init__()
        self.encoder = Encoder(graph, hid2_dim)
        self.decoder = TheoryAwareBernoulliDecoder(W)

        self.learnable_nu = learnable_nu
        if learnable_nu:
            self.nu_raw = nn.Parameter(torch.tensor([math.log(math.exp(init_nu-2)-1)], dtype=torch.float32))
            self.softplus = nn.Softplus()
        else:
            self.register_buffer('nu_const', torch.tensor([init_nu], dtype=torch.float32))



    def nu(self):
        if self.learnable_nu:
            return 2.0 + self.softplus(self.nu_raw)
        else:
            return self.nu_const


    def forward(self, adj, edge_index, neg_ratio=1.0):
        """
        Returns: dict with ELBO terms
        X: [N,F] dense; A_hat: normalized sparse; adj: scipy csr (for edge list)
        """

        mu, logvar, _, a, b = self.encoder()   # a,b: Gamma params for q(tau)
        var = torch.exp(logvar)
        # ----- Prior/Posterior expectations (closed-form) -----
        # Gaussian conditional prior p(z|tau): N(0, I/tau)
        # Gamma prior p(tau): Gamma(nu/2, nu/2)
        nu = self.nu().to(device).squeeze(0)
        prior_shape = nu / 2.0
        prior_rate  = nu / 2.0


        # Expectations
        E_tau     = a / b                     # [N]
        E_log_tau = torch.digamma(a) - torch.log(b)  # [N]
        E_z2 = (mu**2 + var).sum(dim=1)       # [N]

        d = mu.size(1)

        # Eq[log p(z|tau)]
        E_log_p_z_cond_tau = (
            -0.5 * d * math.log(2*math.pi)
            + 0.5 * E_log_tau
            - 0.5 * E_tau * E_z2
        ).sum()



        # # Eq[log p(tau)] with shared nu (can be vectorized if nu_i)
        E_log_p_tau = (
            (prior_shape - 1) * E_log_tau
            - prior_rate * E_tau
            - torch.lgamma(prior_shape) + prior_shape * torch.log(prior_rate)
        ).sum()


        # Eq[log q(z)] (diagonal Gaussian entropy)
        # H[q(z)] = 0.5 * sum_j log(2πe σ^2)
        E_log_q_z = (0.5 * (math.log(2*math.pi*math.e)) * d + 0.5 * logvar.sum(dim=1)).sum()

        # Eq[log q(tau)] (Gamma)
        E_log_q_tau = (
            (a - 1) * E_log_tau - b * E_tau - torch.lgamma(a) + a * torch.log(b)
        ).sum()

        prior_terms = E_log_p_z_cond_tau + E_log_p_tau - E_log_q_z - E_log_q_tau

        # ----- Reconstruction term (edge BCE with logits = z z^T) -----
        # reparameterized sample of z from q(z): one MC sample is enough (low variance here)
        eps = torch.randn_like(mu)
        z = mu + eps * var # torch.sqrt(var + 1e-12)  # [N,d]

        # Build edge list
        adj_coo = sp.coo_matrix(adj.numpy())
        pos_edges = np.vstack([adj_coo.row, adj_coo.col]).T
        pos_edges = pos_edges[pos_edges[:,0] < pos_edges[:,1]]  # undirected unique

        # negative sampling
        num_pos = len(pos_edges)
        num_neg = int(max(1, neg_ratio * num_pos))
        rng = np.random.RandomState(826)
        # rng = np.random.RandomState()
        neg_edges = sample_neg_edges(adj, num_neg, rng)

        pos_logits, neg_logits = self.decoder(z, pos_edges, neg_edges)
        #

        recon_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction='sum')
        recon_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction='sum')
        recon_term = -(recon_pos + recon_neg)   # log p(A|Z)

        # ELBO
        elbo = recon_term  + prior_terms


        out = {
            'elbo': elbo,
            'recon': recon_term.detach(),
            'prior_terms': prior_terms.detach(),
            'mu': mu, 'logvar': logvar, 'a': a, 'b': b, 'z': z,
            'nu': nu.detach()
        }
        return out





def get_feature(feature_path):
    node_emb = dict()
    with open(feature_path, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = embeds[0]
            node_emb[node_id] = embeds[1:]
        reader.close()
    feature = []
    for i in range(len(node_emb)):
        feature.append(node_emb[i])

    return np.array(feature)
def load_data(name):
    r = f"D:\PyCharm_WORK\MyCode\EGM\datasets\ASSISTments17_{name}.pkl"
    with open(r, 'rb') as f:
        data = pkl.load(f)
    graph = dgl.from_scipy(data['adj'])
    graph.ndata['feat'] = torch.from_numpy(get_feature("datasets\\ASSISTments17_"+name+"_feature.emb"))
    graph.ndata['label'] = torch.from_numpy(data['labels'])
    adj = sp.coo_matrix(data['adj'])
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
    return graph, edge_index


def load_mask(name):
    if name == "RES_BORED":
        r = "AveResBored_mask.pkl"
    elif name == "RES_CONCENTRATING":
        r = "AveResEngcon_mask.pkl"
    elif name == "RES_CONFUSED":
        r = "AveResConf_mask.pkl"
    elif name == "RES_FRUSTRATED":
        r = "AveResFrust_mask.pkl"
    elif name == "RES_OFFTASK":
        r = "AveResOfftask_mask.pkl"
    elif name == "RES_GAMING":
        r = "AveResGaming_mask.pkl"
    else:
        raise FileNotFoundError("找不到文件")
    with open(r, 'rb') as f:
        data = pkl.load(f)

    return data['mask']
def main():

    name = "RES_BORED"
    graph, edge_index = load_data(name)
    W  = load_mask(name)
    W = torch.from_numpy(W).to(torch.float32)

    model = TPriorVGAE(W, graph, 64, learnable_nu=True, init_nu=5.0).to(device) # 64



    opt = Adam(model.parameters(), lr=0.01)


    # ----- train -----
    epochs = 200
    max_nmi = 0
    max_acc = 0
    max_ari = 0
    max_f1 = 0


    for ep in range(0, epochs):
        model.train()

        out = model(model.encoder.adj, edge_index, neg_ratio=0.01) # 其他0.01， FRUSTRATED：0.001

        loss = -out['elbo']  # maximize ELBO


        opt.zero_grad()

        loss.backward()

        opt.step()


        with torch.no_grad():
            acc_or, nmi_or, ari_or, f1_or, cluster_id = eva(model.encoder.clusters, model.encoder.label.numpy(), out['z'])
            sys.stdout.write(
                'VGAE >  ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc_or, nmi_or, ari_or, f1_or))
            sys.stdout.flush()

            if nmi_or >= max_nmi:
                max_nmi = nmi_or
                max_acc = acc_or
                max_ari = ari_or
                max_f1 = f1_or
                best_label = cluster_id


    sys.stdout.write(
        'MAX >  ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (max_acc, max_nmi, max_ari, max_f1))
    sys.stdout.flush()

    np.save(f"{name}_best_label", best_label)

if __name__ == "__main__":
    main()
