import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable

spine_graph = {0: [0],
               1: [1, 11],
               2: [2, 11, 12],
               3: [3, 12, 13],
               4: [4, 13, 14],
               5: [5, 14, 15],
               6: [6, 15, 16],
               7: [7, 16, 17],
               8: [8, 17, 18],
               9: [9, 18, 19],
               10: [10, 19],
               11: [11, 1, 2],
               12: [12, 2, 3],
               13: [13, 3, 4],
               14: [14, 4, 5],
               15: [15, 5, 6],
               16: [16, 6, 7],
               17: [17, 7, 8],
               18: [18, 8, 9],
               19: [19, 9, 10]}

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj_dict, num_classes=17):
    adj = np.zeros((max(adj_dict.keys()) + 1, max(adj_dict.keys()) + 1))
    for node, neighbors in adj_dict.items():
        for neighbor in neighbors:
            adj[node, neighbor] = 1
            adj[neighbor, node] = 1
    adj = adj[:num_classes, :num_classes]
    adj_normalized = normalize_adj(sp.coo_matrix(adj))
    adj_dense = adj_normalized.todense()
    adj_dense = np.asarray(adj_dense)
    if num_classes == 2:
        adj_dense = np.stack([adj_dense, 1 - adj_dense], axis=-1)
    return adj_dense

def normalize_adj_torch(adj, num_classes=17):
    if len(adj.size()) == 4:
        batch_size, num_nodes, _, _ = adj.size()
        new_r = torch.zeros(adj.size()).type_as(adj)
        for i in range(num_nodes):
            adj_item = adj[0, i]
            rowsum = adj_item.sum(1)
            d_inv_sqrt = rowsum.pow(-0.5)
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
            new_r[0, i, ...] = r
        return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    if num_classes == 2:
        r = torch.stack([r, 1 - r], dim=-1)
    return r

if __name__ == '__main__':
    cihp_adj = preprocess_adj(spine_graph, num_classes=17)
    print(f"Preprocessed adj shape: {cihp_adj.shape}")
    adj_tensor = torch.from_numpy(cihp_adj).float()
    print(f"Adj tensor shape: {adj_tensor.shape}")