"""Pure-torch replacement for pytorch3d.ops.knn_points (only what this repo uses)."""
from collections import namedtuple

import torch

_KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, max_elems=2 ** 27):
    """p1: (B,N,3), p2: (B,M,3). Returns squared dists (B,N,K), idx (B,N,K), knn=None.

    Like pytorch3d, dists are squared L2 and sorted ascending. Chunked over N so each distance block has <= max_elems entries (~0.5 GB).
    """
    K = min(K, p2.shape[1])
    chunk = max(1, max_elems // max(1, p2.shape[1]))
    dists, idxs = [], []
    for s in range(0, p1.shape[1], chunk):
        d = torch.cdist(p1[:, s:s + chunk].float(), p2.float()) ** 2
        v, i = d.topk(K, dim=-1, largest=False)
        dists.append(v)
        idxs.append(i)
    return _KNN(torch.cat(dists, 1), torch.cat(idxs, 1), None)
