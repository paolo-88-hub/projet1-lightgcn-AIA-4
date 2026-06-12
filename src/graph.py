# src/graph.py
# ============================================================
# Construction de la matrice d'adjacence normalisée du graphe
# biparti utilisateur-article pour LightGCN
# ============================================================

import os
import sys
import numpy as np
import torch
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_adjacency_matrix(train_data, n_users, n_items):
    """
    Construit la matrice d'adjacence normalisée A_hat.

    Structure du graphe biparti (n_users + n_items) x (n_users + n_items) :

        A_tilde = [  0    R  ]
                  [ R^T   0  ]

    Normalisation symétrique :
        A_hat = D^{-1/2} * A_tilde * D^{-1/2}

    Args:
        train_data : dict {user_idx: [item_idx, ...]}
        n_users    : nombre d'utilisateurs
        n_items    : nombre d'articles

    Returns:
        adj_tensor : torch.sparse.FloatTensor prêt pour PyTorch
    """
    print("[Graph] Construction de la matrice d'adjacence... - graph.py:36")

    # --- 1. Construire R (matrice d'interaction M x N) ---
    rows, cols = [], []
    for user, items in train_data.items():
        for item in items:
            rows.append(user)
            cols.append(item)

    data = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # --- 2. Construire A_tilde (blocs) ---
    zero_uu = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    zero_ii = sp.csr_matrix((n_items, n_items), dtype=np.float32)

    top    = sp.hstack([zero_uu, R])
    bottom = sp.hstack([R.T, zero_ii])
    A_tilde = sp.vstack([top, bottom]).tocsr()

    # --- 3. Normalisation symétrique D^{-1/2} A D^{-1/2} ---
    rowsum    = np.array(A_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-10, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_hat = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt).tocoo()

    n_total = n_users + n_items
    print(f"[Graph] Matrice A_hat : {n_total}x{n_total}, - graph.py:65"
          f"{A_hat.nnz:,} elements non-nuls")

    # --- 4. Conversion en torch.sparse.FloatTensor ---
    indices = torch.LongTensor(np.vstack([A_hat.row, A_hat.col]))
    values  = torch.FloatTensor(A_hat.data)
    shape   = torch.Size(A_hat.shape)
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)

    return adj_tensor