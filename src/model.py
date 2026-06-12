# src/model.py
# ============================================================
# Architecture LightGCN en PyTorch
# Papier : He et al., 2020 — "LightGCN: Simplifying and Powering
# Graph Convolution Network for Recommendation"
# ============================================================

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import EMBEDDING_DIM, NUM_LAYERS, WEIGHT_DECAY, TOP_K


class LightGCN(nn.Module):
    """
    Implémentation complète de LightGCN.

    Principe :
      1. Embeddings initiaux E^(0) pour users et items
      2. Propagation sur K couches : E^(k) = A_hat * E^(k-1)
      3. Agrégation finale : E* = moyenne(E^0 ... E^K)
      4. Score de prédiction : y_ui = <e*_u, e*_i>

    Args:
        n_users       : nombre d'utilisateurs
        n_items       : nombre d'articles
        embedding_dim : dimension des embeddings (défaut : 64)
        n_layers      : nombre de couches K (défaut : 3)
    """

    def __init__(self, n_users, n_items,
                 embedding_dim=EMBEDDING_DIM,
                 n_layers=NUM_LAYERS):
        super(LightGCN, self).__init__()

        self.n_users      = n_users
        self.n_items      = n_items
        self.embedding_dim = embedding_dim
        self.n_layers     = n_layers

        # Embeddings initiaux — SEULS paramètres apprenables
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialisation normale (recommandée par les auteurs)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        n_params = (n_users + n_items) * embedding_dim
        print(f"[LightGCN] {n_users} users | {n_items} items | - model.py:53"
              f"dim={embedding_dim} | K={n_layers} | "
              f"params={n_params:,}")

    # ----------------------------------------------------------
    def forward(self, adj_matrix):
        """
        Propagation avant : propage les embeddings sur le graphe.

        Args:
            adj_matrix : matrice d'adjacence normalisée (torch sparse)

        Returns:
            user_emb : embeddings finaux users (n_users, dim)
            item_emb : embeddings finaux items (n_items, dim)
        """
        # Concaténer users et items : (n_users + n_items, dim)
        E = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        # Stocker toutes les couches pour la moyenne finale
        all_layers = [E]

        for _ in range(self.n_layers):
            # E^(k) = A_hat * E^(k-1)
            E = torch.sparse.mm(adj_matrix, E)
            all_layers.append(E)

        # E* = (1 / K+1) * somme(E^0, ..., E^K)
        E_final = torch.stack(all_layers, dim=1).mean(dim=1)

        # Séparer users et items
        user_emb = E_final[:self.n_users]
        item_emb = E_final[self.n_users:]

        return user_emb, item_emb

    # ----------------------------------------------------------
    def bpr_loss(self, adj_matrix, users, pos_items, neg_items):
        """
        Perte BPR (Bayesian Personalized Ranking).

        L = -sum( log(sigma(y_ui - y_uj)) ) + lambda * ||E^(0)||^2

        Args:
            adj_matrix : matrice d'adjacence
            users      : indices users du batch      (batch,)
            pos_items  : indices items positifs       (batch,)
            neg_items  : indices items négatifs       (batch,)

        Returns:
            loss : scalaire
        """
        user_emb, item_emb = self.forward(adj_matrix)

        u_emb   = user_emb[users]       # (batch, dim)
        pos_emb = item_emb[pos_items]   # (batch, dim)
        neg_emb = item_emb[neg_items]   # (batch, dim)

        # Scores positifs et négatifs (produit scalaire)
        pos_scores = (u_emb * pos_emb).sum(dim=1)   # (batch,)
        neg_scores = (u_emb * neg_emb).sum(dim=1)   # (batch,)

        # Perte BPR
        bpr = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-8
        ).mean()

        # Régularisation L2 sur E^(0) uniquement
        reg = WEIGHT_DECAY * (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / len(users)

        return bpr + reg