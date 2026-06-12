# src/evaluator.py
# ============================================================
# Métriques d'évaluation : Recall@K et NDCG@K
# ============================================================

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TOP_K


class Evaluator:
    """
    Calcule Recall@K et NDCG@K sur le jeu de test.

    Stratégie :
      - Pour chaque user, on calcule les scores sur tous les items.
      - On masque les items déjà vus en train.
      - On prend le Top-K restant et on mesure la qualité.
    """

    def __init__(self, top_k=TOP_K):
        self.top_k = top_k

    # ----------------------------------------------------------
    def evaluate(self, model, adj_matrix, train_data,
                 test_data, device):
        """
        Évalue le modèle sur l'ensemble de test.

        Returns:
            mean_recall : Recall@K moyen sur tous les users
            mean_ndcg   : NDCG@K moyen sur tous les users
        """
        model.eval()
        recalls, ndcgs = [], []

        with torch.no_grad():
            # Calculer TOUS les embeddings une seule fois (efficace)
            user_emb, item_emb = model.forward(adj_matrix)

            for user_idx, test_items in test_data.items():
                if len(test_items) == 0:
                    continue

                # Scores pour tous les items : (n_items,)
                u_vec  = user_emb[user_idx]           # (dim,)
                scores = torch.matmul(u_vec, item_emb.t())  # (n_items,)

                # Masquer les items déjà vus en entraînement
                train_items = train_data.get(user_idx, [])
                if len(train_items) > 0:
                    scores[train_items] = float('-inf')

                # Top-K prédictions
                _, top_k_idx = torch.topk(scores, self.top_k)
                top_k_list   = top_k_idx.cpu().numpy().tolist()

                # --- Recall@K ---
                n_hits  = len(set(top_k_list) & set(test_items))
                recall  = n_hits / min(len(test_items), self.top_k)
                recalls.append(recall)

                # --- NDCG@K ---
                ndcg = self._ndcg(top_k_list, set(test_items))
                ndcgs.append(ndcg)

        return float(np.mean(recalls)), float(np.mean(ndcgs))

    # ----------------------------------------------------------
    def _ndcg(self, predicted, relevant_set):
        """
        NDCG@K = DCG@K / IDCG@K

        DCG@K = sum_{i=1}^{K} rel_i / log2(i+1)
        IDCG  = DCG dans le cas idéal (pertinents en premier)
        """
        dcg  = 0.0
        idcg = 0.0

        for i, item in enumerate(predicted):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)   # log2(rank + 1), rank base 1

        n_rel = min(len(relevant_set), self.top_k)
        for i in range(n_rel):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0