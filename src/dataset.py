# src/dataset.py
# ============================================================
# Chargement et prétraitement du dataset MovieLens
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

# Ajouter le dossier racine au path pour trouver src/config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (RATINGS_FILE, MOVIES_FILE, MIN_INTERACTIONS,
                         RANDOM_SEED, SEED)


class MovieLensDataset:
    """
    Gère le chargement et le prétraitement du dataset MovieLens.

    Attributes:
        train_data : dict {user_idx: [item_idx, ...]} — interactions train
        test_data  : dict {user_idx: [item_idx]}      — 1 item par user (test)
        n_users    : nombre total d'utilisateurs
        n_items    : nombre total d'articles
    """

    def __init__(self):
        print("= - dataset.py:29" * 55)
        print("CHARGEMENT DU DATASET MOVIELENS - dataset.py:30")
        print("= - dataset.py:31" * 55)

        self.ratings_df = None
        self.movies_df  = None
        self.train_data = {}
        self.test_data  = {}
        self.user2idx   = {}
        self.item2idx   = {}
        self.idx2user   = {}
        self.idx2item   = {}
        self.n_users    = 0
        self.n_items    = 0

        self._load()
        self._filter()
        self._reindex()
        self._split()

        print(f"\n[Dataset] Pret : {self.n_users} utilisateurs, - dataset.py:49"
              f"{self.n_items} articles")

    # ----------------------------------------------------------
    def _load(self):
        """Charge les fichiers CSV depuis data/ml-latest-small/."""
        if not os.path.exists(RATINGS_FILE):
            raise FileNotFoundError(
                f"\n[ERREUR] Fichier introuvable : {RATINGS_FILE}\n"
                "Solution : telecharge MovieLens depuis\n"
                "  https://grouplens.org/datasets/movielens/latest/\n"
                "et extrais ml-latest-small/ dans le dossier data/\n"
            )
        self.ratings_df = pd.read_csv(RATINGS_FILE)
        self.movies_df  = pd.read_csv(MOVIES_FILE)
        print(f"[Dataset] Ratings bruts : {len(self.ratings_df):,} lignes - dataset.py:64")
        print(f"[Dataset] Films         : {len(self.movies_df):,} - dataset.py:65")

    # ----------------------------------------------------------
    def _filter(self):
        """
        Supprime les utilisateurs et items avec moins de
        MIN_INTERACTIONS interactions. Itère jusqu'à stabilisation.
        """
        print(f"[Dataset] Filtrage (min {MIN_INTERACTIONS} interactions)... - dataset.py:73")
        n_before = len(self.ratings_df)

        while True:
            user_counts = self.ratings_df['userId'].value_counts()
            item_counts = self.ratings_df['movieId'].value_counts()

            valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
            valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index

            df_new = self.ratings_df[
                self.ratings_df['userId'].isin(valid_users) &
                self.ratings_df['movieId'].isin(valid_items)
            ]
            if len(df_new) == len(self.ratings_df):
                break
            self.ratings_df = df_new.reset_index(drop=True)

        n_after = len(self.ratings_df)
        print(f"[Dataset] Apres filtrage : {n_after:,} interactions - dataset.py:92"
              f"(supprime {n_before - n_after:,})")

    # ----------------------------------------------------------
    def _reindex(self):
        """
        Réindexe userId et movieId en indices continus 0..N-1.
        Indispensable pour les embeddings PyTorch (nn.Embedding).
        """
        users = sorted(self.ratings_df['userId'].unique())
        items = sorted(self.ratings_df['movieId'].unique())

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        self.ratings_df['user_idx'] = (
            self.ratings_df['userId'].map(self.user2idx))
        self.ratings_df['item_idx'] = (
            self.ratings_df['movieId'].map(self.item2idx))

        self.n_users = len(users)
        self.n_items = len(items)

    # ----------------------------------------------------------
    def _split(self):
        """
        Stratégie leave-one-out :
          - Pour chaque user, la DERNIERE interaction (par timestamp)
            va dans test_data.
          - Tout le reste va dans train_data.
        """
        print("[Dataset] Split train/test (leaveoneout)... - dataset.py:125")

        # Trier par user puis par timestamp
        df = self.ratings_df.sort_values(['user_idx', 'timestamp'])

        for user_idx, group in df.groupby('user_idx'):
            items = group['item_idx'].tolist()
            if len(items) < 2:
                self.train_data[user_idx] = items
                self.test_data[user_idx]  = []
            else:
                self.train_data[user_idx] = items[:-1]   # tous sauf dernier
                self.test_data[user_idx]  = [items[-1]]  # dernier = test

        n_train = sum(len(v) for v in self.train_data.values())
        n_test  = sum(len(v) for v in self.test_data.values())
        print(f"[Dataset] Train : {n_train:,} | Test : {n_test:,} interactions - dataset.py:141")

    # ----------------------------------------------------------
    def get_movie_title(self, item_idx):
        """Retourne le titre d'un film depuis son index interne."""
        original_id = self.idx2item.get(item_idx)
        if original_id is None:
            return "Inconnu"
        row = self.movies_df[self.movies_df['movieId'] == original_id]
        if len(row) == 0:
            return f"Film ID {original_id}"
        return row.iloc[0]['title']