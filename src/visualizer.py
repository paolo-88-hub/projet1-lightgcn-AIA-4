"""
=============================================================
Projet 1 — LightGCN  |  ANI-IA 4  |  BILOA ABADJECK Paolo
=============================================================
Module : visualizer.py
Description : Visualisations — courbe de perte, t-SNE des
              embeddings, analyse des recommandations.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── 1. Courbe de perte ───────────────────────────────────

def plot_loss_curve(history: dict,
                   save: bool = True,
                   filename: str = "loss_curve.png") -> None:
    """
    Trace la courbe de perte BPR par époque.

    Paramètres
    ----------
    history  : dict {'epoch': [...], 'loss': [...]}
    save     : sauvegarder la figure
    filename : nom du fichier de sortie
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["epoch"], history["loss"],
            color="#2196F3", linewidth=2, label="Perte BPR")
    ax.fill_between(history["epoch"], history["loss"],
                    alpha=0.1, color="#2196F3")

    ax.set_xlabel("Époque", fontsize=13)
    ax.set_ylabel("Perte BPR", fontsize=13)
    ax.set_title("Courbe de convergence — LightGCN (BPR Loss)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(path, dpi=150)
        print(f"[INFO] Courbe de perte sauvegardée : {path}")

    plt.close(fig)


# ─── 2. t-SNE des embeddings ──────────────────────────────

def plot_tsne_embeddings(model,
                         A_hat,
                         n_users: int,
                         n_items: int,
                         n_sample_users: int = 200,
                         n_sample_items: int = 300,
                         save: bool = True,
                         filename: str = "tsne_embeddings.png") -> None:
    """
    Visualise les embeddings utilisateurs et items en 2D via t-SNE.

    Paramètres
    ----------
    model           : modèle LightGCN entraîné
    A_hat           : graphe normalisé
    n_users         : nombre d'utilisateurs
    n_items         : nombre d'items
    n_sample_users  : nb d'utilisateurs à visualiser (sous-échantillon)
    n_sample_items  : nb d'items à visualiser
    save            : sauvegarder la figure
    filename        : nom du fichier
    """
    model.eval()
    with torch.no_grad():
        e_users, e_items = model.get_all_embeddings(A_hat)
        e_users = e_users.cpu().numpy()
        e_items = e_items.cpu().numpy()

    # Sous-échantillonnage
    u_idx = np.random.choice(n_users, min(n_sample_users, n_users), replace=False)
    i_idx = np.random.choice(n_items, min(n_sample_items, n_items), replace=False)

    emb_u = e_users[u_idx]
    emb_i = e_items[i_idx]

    # Concaténation pour t-SNE
    all_emb = np.vstack([emb_u, emb_i])
    labels  = ["Utilisateur"] * len(emb_u) + ["Item"] * len(emb_i)

    print("[INFO] Calcul du t-SNE (peut prendre 1-2 min)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(all_emb)

    # Séparation
    emb_u_2d = emb_2d[:len(emb_u)]
    emb_i_2d = emb_2d[len(emb_u):]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(emb_u_2d[:, 0], emb_u_2d[:, 1],
               c="#E53935", s=40, alpha=0.6, label="Utilisateurs", marker="o")
    ax.scatter(emb_i_2d[:, 0], emb_i_2d[:, 1],
               c="#1E88E5", s=20, alpha=0.4, label="Items (Films)", marker="^")

    ax.set_title("Visualisation t-SNE — Embeddings LightGCN", fontsize=16)
    ax.set_xlabel("Composante 1", fontsize=12)
    ax.set_ylabel("Composante 2", fontsize=12)
    ax.legend(fontsize=12, markerscale=1.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(path, dpi=150)
        print(f"[INFO] t-SNE sauvegardé : {path}")

    plt.close(fig)


# ─── 3. Distribution des interactions ─────────────────────

def plot_interaction_distribution(df,
                                  save: bool = True,
                                  filename: str = "interaction_dist.png") -> None:
    """
    Histogramme de la distribution des interactions par utilisateur et item.
    """
    user_counts = df.groupby("user_idx")["item_idx"].count()
    item_counts = df.groupby("item_idx")["user_idx"].count()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(user_counts, bins=50, color="#4CAF50", edgecolor="white")
    axes[0].set_title("Interactions par Utilisateur", fontsize=14)
    axes[0].set_xlabel("Nombre d'interactions", fontsize=12)
    axes[0].set_ylabel("Fréquence", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(item_counts, bins=50, color="#FF9800", edgecolor="white")
    axes[1].set_title("Interactions par Item (Film)", fontsize=14)
    axes[1].set_xlabel("Nombre d'interactions", fontsize=12)
    axes[1].set_ylabel("Fréquence", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Distribution des Interactions — MovieLens", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Distribution sauvegardée : {path}")

    plt.close(fig)


# ─── 4. Tableau de résultats ──────────────────────────────

def plot_metrics_bar(results_dict: dict,
                     save: bool = True,
                     filename: str = "metrics_bar.png") -> None:
    """
    Diagramme en barres des métriques Recall@K et NDCG@K.

    Paramètres
    ----------
    results_dict : {'Recall@10': 0.xx, 'NDCG@10': 0.xx, ...}
    """
    keys   = list(results_dict.keys())
    values = list(results_dict.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E53935"][:len(keys)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Résultats d'Évaluation — LightGCN", fontsize=15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(path, dpi=150)
        print(f"[INFO] Métriques sauvegardées : {path}")

    plt.close(fig)


# ─── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    # Test courbe de perte
    history = {
        "epoch": list(range(1, 11)),
        "loss":  [0.9, 0.7, 0.6, 0.55, 0.5, 0.47, 0.44, 0.42, 0.41, 0.4]
    }
    plot_loss_curve(history, save=True)
    print("[INFO] Visualiser.py : test OK")
