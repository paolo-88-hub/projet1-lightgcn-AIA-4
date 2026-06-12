# src/utils.py
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FIGURES_DIR, RESULTS_DIR, TOP_K, SEED


def create_directories():
    from src.config import OUTPUT_DIR, MODELS_DIR
    for d in [OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[Utils] Dossiers outputs/ crees. - utils.py:17")


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('LightGCN — Courbes d\'entrainement', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(range(1, len(history['train_loss']) + 1),
            history['train_loss'],
            color='#2563eb', linewidth=2, label='BPR Loss')
    ax.set_xlabel('Epoque')
    ax.set_ylabel('Perte BPR')
    ax.set_title('Convergence de la perte')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['epochs'], history['recall'],
            color='#16a34a', linewidth=2, marker='o',
            label=f'Recall@{TOP_K}')
    ax.plot(history['epochs'], history['ndcg'],
            color='#dc2626', linewidth=2, marker='s',
            label=f'NDCG@{TOP_K}')
    ax.set_xlabel('Epoque')
    ax.set_ylabel('Score')
    ax.set_title('Metriques par epoque')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Utils] Courbes d'entrainement > {path} - utils.py:51")


def visualize_tsne(model, adj_matrix, dataset, device, n_samples=300):
    import torch
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        _, item_emb = model.forward(adj_matrix)
        emb_np = item_emb.cpu().numpy()

    n = min(n_samples, len(emb_np))
    np.random.seed(SEED)
    idx = np.random.choice(len(emb_np), n, replace=False)
    emb_sample = emb_np[idx]

    print(f"[Utils] tSNE sur {n} items... (23 minutes) - utils.py:68")
    tsne = TSNE(n_components=2, perplexity=30,
                random_state=SEED, max_iter=1000, verbose=1)
    emb_2d = tsne.fit_transform(emb_sample)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=range(n), cmap='tab20',
                          alpha=0.6, s=25, edgecolors='none')
    plt.colorbar(scatter, label='Index item')
    plt.title(f't-SNE des embeddings LightGCN ({n} films)', fontsize=13)
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'tsne_embeddings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Utils] tSNE > {path} - utils.py:86")


def show_recommendations(model, adj_matrix, dataset, device,
                          user_indices=None, k=TOP_K):
    import torch

    if user_indices is None:
        np.random.seed(SEED)
        user_indices = np.random.choice(
            dataset.n_users, min(5, dataset.n_users), replace=False)

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward(adj_matrix)

    print("\n - utils.py:102" + "=" * 60)
    print(f"RECOMMANDATIONS TOP{k} - utils.py:103")
    print("= - utils.py:104" * 60)

    for user_idx in user_indices:
        scores = torch.matmul(user_emb[user_idx], item_emb.t())

        seen = dataset.train_data.get(int(user_idx), [])
        if len(seen) > 0:
            scores[seen] = float('-inf')

        _, top_items = torch.topk(scores, k)
        top_items = top_items.cpu().numpy().tolist()

        print(f"\n  User #{user_idx} (a vu {len(seen)} films) : - utils.py:116")
        for rank, item_idx in enumerate(top_items, 1):
            title = dataset.get_movie_title(item_idx)
            print(f"{rank:2d}. {title} - utils.py:119")
        print("  " + "-" * 50)