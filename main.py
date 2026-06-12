# main.py
# ============================================================
# POINT D'ENTREE PRINCIPAL — Projet 1 LightGCN
# Lance tout le pipeline de A a Z :
#   donnees -> graphe -> modele -> entrainement -> evaluation
#
# USAGE :
#   python main.py
# ============================================================

import os
import sys
import torch

# Importer tous les modules
from src.config import (TOP_K, MODELS_DIR, OUTPUT_DIR,
                         FIGURES_DIR, RESULTS_DIR)
from src.dataset  import MovieLensDataset
from src.graph    import build_adjacency_matrix
from src.model    import LightGCN
from src.trainer  import train
from src.evaluator import Evaluator
from src.utils    import (create_directories, plot_training_curves,
                           visualize_tsne, show_recommendations)


def main():
    print("\n - main.py:28" + "=" * 55)
    print("PROJET 1  LIGHTGCN  RECOMMENDATION DE FILMS - main.py:29")
    print("ENSPY Yaounde  ANIIA 4  2025/2026 - main.py:30")
    print("= - main.py:31" * 55)

    # --------------------------------------------------
    # 0. Créer les dossiers de sortie
    # --------------------------------------------------
    create_directories()

    # --------------------------------------------------
    # 1. Choisir le device (GPU si disponible, sinon CPU)
    # --------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Main] Device : {device} - main.py:42")
    if device.type == 'cuda':
        print(f"[Main] GPU     : {torch.cuda.get_device_name(0)} - main.py:44")

    # --------------------------------------------------
    # 2. Charger les données
    # --------------------------------------------------
    dataset = MovieLensDataset()

    # --------------------------------------------------
    # 3. Construire le graphe biparti
    # --------------------------------------------------
    adj_matrix = build_adjacency_matrix(
        dataset.train_data,
        dataset.n_users,
        dataset.n_items
    ).to(device)

    # --------------------------------------------------
    # 4. Initialiser le modèle LightGCN
    # --------------------------------------------------
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items
    ).to(device)

    # --------------------------------------------------
    # 5. Initialiser l'évaluateur
    # --------------------------------------------------
    evaluator = Evaluator(top_k=TOP_K)

    # --------------------------------------------------
    # 6. Entraîner
    # --------------------------------------------------
    history = train(
        model      = model,
        adj_matrix = adj_matrix,
        train_data = dataset.train_data,
        test_data  = dataset.test_data,
        n_items    = dataset.n_items,
        device     = device,
        evaluator  = evaluator,
    )

    # --------------------------------------------------
    # 7. Évaluation finale avec le meilleur modèle
    # --------------------------------------------------
    print("\n - main.py:89" + "=" * 55)
    print("EVALUATION FINALE - main.py:90")
    print("= - main.py:91" * 55)

    best_path = os.path.join(MODELS_DIR, 'lightgcn_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device,
                       weights_only=True))
        print(f"[Main] Meilleur modele charge : {best_path} - main.py:98")

    recall, ndcg = evaluator.evaluate(
        model, adj_matrix,
        dataset.train_data, dataset.test_data, device)

    print(f"\n  Recall@{TOP_K}  : {recall:.4f} - main.py:104")
    print(f"NDCG@{TOP_K}    : {ndcg:.4f} - main.py:105")

    # --------------------------------------------------
    # 8. Visualisations
    # --------------------------------------------------
    print("\n[Main] Generation des visualisations... - main.py:110")
    plot_training_curves(history)
    visualize_tsne(model, adj_matrix, dataset, device)
    show_recommendations(model, adj_matrix, dataset, device)

    print(f"\n[Main] TERMINE ! Resultats dans : {OUTPUT_DIR}/ - main.py:115")
    print(f"outputs/models/lightgcn_best.pth - main.py:116")
    print(f"outputs/figures/training_curves.png - main.py:117")
    print(f"outputs/figures/tsne_embeddings.png - main.py:118")


if __name__ == "__main__":
    main()