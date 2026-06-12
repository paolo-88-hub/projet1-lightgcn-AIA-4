# Projet 1 — LightGCN : Réseaux de Neurones Convolutifs pour Systèmes de Recommandation
**ANI-IA 4 | BILOA ABADJECK Paolo | ENSPY Yaoundé**

---

## Description

Implémentation complète de **LightGCN** (Light Graph Convolutional Network) pour
la recommandation collaborative sur le dataset **MovieLens ml-latest-small**.

LightGCN modélise les interactions utilisateur-film via un graphe biparti et
propage les embeddings à travers plusieurs couches de convolution graphique
simplifiée (sans transformation ni activation non-linéaire).

---

## Arborescence du projet

```
projet1_lightgcn/
│
├── README.md                  ← Ce fichier
├── requirements.txt           ← Dépendances Python
│
├── data/                      ← Dataset (téléchargé automatiquement)
│   └── ml-latest-small/
│       ├── ratings.csv
│       ├── movies.csv
│       └── ...
│
├── src/                       ← Code source
│   ├── data_loader.py         ← Chargement & prétraitement MovieLens
│   ├── graph_builder.py       ← Construction du graphe biparti A_hat
│   ├── model.py               ← Architecture LightGCN (PyTorch)
│   ├── trainer.py             ← Boucle entraînement BPR Loss
│   ├── evaluator.py           ← Métriques Recall@K, NDCG@K
│   ├── visualizer.py          ← Courbes, t-SNE, graphiques
│   └── train.py               ← Script principal (pipeline complet)
│
├── notebooks/
│   └── LightGCN_Notebook.ipynb  ← Notebook Jupyter interactif
│
└── outputs/                   ← Résultats générés (créé automatiquement)
    ├── lightgcn_best.pt       ← Meilleur modèle sauvegardé
    ├── lightgcn_last.pt       ← Dernier modèle
    ├── results.json           ← Métriques finales
    ├── loss_curve.png         ← Courbe de convergence
    ├── tsne_embeddings.png    ← Visualisation t-SNE
    ├── interaction_dist.png   ← Distribution des interactions
    └── metrics_bar.png        ← Diagramme des métriques
```

---

## Installation

### 1. Créer un environnement virtuel

```bash
# Avec conda (recommandé)
conda create -n lightgcn python=3.10
conda activate lightgcn

# Ou avec venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Installer les dépendances

```bash
cd projet1_lightgcn
pip install -r requirements.txt
```

### 3. (Optionnel) GPU — installer PyTorch CUDA

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Utilisation

### Lancer le pipeline complet

```bash
cd projet1_lightgcn
python src/train.py
```

Ce script effectue automatiquement :
1. Téléchargement du dataset MovieLens
2. Prétraitement et construction du graphe biparti
3. Entraînement LightGCN (100 époques, BPR Loss)
4. Évaluation Recall@10 et NDCG@10
5. Génération des visualisations (t-SNE, courbes)
6. Sauvegarde du modèle et des résultats

### Lancer le notebook

```bash
jupyter notebook notebooks/LightGCN_Notebook.ipynb
```

---

## Concepts clés

| Concept | Description |
|---------|-------------|
| **Graphe biparti** | Représentation utilisateurs ↔ films |
| **A_hat** | Matrice d'adjacence normalisée D^{-1/2} A D^{-1/2} |
| **Propagation** | E^(k) = A_hat · E^(k-1) |
| **Agrégation** | E* = moyenne de E^(0) à E^(K) |
| **BPR Loss** | Bayesian Personalized Ranking |
| **Recall@K** | Items pertinents retrouvés dans le Top-K |
| **NDCG@K** | Qualité de l'ordre des recommandations |

---

## Configuration

Dans `src/train.py`, modifiez le dictionnaire `CONFIG` :

```python
CONFIG = {
    "embed_dim" : 64,    # Dimension des embeddings (64 ou 128)
    "n_layers"  : 3,     # Couches de propagation
    "n_epochs"  : 100,   # Nombre d'époques
    "lr"        : 1e-3,  # Taux d'apprentissage
    "top_k"     : 10,    # Top-K pour l'évaluation
}
```

---

## Résultats attendus

| Métrique | Valeur typique |
|----------|----------------|
| Recall@10 | 0.15 — 0.20 |
| NDCG@10   | 0.10 — 0.15 |

---

## Auteur

**BILOA ABADJECK Paolo** — ENSPY, Département Génie Informatique  
Encadrant : M. BITHA Junior
