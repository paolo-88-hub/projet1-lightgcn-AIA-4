# src/config.py
# ============================================================
# CONFIGURATION CENTRALE — tous les hyperparamètres ici
# Modifie ce fichier pour changer les paramètres du modèle
# ============================================================

import os

# --- Chemins ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "ml-latest-small")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR= os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR= os.path.join(OUTPUT_DIR, "results")

RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_FILE  = os.path.join(DATA_DIR, "movies.csv")

# --- Filtrage des données ---
MIN_INTERACTIONS = 5     # Garder users/items avec au moins 5 interactions
RANDOM_SEED      = 42

# --- Modèle LightGCN ---
EMBEDDING_DIM = 64       # Dimension des embeddings (64 ou 128)
NUM_LAYERS    = 3        # Nombre de couches de propagation K

# --- Entraînement ---
LEARNING_RATE = 0.001    # Taux d'apprentissage Adam
WEIGHT_DECAY  = 1e-4     # Régularisation L2 (lambda)
BATCH_SIZE    = 1024     # Taille des mini-batches
NUM_EPOCHS    = 100      # Nombre total d'époques
EVAL_EVERY    = 10       # Évaluer tous les N époques
SEED          = 42

# --- Évaluation ---
TOP_K = 10               # Recall@10 et NDCG@10