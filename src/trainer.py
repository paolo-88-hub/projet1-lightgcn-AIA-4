# src/trainer.py
# ============================================================
# Boucle d'entraînement avec échantillonnage négatif BPR
# ============================================================

import os
import sys
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
                         EVAL_EVERY, SEED, TOP_K, MODELS_DIR)


def set_seed(seed=SEED):
    """Fixe toutes les graines pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_negatives(train_data, n_items, batch_size):
    """
    Échantillonnage négatif pour la perte BPR.

    Pour chaque paire (user, pos_item) tirée aléatoirement,
    on sélectionne un neg_item que l'utilisateur n'a PAS vu.

    Returns:
        users, pos_items, neg_items : LongTensor (batch,)
    """
    users, pos_items, neg_items = [], [], []

    # Toutes les paires positives
    all_pairs = [
        (u, item)
        for u, items in train_data.items()
        for item in items
    ]

    # Échantillonner batch_size paires
    sample_size = min(batch_size, len(all_pairs))
    sampled = random.sample(all_pairs, sample_size)

    for user, pos_item in sampled:
        users.append(user)
        pos_items.append(pos_item)

        # Tirer un item négatif (non vu par l'user)
        user_item_set = set(train_data[user])
        while True:
            neg = random.randint(0, n_items - 1)
            if neg not in user_item_set:
                neg_items.append(neg)
                break

    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.LongTensor(neg_items),
    )


def train_one_epoch(model, optimizer, adj_matrix,
                    train_data, n_items, device):
    """
    Une époque d'entraînement complète.

    Returns:
        avg_loss : perte BPR moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0
    n_interactions = sum(len(v) for v in train_data.values())
    n_batches = max(1, n_interactions // BATCH_SIZE)

    for _ in range(n_batches):
        users, pos_items, neg_items = sample_negatives(
            train_data, n_items, BATCH_SIZE)

        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        optimizer.zero_grad()
        loss = model.bpr_loss(adj_matrix, users, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


def train(model, adj_matrix, train_data, test_data,
          n_items, device, evaluator):
    """
    Boucle d'entraînement complète sur NUM_EPOCHS époques.

    Returns:
        history : dict avec 'train_loss', 'recall', 'ndcg', 'epochs'
    """
    set_seed()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [],
        'recall':     [],
        'ndcg':       [],
        'epochs':     [],
    }

    best_recall    = 0.0
    best_model_path = os.path.join(MODELS_DIR, 'lightgcn_best.pth')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'=' * 55} - trainer.py:123")
    print(f"ENTRAINEMENT  {NUM_EPOCHS} epoques - trainer.py:124")
    print(f"{'=' * 55} - trainer.py:125")

    for epoch in range(1, NUM_EPOCHS + 1):

        avg_loss = train_one_epoch(
            model, optimizer, adj_matrix,
            train_data, n_items, device)
        history['train_loss'].append(avg_loss)

        # Évaluation périodique
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            recall, ndcg = evaluator.evaluate(
                model, adj_matrix, train_data, test_data, device)

            history['recall'].append(recall)
            history['ndcg'].append(ndcg)
            history['epochs'].append(epoch)

            marker = ""
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), best_model_path)
                marker = "  <-- meilleur"

            print(f"Ep {epoch:3d}/{NUM_EPOCHS} | - trainer.py:149"
                  f"Loss={avg_loss:.4f} | "
                  f"Recall@{TOP_K}={recall:.4f} | "
                  f"NDCG@{TOP_K}={ndcg:.4f}{marker}")
        else:
            if epoch % 20 == 0:
                print(f"Ep {epoch:3d}/{NUM_EPOCHS} | - trainer.py:155"
                      f"Loss={avg_loss:.4f}")

    print(f"\n  Meilleur Recall@{TOP_K} : {best_recall:.4f} - trainer.py:158")
    print(f"Modele sauvegarde : {best_model_path} - trainer.py:159")
    return history