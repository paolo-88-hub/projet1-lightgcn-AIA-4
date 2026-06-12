# dashboard.py — Projet 1 LightGCN
# ============================================================
# Dashboard temps réel — Recommandation LightGCN
# USAGE : python dashboard.py
# Ouvre  : http://localhost:5000
# ============================================================

from flask import Flask
from flask_socketio import SocketIO, emit
import threading, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config    import (TOP_K, NUM_EPOCHS, EVAL_EVERY, EMBEDDING_DIM,
                            NUM_LAYERS, LEARNING_RATE, WEIGHT_DECAY,
                            BATCH_SIZE, MODELS_DIR, FIGURES_DIR,
                            RESULTS_DIR, OUTPUT_DIR, SEED)
from src.dataset   import MovieLensDataset
from src.graph     import build_adjacency_matrix
from src.model     import LightGCN
from src.evaluator import Evaluator
from src.utils     import create_directories

import torch
import torch.optim as optim
import random

app      = Flask(__name__)
app.config['SECRET_KEY'] = 'lightgcn_enspy_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── État global ──────────────────────────────────────────────
state = {
    'running':  False,
    'done':     False,
    'phase':    'idle',   # idle | dataset | graph | training | evaluation | done
    'pipeline': [],
    'progress': {
        'dataset':    {'pct': 0, 'status': 'waiting', 'detail': ''},
        'graph':      {'pct': 0, 'status': 'waiting', 'detail': ''},
        'training':   {'pct': 0, 'status': 'waiting', 'detail': ''},
        'evaluation': {'pct': 0, 'status': 'waiting', 'detail': ''},
    },
    'training': {
        'epoch': 0, 'total': NUM_EPOCHS,
        'loss': 0.0, 'recall': 0.0, 'ndcg': 0.0,
        'loss_history':   [],
        'recall_history': [],
        'ndcg_history':   [],
        'epoch_history':  [],
        'best_recall': 0.0,
        'best_epoch':  0,
    },
    'stats': {},
    'recommendations': [],
    'conclusion': {},
}

def emit_update():
    socketio.emit('update', state)

def step(msg):
    state['pipeline'].append(msg)
    emit_update()


# ── Fonctions d'entraînement ────────────────────────────────
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_negatives(train_data, n_items, batch_size):
    users, pos_items, neg_items = [], [], []
    all_pairs = [(u, it) for u, items in train_data.items() for it in items]
    sample_size = min(batch_size, len(all_pairs))
    sampled = random.sample(all_pairs, sample_size)
    for user, pos_item in sampled:
        users.append(user)
        pos_items.append(pos_item)
        user_set = set(train_data[user])
        while True:
            neg = random.randint(0, n_items - 1)
            if neg not in user_set:
                neg_items.append(neg)
                break
    return (torch.LongTensor(users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items))


def run_pipeline():
    global state
    state['running'] = True
    state['done']    = False
    set_seed()

    device = torch.device('cpu')

    # ── Phase 1 : Dataset ────────────────────────────────────
    state['phase'] = 'dataset'
    state['progress']['dataset'] = {'pct': 10, 'status': 'running', 'detail': 'Chargement ratings.csv...'}
    step("Chargement du dataset MovieLens...")
    emit_update()

    create_directories()
    dataset = MovieLensDataset()

    state['progress']['dataset'] = {'pct': 60, 'status': 'running', 'detail': 'Filtrage et réindexation...'}
    emit_update()
    time.sleep(0.3)

    n_train = sum(len(v) for v in dataset.train_data.values())
    n_test  = sum(len(v) for v in dataset.test_data.values())

    state['stats'] = {
        'n_users': dataset.n_users,
        'n_items': dataset.n_items,
        'n_train': n_train,
        'n_test':  n_test,
        'embedding_dim': EMBEDDING_DIM,
        'n_layers': NUM_LAYERS,
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'top_k': TOP_K,
        'n_epochs': NUM_EPOCHS,
    }

    state['progress']['dataset'] = {'pct': 100, 'status': 'done',
        'detail': f'{dataset.n_users} users · {dataset.n_items} items · {n_train:,} interactions'}
    step(f"Dataset pret : {dataset.n_users} users, {dataset.n_items} items, {n_train:,} interactions train")
    emit_update()

    # ── Phase 2 : Graphe ─────────────────────────────────────
    state['phase'] = 'graph'
    state['progress']['graph'] = {'pct': 10, 'status': 'running', 'detail': 'Construction matrice R...'}
    step("Construction du graphe biparti utilisateur-article...")
    emit_update()

    adj_matrix = build_adjacency_matrix(
        dataset.train_data, dataset.n_users, dataset.n_items)
    adj_matrix = adj_matrix.to(device)

    state['progress']['graph'] = {'pct': 60, 'status': 'running', 'detail': 'Normalisation D^{-1/2} A D^{-1/2}...'}
    emit_update()
    time.sleep(0.3)

    n_total = dataset.n_users + dataset.n_items
    state['progress']['graph'] = {'pct': 100, 'status': 'done',
        'detail': f'Matrice {n_total}x{n_total} · normalisation symétrique'}
    step(f"Graphe construit : {n_total}x{n_total}, normalisation D^-1/2 A D^-1/2")
    emit_update()

    # ── Phase 3 : Modèle + Entraînement ──────────────────────
    state['phase'] = 'training'
    state['progress']['training'] = {'pct': 0, 'status': 'running', 'detail': 'Initialisation LightGCN...'}
    step(f"LightGCN initialise : dim={EMBEDDING_DIM}, K={NUM_LAYERS}, params={(dataset.n_users+dataset.n_items)*EMBEDDING_DIM:,}")
    emit_update()

    model = LightGCN(n_users=dataset.n_users, n_items=dataset.n_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    evaluator = Evaluator(top_k=TOP_K)

    best_recall    = 0.0
    best_epoch     = 0
    n_interactions = n_train
    n_batches      = max(1, n_interactions // BATCH_SIZE)

    step(f"Demarrage entrainement : {NUM_EPOCHS} epoques, batch={BATCH_SIZE}, lr={LEARNING_RATE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(n_batches):
            users, pos_items, neg_items = sample_negatives(
                dataset.train_data, dataset.n_items, BATCH_SIZE)
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(adj_matrix, users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        pct      = int((epoch / NUM_EPOCHS) * 90) + 5

        state['training']['epoch']        = epoch
        state['training']['loss']         = round(avg_loss, 4)
        state['training']['loss_history'].append(round(avg_loss, 4))
        state['progress']['training']     = {
            'pct': pct, 'status': 'running',
            'detail': f'Epoque {epoch}/{NUM_EPOCHS} · Loss={avg_loss:.4f}'
        }

        # Évaluation périodique
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            recall, ndcg = evaluator.evaluate(
                model, adj_matrix,
                dataset.train_data, dataset.test_data, device)

            state['training']['recall']         = round(recall, 4)
            state['training']['ndcg']           = round(ndcg, 4)
            state['training']['recall_history'].append(round(recall, 4))
            state['training']['ndcg_history'].append(round(ndcg, 4))
            state['training']['epoch_history'].append(epoch)

            if recall > best_recall:
                best_recall = recall
                best_epoch  = epoch
                state['training']['best_recall'] = round(best_recall, 4)
                state['training']['best_epoch']  = best_epoch
                torch.save(model.state_dict(),
                           os.path.join(MODELS_DIR, 'lightgcn_best.pth'))

            step(f"Ep {epoch:3d} | Loss={avg_loss:.4f} | Recall@{TOP_K}={recall:.4f} | NDCG@{TOP_K}={ndcg:.4f}"
                 + (" <-- meilleur" if recall == best_recall else ""))

        if epoch % 5 == 0:
            emit_update()

    state['progress']['training'] = {
        'pct': 100, 'status': 'done',
        'detail': f'100 epoques terminees · Meilleur Recall@{TOP_K}={best_recall:.4f} (ep {best_epoch})'
    }
    emit_update()

    # ── Phase 4 : Évaluation finale ──────────────────────────
    state['phase'] = 'evaluation'
    state['progress']['evaluation'] = {'pct': 20, 'status': 'running', 'detail': 'Chargement meilleur modele...'}
    step("Evaluation finale avec le meilleur modele...")
    emit_update()

    best_path = os.path.join(MODELS_DIR, 'lightgcn_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    state['progress']['evaluation'] = {'pct': 60, 'status': 'running', 'detail': 'Calcul Recall@K et NDCG@K...'}
    emit_update()

    final_recall, final_ndcg = evaluator.evaluate(
        model, adj_matrix,
        dataset.train_data, dataset.test_data, device)

    state['progress']['evaluation'] = {'pct': 80, 'status': 'running', 'detail': 'Generation des recommandations...'}
    emit_update()

    # Recommandations Top-K pour 5 users
    model.eval()
    reco_list = []
    np.random.seed(SEED)
    sample_users = np.random.choice(
        dataset.n_users, min(5, dataset.n_users), replace=False)

    with torch.no_grad():
        user_emb, item_emb = model.forward(adj_matrix)

    for uid in sample_users:
        scores = torch.matmul(user_emb[uid], item_emb.t())
        seen   = dataset.train_data.get(int(uid), [])
        if seen:
            scores[seen] = float('-inf')
        _, top_items = torch.topk(scores, TOP_K)
        top_items    = top_items.cpu().numpy().tolist()
        recs = [{'rank': i+1, 'title': dataset.get_movie_title(it)}
                for i, it in enumerate(top_items)]
        reco_list.append({
            'user_idx': int(uid),
            'n_seen':   len(seen),
            'recs':     recs
        })

    state['recommendations'] = reco_list

    state['progress']['evaluation'] = {
        'pct': 100, 'status': 'done',
        'detail': f'Recall@{TOP_K}={final_recall:.4f} · NDCG@{TOP_K}={final_ndcg:.4f}'
    }
    step(f"Evaluation finale : Recall@{TOP_K}={final_recall:.4f} | NDCG@{TOP_K}={final_ndcg:.4f}")

    # ── Conclusion ───────────────────────────────────────────
    loss_start = state['training']['loss_history'][0] if state['training']['loss_history'] else 0
    loss_end   = state['training']['loss_history'][-1] if state['training']['loss_history'] else 0
    loss_drop  = round((1 - loss_end / max(loss_start, 1e-8)) * 100, 1)

    recall_start = state['training']['recall_history'][0] if state['training']['recall_history'] else 0
    recall_gain  = round((final_recall - recall_start) / max(recall_start, 1e-8) * 100, 1)

    quality = 'excellent' if final_recall >= 0.10 else \
              'tres bon'  if final_recall >= 0.08 else \
              'bon'       if final_recall >= 0.06 else 'satisfaisant'

    state['conclusion'] = {
        'recall':      round(final_recall, 4),
        'ndcg':        round(final_ndcg, 4),
        'best_epoch':  best_epoch,
        'loss_start':  round(loss_start, 4),
        'loss_end':    round(loss_end, 4),
        'loss_drop':   loss_drop,
        'recall_start':round(recall_start, 4),
        'recall_gain': recall_gain,
        'quality':     quality,
        'n_users':     dataset.n_users,
        'n_items':     dataset.n_items,
        'n_train':     n_train,
        'top_k':       TOP_K,
    }

    step("Pipeline LightGCN complet termine avec succes !")
    state['phase']   = 'done'
    state['running'] = False
    state['done']    = True
    emit_update()


# ── Socket events ────────────────────────────────────────────
@socketio.on('start')
def handle_start():
    global state
    if state['running']:
        return
    state = {
        'running': True, 'done': False, 'phase': 'dataset',
        'pipeline': [],
        'progress': {
            'dataset':    {'pct': 0, 'status': 'waiting', 'detail': ''},
            'graph':      {'pct': 0, 'status': 'waiting', 'detail': ''},
            'training':   {'pct': 0, 'status': 'waiting', 'detail': ''},
            'evaluation': {'pct': 0, 'status': 'waiting', 'detail': ''},
        },
        'training': {
            'epoch': 0, 'total': NUM_EPOCHS,
            'loss': 0.0, 'recall': 0.0, 'ndcg': 0.0,
            'loss_history': [], 'recall_history': [],
            'ndcg_history': [], 'epoch_history': [],
            'best_recall': 0.0, 'best_epoch': 0,
        },
        'stats': {}, 'recommendations': [], 'conclusion': {},
    }
    threading.Thread(target=run_pipeline, daemon=True).start()
    emit('update', state)


@socketio.on('get_state')
def handle_get():
    emit('update', state)


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'dashboard.html'),
              encoding='utf-8') as f:
        return f.read()


if __name__ == '__main__':
    print("[Dashboard] Ouvre http://localhost:5000  dashboard_proj1.py:363 - dashboard.py:363")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False,
                 allow_unsafe_werkzeug=True)