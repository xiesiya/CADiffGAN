import os
import copy
import numpy as np
import torch
import torch.nn as nn

from data_loader import load_heterograph_data, sample_negatives
from model import Generator, Discriminator
from utils import compute_metrics, find_optimal_threshold


def _load_data(cfg, fold, device):
    return load_heterograph_data(
        cfg.data_dir, device=str(device),
        fold=fold, n_folds=cfg.n_folds, seed=cfg.seed,
        feat_dim=cfg.feat_dim, test_ratio=cfg.test_ratio,
        sim_top_k=cfg.sim_top_k, val_ratio=cfg.val_ratio)


def _build_generator(cfg, data, device):
    in_dims = [f.shape[1] for f in data['node_feats']]
    return Generator(
        in_dims=in_dims, n_hid=cfg.n_hid, n_adjs=data['n_adjs'],
        n_steps=[cfg.n_steps] * cfg.n_cells, cstr=[0],
        n_drug=data['offsets']['n_drug'],
        attn_dim=cfg.attn_dim, edge_emb_dim=cfg.edge_emb_dim,
        dropout=cfg.dropout, timesteps=cfg.timesteps,
        beta_min=cfg.beta_min, beta_max=cfg.beta_max,
        attn_diff_weight=cfg.attn_diff_weight).to(device)


def train_single_fold(cfg, data, device, fold):
    n_mirna = data['offsets']['n_mirna']
    n_drug = data['offsets']['n_drug']
    n_adjs = data['n_adjs']
    in_dims = [f.shape[1] for f in data['node_feats']]

    cstr = [0]
    n_steps = [cfg.n_steps] * cfg.n_cells

    G = Generator(
        in_dims=in_dims, n_hid=cfg.n_hid, n_adjs=n_adjs,
        n_steps=n_steps, cstr=cstr, n_drug=n_drug,
        attn_dim=cfg.attn_dim, edge_emb_dim=cfg.edge_emb_dim,
        dropout=cfg.dropout, timesteps=cfg.timesteps,
        beta_min=cfg.beta_min, beta_max=cfg.beta_max,
        attn_diff_weight=cfg.attn_diff_weight).to(device)

    D = Discriminator(n_drug=n_drug, n_hid=cfg.n_hid,
                      weight_clip=cfg.weight_clip).to(device)

    adjs = data['adjs']
    node_feats = data['node_feats']
    node_types = data['node_types']
    offsets = data['offsets']
    adj_train = data['adj_train'].to(device)
    train_pos = data['train_pos']
    val_pos = data['val_pos']

    g_optimizer = torch.optim.RMSprop(G.parameters(), lr=cfg.g_lr)
    d_optimizer = torch.optim.RMSprop(D.parameters(), lr=cfg.d_lr)
    arch_optimizer = torch.optim.Adam(G.alphas(), lr=cfg.arch_lr)

    if cfg.lr_decay > 0:
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            g_optimizer, gamma=cfg.lr_decay)
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            d_optimizer, gamma=cfg.lr_decay)
    else:
        g_scheduler = d_scheduler = None

    recon_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    best_auc = 0.0
    best_state = None

    for epoch in range(cfg.epochs):
        G.train()
        D.train()

        if epoch < cfg.arch_warmup:
            eps = 0.0
        else:
            eps = cfg.eps_start * (cfg.eps_decay ** (epoch - cfg.arch_warmup))
            eps = max(eps, 0.001)

        idxes_seq, idxes_res = G.sample_paths(eps)

        for _ in range(cfg.g_steps):
            g_optimizer.zero_grad()
            batch_idx = torch.randint(0, n_mirna, (cfg.batch_size,), device=device)
            adj_rows = adj_train[batch_idx]
            k_pm, k_zr = sample_negatives(adj_rows, cfg.pro_pm, cfg.pro_zr, n_drug)
            known_mask = adj_rows + k_pm + k_zr
            target = adj_rows

            fake_emb, pred_rows, t_batch = G(
                node_feats, node_types, adjs, adj_rows,
                batch_idx, n_mirna, idxes_seq, idxes_res)

            pred_matrix = pred_rows * known_mask
            fake_score = D(pred_matrix, fake_emb)

            g_loss = (-torch.mean(fake_score)
                      + cfg.alpha * recon_loss(pred_matrix, target * known_mask)
                      + cfg.bce_weight * bce_loss(pred_rows, adj_rows))
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            g_optimizer.step()

        for _ in range(cfg.d_steps):
            d_optimizer.zero_grad()
            batch_idx = torch.randint(0, n_mirna, (cfg.batch_size,), device=device)
            adj_rows = adj_train[batch_idx]
            k_pm, _ = sample_negatives(adj_rows, cfg.pro_pm, cfg.pro_zr, n_drug)
            known_mask = adj_rows + k_pm

            with torch.no_grad():
                fake_emb, pred_rows, _ = G(
                    node_feats, node_types, adjs, adj_rows,
                    batch_idx, n_mirna, idxes_seq, idxes_res)
            fake_matrix = (pred_rows * known_mask).detach()
            fake_emb = fake_emb.detach()
            fake_score = D(fake_matrix, fake_emb)

            with torch.no_grad():
                real_emb, _, _ = G(
                    node_feats, node_types, adjs, adj_rows,
                    batch_idx, n_mirna, idxes_seq, idxes_res)
            real_matrix = adj_rows
            real_score = D(real_matrix, real_emb.detach())

            d_loss = (-cfg.cost_fn * torch.mean(real_score)
                      + cfg.cost_fp * torch.mean(fake_score))

            if cfg.use_gp:
                gp = D.gradient_penalty(
                    real_matrix, fake_matrix, real_emb.detach(), fake_emb)
                d_loss = d_loss + cfg.gp_lambda * gp
                d_loss.backward()
                d_optimizer.step()
            else:
                d_loss.backward()
                d_optimizer.step()
                D.clip_weights()

        if epoch >= cfg.arch_warmup and len(val_pos) > 0:
            arch_optimizer.zero_grad()
            val_batch_size = min(cfg.batch_size, len(val_pos))
            val_idx = torch.tensor(
                val_pos[:val_batch_size, 0], dtype=torch.long, device=device)
            val_adj_rows = adj_train[val_idx]
            _, val_pred, _ = G(
                node_feats, node_types, adjs, val_adj_rows,
                val_idx, n_mirna, idxes_seq, idxes_res)
            val_loss = bce_loss(val_pred, val_adj_rows)
            val_loss.backward()
            arch_optimizer.step()

        if g_scheduler is not None:
            g_scheduler.step()
            d_scheduler.step()

        if (epoch + 1) % cfg.print_every == 0 or epoch == 0:
            metrics = evaluate(G, data, device, cfg)
            auc = metrics['AUC']
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(G.state_dict())

    G.load_state_dict(best_state)
    best_thr, val_f1 = find_val_optimal_threshold(G, data, device, cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(cfg.save_dir, f'fold{fold}_G_best.pth'))

    return best_state, best_auc, best_thr


@torch.no_grad()
def find_val_optimal_threshold(G, data, device, cfg):
    G.eval()
    n_mirna = data['offsets']['n_mirna']
    n_drug = data['offsets']['n_drug']
    adj_train = data['adj_train'].to(device)
    val_pos = data['val_pos']
    train_pos = data['train_pos']

    idxes_seq, idxes_res = G.sample_paths(0.0)

    all_idx = torch.arange(n_mirna, device=device)
    all_preds = []
    batch_size = cfg.batch_size * 4
    for start in range(0, n_mirna, batch_size):
        end = min(start + batch_size, n_mirna)
        batch_idx = all_idx[start:end]
        adj_rows = adj_train[batch_idx]
        _, pred_rows, _ = G(
            data['node_feats'], data['node_types'], data['adjs'],
            adj_rows, batch_idx, n_mirna, idxes_seq, idxes_res)
        all_preds.append(pred_rows.cpu())
    pred_matrix = torch.cat(all_preds, dim=0).numpy()

    pos_scores = pred_matrix[val_pos[:, 0], val_pos[:, 1]]

    rng = np.random.RandomState(42)
    pos_set = set((int(r), int(c)) for r, c in val_pos)
    for r, c in train_pos:
        pos_set.add((int(r), int(c)))

    neg_scores = []
    while len(neg_scores) < len(val_pos):
        r = rng.randint(0, n_mirna)
        c = rng.randint(0, n_drug)
        if (r, c) not in pos_set:
            neg_scores.append(pred_matrix[r, c])
    neg_scores = np.array(neg_scores)

    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    best_thr, best_f1 = find_optimal_threshold(labels, scores, metric='f1')
    return best_thr, best_f1


@torch.no_grad()
def predict_test_matrix(G, data, device, cfg):
    G.eval()
    n_mirna = data['offsets']['n_mirna']
    n_drug = data['offsets']['n_drug']
    adj_train = data['adj_train'].to(device)
    idxes_seq, idxes_res = G.sample_paths(0.0)

    all_idx = torch.arange(n_mirna, device=device)
    all_preds = []
    batch_size = cfg.batch_size * 4
    for start in range(0, n_mirna, batch_size):
        end = min(start + batch_size, n_mirna)
        batch_idx = all_idx[start:end]
        adj_rows = adj_train[batch_idx]
        _, pred_rows, _ = G(
            data['node_feats'], data['node_types'], data['adjs'],
            adj_rows, batch_idx, n_mirna, idxes_seq, idxes_res)
        all_preds.append(pred_rows.cpu())
    return torch.cat(all_preds, dim=0).numpy()


@torch.no_grad()
def evaluate(G, data, device, cfg):
    G.eval()
    n_mirna = data['offsets']['n_mirna']
    n_drug = data['offsets']['n_drug']
    adj_train = data['adj_train'].to(device)
    test_edges = data['test_edges']
    test_labels = data['test_labels']
    idxes_seq, idxes_res = G.sample_paths(0.0)

    all_idx = torch.arange(n_mirna, device=device)
    all_preds = []
    batch_size = cfg.batch_size * 4
    for start in range(0, n_mirna, batch_size):
        end = min(start + batch_size, n_mirna)
        batch_idx = all_idx[start:end]
        adj_rows = adj_train[batch_idx]
        _, pred_rows, _ = G(
            data['node_feats'], data['node_types'], data['adjs'],
            adj_rows, batch_idx, n_mirna, idxes_seq, idxes_res)
        all_preds.append(pred_rows.cpu())
    pred_matrix = torch.cat(all_preds, dim=0).numpy()

    if test_edges is not None and test_labels is not None:
        labels = test_labels
        preds = pred_matrix[test_edges[:, 0], test_edges[:, 1]]
    else:
        train_pos = data['train_pos']
        labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_pos))])
        neg_r = np.random.randint(0, n_mirna, len(train_pos))
        neg_c = np.random.randint(0, n_drug, len(train_pos))
        preds = np.concatenate([
            pred_matrix[train_pos[:, 0], train_pos[:, 1]],
            pred_matrix[neg_r, neg_c]])
    return compute_metrics(labels, preds)


def train(cfg, data):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    n_folds = cfg.n_folds

    fold_models = []
    fold_aucs = []
    fold_thresholds = []
    fold_metrics_list = []
    eval_G = None

    for fold in range(n_folds):
        fold_data = _load_data(cfg, fold, device)

        best_state, best_auc, best_thr = train_single_fold(
            cfg, fold_data, device, fold)
        fold_models.append(best_state)
        fold_aucs.append(best_auc)
        fold_thresholds.append(best_thr)

        if eval_G is None:
            eval_G = _build_generator(cfg, fold_data, device)
        eval_G.load_state_dict(best_state)
        fold_pred_mat = predict_test_matrix(eval_G, fold_data, device, cfg)

        test_edges = fold_data['test_edges']
        test_labels = fold_data['test_labels']
        if test_edges is not None and test_labels is not None:
            fold_labels = test_labels
            fold_preds = fold_pred_mat[test_edges[:, 0], test_edges[:, 1]]
        else:
            train_pos = fold_data['train_pos']
            fold_labels = np.concatenate(
                [np.ones(len(train_pos)), np.zeros(len(train_pos))])
            neg_r = np.random.randint(
                0, fold_data['offsets']['n_mirna'], len(train_pos))
            neg_c = np.random.randint(
                0, fold_data['offsets']['n_drug'], len(train_pos))
            fold_preds = np.concatenate([
                fold_pred_mat[train_pos[:, 0], train_pos[:, 1]],
                fold_pred_mat[neg_r, neg_c]])
        fold_metrics = compute_metrics(fold_labels, fold_preds,
                                       threshold=best_thr)
        fold_metrics_list.append(fold_metrics)

    base_data = _load_data(cfg, 0, device)

    n_mirna = base_data['offsets']['n_mirna']
    n_drug = base_data['offsets']['n_drug']

    G = _build_generator(cfg, base_data, device)

    ensemble_matrix = np.zeros((n_mirna, n_drug), dtype=np.float32)
    for fold, state in enumerate(fold_models):
        G.load_state_dict(state)
        fold_data = _load_data(cfg, fold, device)
        pred_mat = predict_test_matrix(G, fold_data, device, cfg)
        ensemble_matrix += pred_mat
    ensemble_matrix /= n_folds

    test_edges = base_data['test_edges']
    test_labels = base_data['test_labels']
    if test_edges is not None and test_labels is not None:
        labels = test_labels
        preds = ensemble_matrix[test_edges[:, 0], test_edges[:, 1]]
    else:
        train_pos = base_data['train_pos']
        labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_pos))])
        neg_r = np.random.randint(0, n_mirna, len(train_pos))
        neg_c = np.random.randint(0, n_drug, len(train_pos))
        preds = np.concatenate([
            ensemble_matrix[train_pos[:, 0], train_pos[:, 1]],
            ensemble_matrix[neg_r, neg_c]])

    metrics_05 = compute_metrics(labels, preds, threshold=0.5)
    avg_thr = float(np.mean(fold_thresholds))
    metrics_opt = compute_metrics(labels, preds, threshold=avg_thr)
    test_thr, test_f1 = find_optimal_threshold(labels, preds, metric='f1')
    metrics_test_opt = compute_metrics(labels, preds, threshold=test_thr)

    os.makedirs(cfg.save_dir, exist_ok=True)
    np.save(os.path.join(cfg.save_dir, 'ensemble_predictions.npy'), ensemble_matrix)

    return np.mean(fold_aucs), metrics_opt
