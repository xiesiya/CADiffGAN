import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)


def compute_metrics(labels, preds, threshold=0.5):
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    auc = roc_auc_score(labels, preds)
    aupr = average_precision_score(labels, preds)

    pred_binary = (preds >= threshold).astype(int)
    acc = accuracy_score(labels, pred_binary)
    prec = precision_score(labels, pred_binary, zero_division=0)
    rec = recall_score(labels, pred_binary, zero_division=0)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    mcc = matthews_corrcoef(labels, pred_binary)

    tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUC': auc, 'AUPR': aupr, 'ACC': acc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'MCC': mcc, 'Specificity': specificity,
    }


def find_optimal_threshold(labels, preds, metric='f1', num_thresholds=91):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    best_thr = 0.5
    best_val = -1.0
    for thr in thresholds:
        pred_binary = (preds >= thr).astype(int)
        if metric == 'f1':
            val = f1_score(labels, pred_binary, zero_division=0)
        elif metric == 'youden':
            rec = recall_score(labels, pred_binary, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(
                labels, pred_binary, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            val = rec + spec - 1.0
        elif metric == 'mcc':
            val = matthews_corrcoef(labels, pred_binary)
        else:
            val = f1_score(labels, pred_binary, zero_division=0)
        if val > best_val:
            best_val = val
            best_thr = thr
    return best_thr, best_val


def format_metrics(metrics, prefix=''):
    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    return prefix + ', '.join(parts)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
