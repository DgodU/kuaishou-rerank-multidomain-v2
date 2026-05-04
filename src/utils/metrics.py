from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def compute_gauc(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    user_ids = np.asarray(user_ids)

    total_weight = 0
    weighted_auc = 0.0

    for uid in np.unique(user_ids):
        mask = user_ids == uid
        y_u = y_true[mask]
        p_u = y_pred[mask]
        if y_u.size == 0 or np.unique(y_u).size < 2:
            continue
        auc_u = roc_auc_score(y_u, p_u)
        weight = int(y_u.shape[0])
        weighted_auc += auc_u * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return float(weighted_auc / total_weight)


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred, labels=[0, 1]))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> Dict[str, float]:
    return {
        "auc": compute_auc(y_true, y_pred),
        "gauc": compute_gauc(y_true, y_pred, user_ids),
        "logloss": compute_logloss(y_true, y_pred),
    }
