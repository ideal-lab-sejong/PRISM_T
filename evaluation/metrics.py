from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores) * 100)


def aucpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores) * 100)


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    if np.all(np.isfinite(scores)):
        return scores
    finite = scores[np.isfinite(scores)]
    return np.nan_to_num(scores, nan=finite.min() if finite.size else 0.0)
