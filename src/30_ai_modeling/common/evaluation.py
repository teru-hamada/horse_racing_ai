import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))
