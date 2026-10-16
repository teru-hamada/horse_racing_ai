# 二値分類のROC AUCを、算出できないケースを考慮して求める。
# 正解ラベルが片方のクラスだけの場合は、指標の計算例外で処理全体を停止させない。
#
import numpy as np
from sklearn.metrics import roc_auc_score


# 正解が一種類しかない場合はAUC未定義のNaNを返し、二種類ある場合だけ指標を計算する。
def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))
