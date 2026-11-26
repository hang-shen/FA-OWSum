import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_curve,
    roc_auc_score,
)


def _safe_macro(metric, y_true, scores):#逐标签计算ROC-AUC或AP，再求平均。
    values = [
        metric(y_true[:, column], scores[:, column])
        for column in range(y_true.shape[1])
        if np.unique(y_true[:, column]).size >= 2
    ]
    return float(np.mean(values)) if values else np.nan


def score_metrics(y_true, scores):
    """Top1 membership accuracy and macro ranking metrics for every model."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores.argmax(axis=1)
    return {
        "top1": float(y_true[np.arange(len(y_true)), predictions].mean()),
        "roc": _safe_macro(roc_auc_score, y_true, scores),
        "pr": _safe_macro(average_precision_score, y_true, scores),
        "predictions": predictions,
    }


def learn_label_thresholds(y_true, scores):#为每个标签分别选择使F1最大的预测阈值。

    thresholds = np.zeros(y_true.shape[1], dtype=float)
    for column in range(y_true.shape[1]):
        truth = np.asarray(y_true[:, column], dtype=np.int8)
        values = np.asarray(scores[:, column], dtype=float)
        precision, recall, candidates = precision_recall_curve(truth, values)
        if candidates.size == 0:
            thresholds[column] = 0.5
            continue
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(candidates),
            where=(precision[:-1] + recall[:-1]) > 0,
        )
        best = np.flatnonzero(np.isclose(f1, f1.max()))
        thresholds[column] = float(candidates[best[-1]])
    return thresholds


def multilabel_metrics(y_true, scores, thresholds):#根据逐标签阈值生成二值多标签预测，然后计算指标
    predictions = (scores >= np.asarray(thresholds).reshape(1, -1)).astype(np.int8)
    top1 = np.asarray(scores).argmax(axis=1)
    return {
        "subset_accuracy": accuracy_score(y_true, predictions),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, predictions, average="micro", zero_division=0),
        "f1_samples": f1_score(y_true, predictions, average="samples", zero_division=0),
        "jaccard_samples": jaccard_score(
            y_true, predictions, average="samples", zero_division=0
        ),
        "hamming_loss": hamming_loss(y_true, predictions),
        "roc_auc_macro": _safe_macro(roc_auc_score, y_true, scores),
        "average_precision_macro": _safe_macro(
            average_precision_score, y_true, scores
        ),
        "top1_hit": float(y_true[np.arange(len(y_true)), top1].mean()),
        "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
        "empty_prediction_fraction": float((predictions.sum(axis=1) == 0).mean()),
    }


__all__ = ["learn_label_thresholds", "multilabel_metrics", "score_metrics"]
