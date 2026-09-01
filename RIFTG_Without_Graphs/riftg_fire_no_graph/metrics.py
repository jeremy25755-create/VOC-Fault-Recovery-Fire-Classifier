"""Dependency-light metrics for ternary fire classification."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted shapes differ")
    if np.any(actual < 0) or np.any(actual >= n_classes):
        raise ValueError("actual contains an invalid class")
    if np.any(predicted < 0) or np.any(predicted >= n_classes):
        raise ValueError("predicted contains an invalid class")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: tuple[str, ...] = ("Background", "Fire", "Nuisance"),
) -> dict[str, object]:
    matrix = confusion_matrix(actual, predicted, len(class_names))
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = max(float(matrix.sum()), 1.0)
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "accuracy": float(true_positive.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1.0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
