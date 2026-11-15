"""Task quality and how well a confidence score separates correct from wrong predictions."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def task_quality(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Micro-F1, macro-F1 and exact-set accuracy for (items, labels) 0/1 arrays.

    Macro-F1 averages over the labels that have at least one gold positive in `gold`; a label
    with no positives has no defined recall and would only add zeros.
    """
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    supported = gold.sum(axis=0) > 0
    per_label = f1_score(gold, predicted, average=None, zero_division=0)
    return {
        "micro_f1": float(f1_score(gold, predicted, average="micro", zero_division=0)),
        "macro_f1": float(per_label[supported].mean()) if supported.any() else float("nan"),
        "exact_set_accuracy": float((gold == predicted).all(axis=1).mean()),
        "n_items": int(gold.shape[0]),
        "n_supported_labels": int(supported.sum()),
    }


def auroc(correct: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve for predicting correctness from a score (higher = more confident).

    Tied scores are handled by mid-ranks. Returns NaN when all predictions are correct or all
    are wrong, because the quantity is undefined then.
    """
    correct = np.asarray(correct).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    keep = ~np.isnan(score)
    correct, score = correct[keep], score[keep]
    if correct.size == 0 or correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, score))


def error_pr_auc(correct: np.ndarray, score: np.ndarray) -> float:
    """Average precision for detecting errors: errors are the positive class, ranked by low score."""
    correct = np.asarray(correct).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    keep = ~np.isnan(score)
    correct, score = correct[keep], score[keep]
    if correct.size == 0 or correct.min() == correct.max():
        return float("nan")
    return float(average_precision_score(1 - correct, -score))


def cluster_bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for `statistic(row_indices)`, resampling whole groups (articles).

    Resamples in which the statistic is undefined (NaN) are left out of the percentiles.
    """
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        drawn = rng.integers(0, len(unique), size=len(unique))
        values[b] = statistic(np.concatenate([rows_of[g] for g in drawn]))
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))
