"""Confidence signals computed from model outputs alone. Nothing here sees gold labels.

`samples` is always an (runs, labels) 0/1 array holding the repeated outputs for one item.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def majority_vote(samples: np.ndarray) -> np.ndarray:
    """Per-label majority decision. A label needs a strict majority of the runs to be predicted."""
    samples = np.asarray(samples)
    return (2 * samples.sum(axis=0) > samples.shape[0]).astype(np.int8)


def label_agreement(samples: np.ndarray) -> np.ndarray:
    """Per label, the share of runs that agree with the majority decision (0.5 to 1)."""
    samples = np.asarray(samples)
    runs = samples.shape[0]
    positive = samples.sum(axis=0)
    return np.maximum(positive, runs - positive) / runs


def mean_pairwise_jaccard(samples: np.ndarray) -> float:
    """Mean Jaccard similarity between the label sets of every pair of runs.

    Two empty sets count as identical (similarity 1). Primary item-level consistency score.
    """
    samples = np.asarray(samples, dtype=np.int64)
    runs = samples.shape[0]
    if runs < 2:
        return float("nan")
    intersection = samples @ samples.T
    sizes = samples.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    similarity = np.where(union == 0, 1.0, intersection / np.maximum(union, 1))
    upper = np.triu_indices(runs, k=1)
    return float(similarity[upper].mean())


def modal_set_frequency(samples: np.ndarray) -> float:
    """Share of runs that produced the most frequent complete label set."""
    samples = np.asarray(samples)
    counts = Counter(map(bytes, np.ascontiguousarray(samples, dtype=np.int8)))
    return max(counts.values()) / samples.shape[0]


def hamming_agreement(samples: np.ndarray) -> float:
    """1 minus the mean pairwise Hamming distance between runs, as a share of the labels."""
    samples = np.asarray(samples, dtype=np.int64)
    runs, n_labels = samples.shape
    if runs < 2:
        return float("nan")
    positive = samples.sum(axis=0)
    disagreeing_pairs = (positive * (runs - positive)).sum()
    pairs = runs * (runs - 1) / 2
    return float(1.0 - disagreeing_pairs / (pairs * n_labels))


def decision_confidence(p1: np.ndarray, decision: np.ndarray) -> np.ndarray:
    """Probability the model gave to each decision, from P(decision = 1) per label."""
    p1 = np.asarray(p1, dtype=float)
    return np.where(np.asarray(decision) == 1, p1, 1.0 - p1)


def joint_log_probability(p1: np.ndarray, decision: np.ndarray) -> float:
    """Sum over labels of the log-probability of the given decisions."""
    return float(np.log(np.clip(decision_confidence(p1, decision), 1e-9, 1.0)).sum())
