import math

import numpy as np
import pytest

from routing.metrics import auroc, cluster_bootstrap_ci, error_pr_auc, task_quality


def test_task_quality_known_values():
    gold = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    predicted = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
    quality = task_quality(gold, predicted)
    # true positives 2, false positives 1, false negatives 2
    assert quality["micro_f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 2))
    assert quality["exact_set_accuracy"] == 0.5
    # label 0: F1 1.0; label 1: F1 0.0; label 2 has no gold positives and is left out
    assert quality["macro_f1"] == pytest.approx(0.5)
    assert quality["n_supported_labels"] == 2


def test_auroc_perfect_reversed_and_tied():
    correct = np.array([1, 1, 0, 0])
    assert auroc(correct, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auroc(correct, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert auroc(correct, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # ties count half


def test_auroc_undefined_for_one_class():
    assert math.isnan(auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    assert math.isnan(error_pr_auc(np.array([0, 0]), np.array([0.1, 0.2])))


def test_auroc_ignores_missing_scores():
    correct = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.1, np.nan, np.nan])
    assert auroc(correct, score) == 1.0


def test_error_pr_auc_perfect():
    correct = np.array([1, 1, 0, 0])
    assert error_pr_auc(correct, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_cluster_bootstrap_interval_covers_the_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 5)
    correct = rng.integers(0, 2, size=200)
    score = correct + rng.normal(0, 1.0, size=200)
    estimate = auroc(correct, score)
    low, high = cluster_bootstrap_ci(lambda idx: auroc(correct[idx], score[idx]), groups, n_resamples=300)
    assert low < estimate < high
    assert 0.0 <= low and high <= 1.0


def test_cluster_bootstrap_is_reproducible():
    groups = np.repeat(np.arange(10), 3)
    values = np.arange(30, dtype=float)
    first = cluster_bootstrap_ci(lambda idx: values[idx].mean(), groups, n_resamples=100, seed=3)
    second = cluster_bootstrap_ci(lambda idx: values[idx].mean(), groups, n_resamples=100, seed=3)
    assert first == second
