"""Routing: accept the model's label set when a confidence score is high enough, else ask a human.

The rule is fixed before any test result exists (PLAN.md, "Routing and cost"):

- On calibration data, take the threshold with the highest coverage whose accepted items reach
  the target exact-set accuracy, with at least `min_accepted` accepted items. If no threshold
  qualifies, the target is infeasible for that signal and is reported as such.
- Items tied at the threshold are accepted together. An item without a score (an invalid model
  output) is never accepted and stays in the coverage denominator.
- On evaluation data the threshold is applied as it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

TARGETS = (0.95, 0.90, 0.80)
MIN_ACCEPTED = 50


def risk_coverage(correct: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Coverage and accepted-item accuracy for every distinct threshold, most confident first.

    Row k accepts every item whose score is at least the k-th highest distinct score.
    """
    correct = np.asarray(correct, dtype=bool)
    score = np.asarray(score, dtype=float)
    n = len(score)
    have = ~np.isnan(score)
    order = np.argsort(-score[have], kind="stable")
    sorted_score, sorted_correct = score[have][order], correct[have][order]
    if sorted_score.size == 0:
        return pd.DataFrame(columns=["threshold", "accepted", "coverage", "accepted_accuracy"])
    last_of_tie = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    accepted = np.arange(1, sorted_score.size + 1)[last_of_tie]
    right = np.cumsum(sorted_correct)[last_of_tie]
    return pd.DataFrame(
        {
            "threshold": sorted_score[last_of_tie],
            "accepted": accepted,
            "coverage": accepted / n,
            "accepted_accuracy": right / accepted,
        }
    )


def choose_threshold(
    correct: np.ndarray, score: np.ndarray, target: float, min_accepted: int = MIN_ACCEPTED
) -> float | None:
    """Threshold with the highest coverage that meets the target on these (calibration) items."""
    curve = risk_coverage(correct, score)
    feasible = curve[(curve["accepted_accuracy"] >= target) & (curve["accepted"] >= min_accepted)]
    if feasible.empty:
        return None
    return float(feasible.loc[feasible["coverage"].idxmax(), "threshold"])


def accepted_mask(score: np.ndarray, threshold: float | None) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if threshold is None:
        return np.zeros(len(score), dtype=bool)
    with np.errstate(invalid="ignore"):
        return ~np.isnan(score) & (score >= threshold)


def evaluate(correct: np.ndarray, score: np.ndarray, threshold: float | None) -> dict:
    """Outcome of applying a fixed threshold: who is accepted and how good the result is.

    `final_exact_accuracy` assumes that a human corrects every routed item perfectly (the gold
    label stands in for the human), so only accepted errors remain.
    """
    correct = np.asarray(correct, dtype=bool)
    accepted = accepted_mask(score, threshold)
    n, k = len(correct), int(accepted.sum())
    return {
        "accepted": k,
        "coverage": k / n if n else float("nan"),
        "accepted_accuracy": float(correct[accepted].mean()) if k else float("nan"),
        "final_exact_accuracy": float(1.0 - (~correct[accepted]).sum() / n) if n else float("nan"),
        "errors_accepted": int((~correct[accepted]).sum()),
        "errors_caught": int((~correct[~accepted]).sum()),
    }


def area_under_risk_coverage(correct: np.ndarray, score: np.ndarray) -> float:
    """Mean error rate of the accepted items over all coverage levels (lower is better)."""
    curve = risk_coverage(correct, score)
    if curve.empty:
        return float("nan")
    coverage = np.r_[0.0, curve["coverage"].to_numpy()]
    risk = 1.0 - curve["accepted_accuracy"].to_numpy()
    return float(np.sum(np.diff(coverage) * risk) / coverage[-1])


def fit_calibrator(correct: np.ndarray, score: np.ndarray) -> IsotonicRegression:
    """Monotone map from a score to the probability that the prediction is right."""
    correct, score = np.asarray(correct, dtype=float), np.asarray(score, dtype=float)
    have = ~np.isnan(score)
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score[have], correct[have])


def expected_calibration_error(correct: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    """Mean gap between stated probability and observed accuracy over equal-width bins."""
    correct, probability = np.asarray(correct, dtype=float), np.asarray(probability, dtype=float)
    have = ~np.isnan(probability)
    correct, probability = correct[have], probability[have]
    if correct.size == 0:
        return float("nan")
    bins = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        inside = bins == b
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - probability[inside].mean())
    return float(error)
