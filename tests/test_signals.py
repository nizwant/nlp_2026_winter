import numpy as np
import pytest

from routing import signals


def test_majority_vote_needs_strict_majority():
    samples = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert signals.majority_vote(samples).tolist() == [1, 1, 0, 0]
    even = np.array([[1, 0], [0, 0]])  # a tie is not a majority
    assert signals.majority_vote(even).tolist() == [0, 0]


def test_label_agreement():
    samples = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    assert signals.label_agreement(samples).tolist() == [0.8, 0.6, 1.0]


def test_jaccard_of_empty_sets_is_one():
    assert signals.mean_pairwise_jaccard(np.zeros((5, 4), dtype=int)) == 1.0


def test_jaccard_known_value():
    # Sets {a}, {a, b}, {}: pairs give 1/2, 0, 0.
    samples = np.array([[1, 0], [1, 1], [0, 0]])
    assert signals.mean_pairwise_jaccard(samples) == pytest.approx(0.5 / 3)


def test_identical_runs_are_fully_consistent():
    samples = np.tile(np.array([1, 0, 1, 0]), (5, 1))
    assert signals.mean_pairwise_jaccard(samples) == 1.0
    assert signals.modal_set_frequency(samples) == 1.0
    assert signals.hamming_agreement(samples) == 1.0


def test_modal_set_frequency():
    samples = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 0]])
    assert signals.modal_set_frequency(samples) == 0.6


def test_hamming_agreement_matches_brute_force():
    rng = np.random.default_rng(1)
    samples = rng.integers(0, 2, size=(5, 7))
    distances = [
        (samples[i] != samples[j]).mean() for i in range(5) for j in range(i + 1, 5)
    ]
    assert signals.hamming_agreement(samples) == pytest.approx(1 - np.mean(distances))


def test_decision_confidence_and_joint_log_probability():
    p1 = np.array([0.9, 0.2, 0.5])
    decision = np.array([1, 0, 1])
    assert signals.decision_confidence(p1, decision).tolist() == pytest.approx([0.9, 0.8, 0.5])
    assert signals.joint_log_probability(p1, decision) == pytest.approx(np.log(0.9 * 0.8 * 0.5))
