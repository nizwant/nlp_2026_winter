import numpy as np
import pytest

from routing import cost, selective


def test_risk_coverage_accepts_ties_together():
    correct = np.array([1, 1, 0, 1, 0], dtype=bool)
    score = np.array([0.9, 0.9, 0.9, 0.5, 0.1])
    curve = selective.risk_coverage(correct, score)
    assert curve["threshold"].tolist() == [0.9, 0.5, 0.1]
    assert curve["accepted"].tolist() == [3, 4, 5]
    assert curve["accepted_accuracy"].tolist() == pytest.approx([2 / 3, 3 / 4, 3 / 5])


def test_items_without_a_score_are_never_accepted_but_count_in_coverage():
    correct = np.array([1, 1, 1, 0], dtype=bool)
    score = np.array([0.9, 0.8, np.nan, np.nan])
    curve = selective.risk_coverage(correct, score)
    assert curve["coverage"].max() == 0.5
    outcome = selective.evaluate(correct, score, threshold=0.0)
    assert outcome["accepted"] == 2 and outcome["coverage"] == 0.5


def test_threshold_is_the_highest_coverage_that_meets_the_target():
    # 100 items: the 60 most confident are all right, the next 20 are half right, the rest wrong.
    score = np.linspace(1.0, 0.0, 100)
    correct = np.r_[np.ones(60), np.tile([1, 0], 10), np.zeros(20)].astype(bool)
    threshold = selective.choose_threshold(correct, score, target=0.95, min_accepted=50)
    accepted = selective.accepted_mask(score, threshold)
    assert correct[accepted].mean() >= 0.95
    assert accepted.sum() == 67  # 64 right of 67 is 95.5%; the next item is wrong and gives 64 of 68 = 94.1%
    assert selective.choose_threshold(correct, score, target=0.99, min_accepted=70) is None
    assert selective.choose_threshold(~correct, score, target=0.95, min_accepted=10) is None


def test_final_accuracy_counts_only_accepted_errors():
    correct = np.array([1, 0, 1, 0], dtype=bool)
    score = np.array([0.9, 0.8, 0.2, 0.1])
    outcome = selective.evaluate(correct, score, threshold=0.5)
    assert outcome["accepted_accuracy"] == 0.5
    assert outcome["final_exact_accuracy"] == 0.75
    assert outcome["errors_accepted"] == 1 and outcome["errors_caught"] == 1
    nothing = selective.evaluate(correct, score, threshold=None)
    assert nothing["coverage"] == 0.0 and nothing["final_exact_accuracy"] == 1.0


def test_calibration_error_is_zero_for_a_perfectly_calibrated_score():
    correct = np.r_[np.ones(80), np.zeros(20), np.ones(20), np.zeros(80)]
    probability = np.r_[np.full(100, 0.8), np.full(100, 0.2)]
    assert selective.expected_calibration_error(correct, probability) == pytest.approx(0.0)
    assert selective.expected_calibration_error(correct, 1 - probability) == pytest.approx(0.6)
    calibrator = selective.fit_calibrator(correct, probability * 10)  # any monotone rescaling
    assert calibrator.predict([8.0, 2.0]).tolist() == pytest.approx([0.8, 0.2])


def test_a_good_score_has_a_smaller_area_under_the_risk_coverage_curve():
    rng = np.random.default_rng(0)
    correct = rng.random(500) < 0.6
    informative = correct + rng.normal(0, 0.5, 500)
    assert selective.area_under_risk_coverage(correct, informative) < selective.area_under_risk_coverage(
        correct, rng.random(500)
    )


def test_cost_model_reports_a_loss_when_too_little_is_accepted():
    rates = {"labour_per_hour": 60.0, "gpu": {"power_watts": 200, "electricity_per_kwh": 1.0, "purchase_price": 3000, "lifetime_hours": 10000}}
    assert cost.gpu_rate_per_hour(rates) == pytest.approx(0.2 + 0.3)
    saving = cost.policy_cost(3600, coverage=0.5, human_seconds_per_item=60, gpu_seconds_per_item=1, rates=rates)
    assert saving["human_hours_without_model"] == pytest.approx(60)
    assert saving["human_hours_saved"] == pytest.approx(30)
    assert saving["gpu_hours"] == pytest.approx(1)
    assert saving["net_saving"] == pytest.approx(30 * 60 - 0.5)
    nothing = cost.policy_cost(3600, coverage=0.0, human_seconds_per_item=60, gpu_seconds_per_item=1, rates=rates)
    assert nothing["net_saving"] == pytest.approx(-0.5)
    assert cost.break_even_coverage(60, 1, rates) == pytest.approx(0.5 / 3600 / 1.0)
