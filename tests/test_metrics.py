import numpy as np
import pandas as pd

from maritime_calibration.metrics import (
    calibration_metrics,
    evaluate_threshold,
    operating_curve,
    select_threshold_at_fp_budget,
)


def test_perfect_calibration_metrics_are_finite():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = calibration_metrics(labels, probabilities, bins=5)
    assert metrics["brier"] < 0.001
    assert metrics["nll"] < 0.03


def test_operating_curve_counts_tp_fp():
    table = pd.DataFrame({"is_tp": [1, 0, 1], "calibrated": [0.9, 0.8, 0.7]})
    curve = operating_curve(table, "calibrated", ground_truth_count=4, image_count=2)
    assert curve["recall"].iloc[-1] == 0.5
    assert curve["fp_per_image"].iloc[-1] == 0.5


def test_operating_curve_groups_tied_thresholds_and_policy_transfers():
    policy = pd.DataFrame({"is_tp": [1, 0, 0], "prob_model": [0.8, 0.8, 0.5]})
    curve = operating_curve(policy, "prob_model", ground_truth_count=4, image_count=10)
    assert len(curve) == 2
    assert curve["cumulative_fp"].iloc[0] == 1
    selected = select_threshold_at_fp_budget(
        policy, "prob_model", ground_truth_count=4, image_count=10, fp_per_image_budget=0.1
    )
    assert selected["threshold"] == 0.8
    evaluation = pd.DataFrame({"is_tp": [1, 0, 1], "prob_model": [0.9, 0.7, 0.8]})
    result = evaluate_threshold(
        evaluation, "prob_model", selected["threshold"], ground_truth_count=4, image_count=2
    )
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
