from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.audit_rare_class_failure import matched_target_annotation_ids
from scripts.audit_rare_class_visual_domain import crop_features
from scripts.prepare_rare_positive_sampling import repeated_training_lines
from scripts.inspect_resumable_checkpoint import tensor_finiteness
from scripts.analyze_rare_class_factorial import metric_comparison_rows
from scripts.analyze_rare_class_factorial_complete import three_cell_metric_rows
from scripts.analyze_rare_class_factorial_2x2 import factorial_effects, four_cell_metric_rows
from scripts.analyze_target_error_taxonomy_2x2 import localized_counts


def test_target_matching_is_same_class_and_one_to_one() -> None:
    annotations = [
        {"id": 1, "image_id": 10, "category_id": 4, "bbox": [0, 0, 10, 10]},
        {"id": 2, "image_id": 10, "category_id": 4, "bbox": [20, 0, 10, 10]},
    ]
    predictions = [
        {"image_id": 10, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 10, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 10, "category_id": 1, "bbox": [20, 0, 10, 10], "score": 0.99},
    ]

    matched = matched_target_annotation_ids(annotations, predictions, 4, 0.5)

    assert matched == {1}


def test_crop_features_detect_local_luminance_contrast() -> None:
    pixels = np.full((20, 20, 3), 40, dtype=np.uint8)
    pixels[8:12, 8:12] = 220
    image = Image.fromarray(pixels, mode="RGB")

    features = crop_features(image, [8, 8, 4, 4], context_factor=4.0)

    assert features["target_luminance_mean"] > features["context_luminance_mean"]
    assert features["absolute_luminance_contrast"] > 0.5
    assert features["standardized_luminance_contrast"] > 1.0


def test_rare_positive_repeat_preserves_unique_images() -> None:
    source = ["C:/images/1.jpg", "C:/images/2.jpg", "C:/images/3.jpg"]

    output, rare = repeated_training_lines(source, {"2.jpg"}, repeat_total=4)

    assert rare == ["C:/images/2.jpg"]
    assert output.count("C:/images/2.jpg") == 4
    assert len(output) == 6
    assert set(output) == set(source)


def test_nested_checkpoint_finiteness_counts_only_float_tensors() -> None:
    floating, nonfinite = tensor_finiteness(
        {
            "finite": torch.tensor([1.0, 2.0]),
            "nested": [
                torch.tensor([float("inf"), float("nan")]),
                torch.tensor([1, 2], dtype=torch.int64),
            ],
        }
    )

    assert floating == 4
    assert nonfinite == 2


def test_factorial_metric_rows_compute_repeat_minus_natural() -> None:
    natural = {
        "aggregate": {"precision": 0.5, "recall": 0.4, "map50": 0.3, "map50_95": 0.2},
        "per_class": {
            "target": {"precision": 0.1, "recall": 0.2, "map50": 0.3, "map50_95": 0.4}
        },
    }
    repeat = {
        "aggregate": {"precision": 0.6, "recall": 0.4, "map50": 0.2, "map50_95": 0.25},
        "per_class": {
            "target": {"precision": 0.2, "recall": 0.3, "map50": 0.4, "map50_95": 0.5}
        },
    }

    aggregate, classes = metric_comparison_rows(natural, repeat)

    assert aggregate.set_index("metric").loc["map50_95", "delta_repeat4_minus_natural"] == pytest.approx(0.05)
    assert classes.set_index("metric").loc["map50_95", "delta_repeat4_minus_natural"] == pytest.approx(0.1)


def test_complete_factorial_rows_keep_both_main_effects() -> None:
    metrics = ("precision", "recall", "map50", "map50_95")
    natural = {
        "aggregate": {metric: 0.2 for metric in metrics},
        "per_class": {"target": {metric: 0.2 for metric in metrics}},
    }
    exposure = {
        "aggregate": {metric: 0.25 for metric in metrics},
        "per_class": {"target": {metric: 0.25 for metric in metrics}},
    }
    resolution = {
        "aggregate": {metric: 0.3 for metric in metrics},
        "per_class": {"target": {metric: 0.3 for metric in metrics}},
    }

    aggregate, classes = three_cell_metric_rows(natural, exposure, resolution)

    row = aggregate.set_index("metric").loc["map50_95"]
    assert row["delta_repeat4_minus_natural640"] == pytest.approx(0.05)
    assert row["delta_natural1280_minus_natural640"] == pytest.approx(0.1)
    class_row = classes.set_index(["category", "metric"]).loc[("target", "map50_95")]
    assert class_row["natural_1280"] == pytest.approx(0.3)


def test_four_cell_factorial_reports_interaction_effect() -> None:
    effects = factorial_effects(0.20, 0.25, 0.30, 0.40)

    assert effects["exposure_effect_640"] == pytest.approx(0.05)
    assert effects["exposure_effect_1280"] == pytest.approx(0.10)
    assert effects["resolution_effect_natural"] == pytest.approx(0.10)
    assert effects["resolution_effect_repeat4"] == pytest.approx(0.15)
    assert effects["difference_in_differences"] == pytest.approx(0.05)


def test_four_cell_metric_rows_preserve_all_cells() -> None:
    metrics = ("precision", "recall", "map50", "map50_95")

    def evaluation(value: float) -> dict:
        return {
            "aggregate": {metric: value for metric in metrics},
            "per_class": {"target": {metric: value for metric in metrics}},
        }

    aggregate, classes = four_cell_metric_rows(
        evaluation(0.20), evaluation(0.25), evaluation(0.30), evaluation(0.40)
    )

    row = aggregate.set_index("metric").loc["map50_95"]
    assert row["repeat4_1280"] == pytest.approx(0.40)
    assert row["difference_in_differences"] == pytest.approx(0.05)
    class_row = classes.set_index(["category", "metric"]).loc[("target", "map50_95")]
    assert class_row["resolution_effect_repeat4"] == pytest.approx(0.15)


def test_error_taxonomy_separates_wrong_class_localization() -> None:
    rows = pd.DataFrame(
        {
            "best_any_iou_natural_640": [0.6, 0.7, 0.2],
            "best_any_category_natural_640": ["target", "boat", "target"],
        }
    )

    counts = localized_counts(rows, "natural_640", 0.5, "target")

    assert counts == {
        "same_class_localized": 1,
        "wrong_class_localized": 1,
        "any_class_localized": 2,
        "no_localized_prediction": 1,
    }
