from __future__ import annotations

import pandas as pd

from scripts.analyze_oracle_target_crops import (
    crop_relative_xywh,
    greedy_same_class_matches,
    json_records,
    summarize,
)


def test_crop_relative_xywh_translates_without_rescaling() -> None:
    annotation = {"bbox": [110, 220, 30, 40]}
    assert crop_relative_xywh(annotation, (100, 200, 500, 600)) == [10.0, 20.0, 30.0, 40.0]


def test_greedy_same_class_matches_each_annotation_once() -> None:
    annotations = [
        {"id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "bbox": [20, 20, 10, 10]},
    ]
    predictions = [
        {"bbox": [0, 0, 10, 10], "score": 0.9},
        {"bbox": [0, 0, 10, 10], "score": 0.8},
        {"bbox": [20, 20, 10, 10], "score": 0.7},
    ]
    assert greedy_same_class_matches(annotations, predictions, 0.5) == {1, 2}


def test_summarize_reports_frozen_recall_and_localization() -> None:
    details = pd.DataFrame(
        [
            {
                "model": "crop4_1280",
                "context_factor": 8.0,
                "same_class_matched_iou_0_50": True,
                "best_same_iou": 0.7,
                "best_any_iou": 0.7,
                "target_predictions_in_crop": 1,
                "target_prediction_max_score": 0.4,
                "target_width_at_1280": 80.0,
                "target_height_at_1280": 60.0,
                "best_any_category": "life_saving_appliances",
            },
            {
                "model": "crop4_1280",
                "context_factor": 8.0,
                "same_class_matched_iou_0_50": False,
                "best_same_iou": 0.0,
                "best_any_iou": 0.2,
                "target_predictions_in_crop": 0,
                "target_prediction_max_score": None,
                "target_width_at_1280": 40.0,
                "target_height_at_1280": 30.0,
                "best_any_category": "boat",
            },
        ]
    )
    row = summarize(details, "life_saving_appliances").iloc[0]
    assert row["target_ground_truth"] == 2
    assert row["same_class_recall_iou_0_50"] == 0.5
    assert row["any_class_localized_iou_0_10"] == 2


def test_json_records_converts_nan_to_null() -> None:
    assert json_records(pd.DataFrame([{"score": float("nan")}])) == [{"score": None}]
