import pytest

from maritime_calibration.matching import build_detection_table, xywh_iou
from scripts.analyze_resolution_by_gt_size import matched_annotation_ids
from scripts.audit_rare_class_failure import (
    best_prediction_for_annotation,
    failure_label,
    optional_median,
)


def test_xywh_iou():
    assert xywh_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert xywh_iou([0, 0, 10, 10], [20, 20, 2, 2]) == 0.0


def test_greedy_matching_marks_duplicate_as_false_positive():
    ground_truth = {
        "images": [
            {
                "id": 1,
                "width": 100,
                "height": 100,
                "meta": {
                    "height_above_takeoff(meter)": 80,
                    "gimbal_pitch(degrees)": 45,
                },
            }
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
        "categories": [{"id": 1, "name": "target"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [11, 11, 20, 20], "score": 0.8},
    ]
    table, summary = build_detection_table(ground_truth, predictions)
    assert table["is_tp"].tolist() == [1, 0]
    assert table["altitude"].tolist() == [80.0, 80.0]
    assert table["source_group"].isna().all()
    assert table["relative_area"].iloc[0] == pytest.approx(0.04)
    assert summary["matched_ground_truth"] == 1


def test_matched_annotation_ids_returns_the_selected_ground_truth_id():
    ground_truth = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 12, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10]},
        ],
        "categories": [{"id": 1, "name": "target"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.9}
    ]

    assert matched_annotation_ids(ground_truth, predictions, 0.5) == {12}


def test_rare_class_failure_label_distinguishes_wrong_class_localization():
    annotation = {"bbox": [10, 10, 20, 20]}
    predictions = [
        {"bbox": [10, 10, 20, 20], "category_id": 5, "score": 0.7},
        {"bbox": [60, 60, 10, 10], "category_id": 4, "score": 0.8},
    ]

    best, overlap = best_prediction_for_annotation(annotation, predictions)

    assert overlap == 1.0
    assert failure_label(4, best, overlap, 0.5) == "wrong_class_localized"
    assert failure_label(4, None, 0.0, 0.5) == "no_localized_prediction"
    assert optional_median([None, 10.0, 20.0]) == 15.0
    assert optional_median([None]) is None
