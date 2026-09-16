from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_blind_tiling_sensitivity import evaluate_predictions, match_predictions
from scripts.export_blind_tiled_predictions import (
    class_aware_nms_indices,
    tile_starts,
    top_score_indices,
    translate_xyxy,
)


def test_tile_starts_cover_image_and_anchor_far_edge() -> None:
    starts = tile_starts(1000, 384, 0.25)
    assert starts[0] == 0
    assert starts[-1] == 616
    assert all(next_start <= start + 384 for start, next_start in zip(starts, starts[1:]))


def test_tile_starts_single_partial_tile() -> None:
    assert tile_starts(320, 384, 0.25) == [0]


def test_translate_xyxy_offsets_and_clips() -> None:
    translated = translate_xyxy(np.array([[1, 2, 90, 100]]), 50, 60, 120, 130)
    assert translated.tolist() == [[51.0, 62.0, 120.0, 130.0]]


def test_class_aware_nms_keeps_overlapping_different_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 0, 1])
    kept = class_aware_nms_indices(boxes, scores, classes, 0.5)
    assert set(kept.tolist()) == {0, 2}


def test_top_score_cap_is_stable() -> None:
    assert top_score_indices(np.array([0.1, 0.9, 0.8]), 2).tolist() == [1, 2]


def test_match_predictions_requires_same_class() -> None:
    iou = np.array([[1.0, 1.0]])
    correct = match_predictions(np.array([0, 1]), np.array([0]), iou)
    assert correct[0].all()
    assert not correct[1].any()


def test_perfect_predictions_have_unit_ap() -> None:
    dataset = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "life_saving_appliances"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    metrics = evaluate_predictions(dataset, predictions, 1)
    assert metrics["aggregate"]["map50_95"] == pytest.approx(0.995)
    assert metrics["target"]["recall_iou_0_50"] == 1.0
