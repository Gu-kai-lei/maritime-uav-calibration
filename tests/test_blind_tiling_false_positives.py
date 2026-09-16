from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_blind_tiling_false_positives import (
    classify_prediction,
    greedy_match_indices,
    grid_geometry,
    internal_tile_boundaries,
)


def test_greedy_match_leaves_second_prediction_as_duplicate() -> None:
    matched = greedy_match_indices(np.array([[0.90, 0.80]], dtype=float))
    assert matched.tolist() == [0, -1]


@pytest.mark.parametrize(
    ("matched", "target_iou", "other_iou", "expected"),
    [
        (0, 0.90, 0.00, "true_positive"),
        (-1, 0.80, 0.00, "duplicate_on_matched_target"),
        (-1, 0.30, 0.80, "target_localization_error"),
        (-1, 0.00, 0.80, "overlaps_other_class"),
        (-1, 0.00, 0.30, "near_other_class"),
        (-1, 0.00, 0.00, "background"),
    ],
)
def test_taxonomy_precedence(
    matched: int, target_iou: float, other_iou: float, expected: str
) -> None:
    assert classify_prediction(matched, target_iou, other_iou) == expected


def test_internal_tile_boundaries_include_starts_and_ends() -> None:
    boundaries = internal_tile_boundaries(1000, 384)
    assert boundaries == [288, 384, 576, 616, 672, 960]


def test_grid_geometry_reports_crossing_without_claiming_tile_origin() -> None:
    distance, crosses = grid_geometry(np.array([280, 100, 300, 120]), 1000, 800, 384)
    assert distance == pytest.approx(2 / 384)
    assert crosses is True
