from __future__ import annotations

import pytest

from scripts.analyze_object_centric_crop_control import metric_rows


def test_metric_rows_compute_crop_minus_repeat_delta() -> None:
    repeat = {
        "aggregate": {"precision": 0.4, "recall": 0.5, "map50": 0.6, "map50_95": 0.3},
        "per_class": {
            "target": {"precision": 0.1, "recall": 0.2, "map50": 0.3, "map50_95": 0.0}
        },
    }
    crop = {
        "aggregate": {"precision": 0.5, "recall": 0.4, "map50": 0.7, "map50_95": 0.35},
        "per_class": {
            "target": {"precision": 0.2, "recall": 0.3, "map50": 0.4, "map50_95": 0.1}
        },
    }

    aggregate, classes = metric_rows(repeat, crop)

    aggregate_delta = aggregate.loc[
        aggregate["metric"] == "map50_95", "delta_crop4_minus_repeat4"
    ].item()
    target_delta = classes.loc[
        (classes["category"] == "target") & (classes["metric"] == "map50_95"),
        "delta_crop4_minus_repeat4",
    ].item()
    assert aggregate_delta == pytest.approx(0.05)
    assert target_delta == pytest.approx(0.1)
