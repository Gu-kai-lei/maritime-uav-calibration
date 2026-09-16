from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_mobdrone_external_holdout.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_mobdrone_external_holdout", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frame_source_parses_terminal_frame_index() -> None:
    assert MODULE.frame_source("DJI_0804_0018_40m_000030.PNG") == "DJI_0804_0018_40m"


def test_frame_source_rejects_unversioned_name() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        MODULE.frame_source("frame.PNG")


def test_target_only_does_not_coerce_other_classes() -> None:
    rows = [
        {"category_id": 4, "image_id": 1},
        {"category_id": 2, "image_id": 1},
    ]
    assert MODULE.target_only(rows, 4) == [{"category_id": 4, "image_id": 1}]


def test_target_operating_counts_includes_negative_frames() -> None:
    dataset = {
        "images": [
            {"id": 1, "file_name": "A_000000.PNG"},
            {"id": 2, "file_name": "A_000030.PNG"},
        ],
        "annotations": [{"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10]}],
    }
    predictions = [
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "category_id": 4, "bbox": [0, 0, 2, 2], "score": 0.1},
    ]
    result = MODULE.target_operating_counts(dataset, predictions, 4)["overall"]
    assert result["target_matches_iou_0_50"] == 1
    assert result["target_negative_images"] == 1
    assert result["target_predictions_on_negative_images"] == 1
    assert result["negative_image_target_prediction_rate"] == 1.0


def test_mobdrone_protocol_config_is_parseable_and_nonselecting() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mobdrone_external_holdout.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["frozen_inference_arms"]["arms"] == [
        "full_frame_1280",
        "blind_tile_384",
        "blind_tile_768",
    ]
    assert config["observed_execution"]["official_validation_used"] is False
    assert config["observed_execution"]["configuration_selected"] is False
