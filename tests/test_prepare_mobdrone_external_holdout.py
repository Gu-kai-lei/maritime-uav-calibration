from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mobdrone_external_holdout.py"
SPEC = importlib.util.spec_from_file_location("prepare_mobdrone_external_holdout", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_frame_name_preserves_source_suffix() -> None:
    assert MODULE.parse_frame_name("DJI_0804_0001_30m_1_000030.PNG") == (
        "DJI_0804_0001_30m_1",
        30,
    )


def test_parse_frame_name_rejects_missing_index() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        MODULE.parse_frame_name("DJI_0804.PNG")


def test_fixed_rate_selection_is_per_source_and_deterministic() -> None:
    images = [
        {"id": 1, "file_name": "DJI_0804_A_000001.PNG"},
        {"id": 2, "file_name": "DJI_0804_A_000000.PNG"},
        {"id": 3, "file_name": "DJI_0804_A_000002.PNG"},
        {"id": 4, "file_name": "DJI_0804_B_000004.PNG"},
        {"id": 5, "file_name": "DJI_0915_C_000000.PNG"},
    ]
    selected, metadata = MODULE.select_fixed_rate(images, "DJI_0804", 2)
    assert [row["id"] for row in selected] == [2, 3, 4]
    assert metadata[2]["position_in_source"] == 0
    assert metadata[3]["position_in_source"] == 2
    assert metadata[4]["video_file_expected"] == "DJI_0804_B.MP4"


def test_validate_coco_rejects_out_of_bounds_box() -> None:
    payload = {
        "images": [{"id": 1, "file_name": "A_000000.PNG", "width": 10, "height": 10}],
        "categories": [{"id": 5, "name": "life_buoy"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [9, 9, 2, 2]}],
    }
    with pytest.raises(ValueError, match="invalid bounding boxes"):
        MODULE.validate_coco(payload)
