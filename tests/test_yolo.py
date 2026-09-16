import pytest

from maritime_calibration.yolo import annotation_to_yolo, category_maps


def test_category_mapping_excludes_ignored_and_is_contiguous():
    dataset = {
        "categories": [
            {"id": 0, "name": "ignored"},
            {"id": 1, "name": "swimmer"},
            {"id": 5, "name": "buoy"},
        ]
    }
    coco_to_yolo, names = category_maps(dataset)
    assert coco_to_yolo == {1: 0, 5: 1}
    assert names == {0: "swimmer", 1: "buoy"}


def test_annotation_conversion_uses_xywh_center_coordinates():
    image = {"id": 2, "width": 100, "height": 50}
    annotation = {"id": 4, "bbox": [10, 5, 20, 10]}
    assert annotation_to_yolo(annotation, image, 3) == (
        "3 0.20000000 0.20000000 0.20000000 0.20000000"
    )


def test_annotation_outside_image_is_rejected():
    image = {"id": 2, "width": 100, "height": 50}
    annotation = {"id": 4, "bbox": [95, 5, 20, 10]}
    with pytest.raises(ValueError, match="outside"):
        annotation_to_yolo(annotation, image, 0)
