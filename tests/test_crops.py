import pytest

from maritime_calibration.crops import (
    box_to_crop_yolo,
    target_context_crop,
    union_box,
    xywh_to_xyxy,
)


def test_xywh_and_union_conversion():
    boxes = [xywh_to_xyxy([10, 20, 5, 6]), xywh_to_xyxy([2, 4, 3, 7])]
    assert boxes == [(10.0, 20.0, 15.0, 26.0), (2.0, 4.0, 5.0, 11.0)]
    assert union_box(boxes) == (2.0, 4.0, 15.0, 26.0)


def test_crop_is_shifted_inside_image_at_boundary():
    crop = target_context_crop(
        100,
        80,
        target_boxes=[(0.0, 0.0, 10.0, 10.0)],
        active_boxes=[(0.0, 0.0, 10.0, 10.0)],
        context_factor=2,
        minimum_side=40,
    )
    assert crop == (0, 0, 40, 40)


def test_crop_expands_to_avoid_cutting_active_neighbor():
    crop = target_context_crop(
        200,
        200,
        target_boxes=[(90.0, 90.0, 110.0, 110.0)],
        active_boxes=[(90.0, 90.0, 110.0, 110.0), (115.0, 95.0, 145.0, 105.0)],
        context_factor=2,
        minimum_side=40,
    )
    assert crop == (80, 80, 145, 120)


def test_crop_relative_yolo_label():
    assert box_to_crop_yolo((10.0, 10.0, 30.0, 30.0), (0, 0, 40, 40), 3) == (
        "3 0.50000000 0.50000000 0.50000000 0.50000000"
    )


def test_crop_rejects_target_outside_image():
    with pytest.raises(ValueError, match="outside"):
        target_context_crop(
            100,
            100,
            target_boxes=[(95.0, 95.0, 105.0, 105.0)],
            active_boxes=[],
            context_factor=2,
            minimum_side=40,
        )
