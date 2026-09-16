from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeAlias

Box: TypeAlias = tuple[float, float, float, float]
IntBox: TypeAlias = tuple[int, int, int, int]


def xywh_to_xyxy(values: Sequence[float]) -> Box:
    if len(values) != 4:
        raise ValueError("bbox must contain four values")
    x, y, width, height = (float(value) for value in values)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    return x, y, x + width, y + height


def union_box(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ValueError("at least one target box is required")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def intersects(first: Box, second: Sequence[float]) -> bool:
    return (
        min(first[2], float(second[2])) > max(first[0], float(second[0]))
        and min(first[3], float(second[3])) > max(first[1], float(second[1]))
    )


def contains(outer: Sequence[float], inner: Box, tolerance: float = 1e-6) -> bool:
    return (
        inner[0] >= float(outer[0]) - tolerance
        and inner[1] >= float(outer[1]) - tolerance
        and inner[2] <= float(outer[2]) + tolerance
        and inner[3] <= float(outer[3]) + tolerance
    )


def _centered_interval(center: float, length: float, limit: int) -> tuple[int, int]:
    length = min(float(limit), length)
    start = max(0.0, min(float(limit) - length, center - length / 2.0))
    return math.floor(start), math.ceil(start + length)


def target_context_crop(
    image_width: int,
    image_height: int,
    target_boxes: Sequence[Box],
    active_boxes: Iterable[Box],
    context_factor: float,
    minimum_side: int,
) -> IntBox:
    """Return a deterministic target-group crop without cutting active boxes.

    The initial crop is square whenever the image permits it. If an active
    annotation crosses a crop boundary, the boundary is expanded until that
    annotation is fully included. This prevents cropped object fragments from
    becoming unlabeled false negatives.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if context_factor <= 0 or minimum_side <= 0:
        raise ValueError("context_factor and minimum_side must be positive")

    target_union = union_box(target_boxes)
    if not contains((0, 0, image_width, image_height), target_union):
        raise ValueError("target box lies outside the image")
    union_width = target_union[2] - target_union[0]
    union_height = target_union[3] - target_union[1]
    requested_side = max(float(minimum_side), context_factor * max(union_width, union_height))
    crop_width = min(float(image_width), max(union_width, requested_side))
    crop_height = min(float(image_height), max(union_height, requested_side))
    center_x = (target_union[0] + target_union[2]) / 2.0
    center_y = (target_union[1] + target_union[3]) / 2.0
    left, right = _centered_interval(center_x, crop_width, image_width)
    top, bottom = _centered_interval(center_y, crop_height, image_height)

    boxes = list(active_boxes)
    for box in boxes:
        if not contains((0, 0, image_width, image_height), box):
            raise ValueError("active box lies outside the image")

    # Expansion can reveal another crossing box, so iterate to a fixed point.
    for _ in range(len(boxes) + 1):
        crop = (left, top, right, bottom)
        crossing = [box for box in boxes if intersects(box, crop) and not contains(crop, box)]
        if not crossing:
            if not all(contains(crop, box) for box in target_boxes):
                raise AssertionError("final crop does not contain every target box")
            return crop
        left = max(0, math.floor(min([left, *[box[0] for box in crossing]])))
        top = max(0, math.floor(min([top, *[box[1] for box in crossing]])))
        right = min(image_width, math.ceil(max([right, *[box[2] for box in crossing]])))
        bottom = min(image_height, math.ceil(max([bottom, *[box[3] for box in crossing]])))

    raise RuntimeError("crop boundary expansion did not converge")


def box_to_crop_yolo(box: Box, crop: IntBox, class_index: int) -> str:
    if not contains(crop, box):
        raise ValueError("box is not fully contained by crop")
    left, top, right, bottom = crop
    crop_width = float(right - left)
    crop_height = float(bottom - top)
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop dimensions must be positive")
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    values = (
        ((x1 + x2) / 2.0 - left) / crop_width,
        ((y1 + y2) / 2.0 - top) / crop_height,
        width / crop_width,
        height / crop_height,
    )
    if any(value < -1e-6 or value > 1.0 + 1e-6 for value in values):
        raise ValueError("crop-relative box cannot be normalized safely")
    return f"{class_index} " + " ".join(f"{value:.8f}" for value in values)
