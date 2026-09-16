from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ALTITUDE_ALIASES = [
    "meta.altitude",
    "meta.height_above_takeoff(meter)",
    "altitude",
]
GIMBAL_PITCH_ALIASES = [
    "meta.gimbal_pitch",
    "meta.gimbal_pitch(degrees)",
    "gimbal_pitch",
]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def metadata_value(image: dict[str, Any], aliases: Iterable[str]) -> float | None:
    containers = [image]
    if isinstance(image.get("meta"), dict):
        containers.append(image["meta"])

    for alias in aliases:
        for container in containers:
            value = _nested_get(container, alias)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def source_group(image: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = _nested_get(image, key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return None


def active_categories(
    dataset: dict[str, Any],
    ignored_ids: Iterable[int] = (0,),
    ignored_names: Iterable[str] = ("ignored", "ignore"),
) -> list[dict[str, Any]]:
    id_set = set(ignored_ids)
    name_set = {name.lower() for name in ignored_names}
    return sorted(
        [
            category
            for category in dataset["categories"]
            if int(category["id"]) not in id_set
            and str(category.get("name", "")).lower() not in name_set
        ],
        key=lambda category: int(category["id"]),
    )


def metadata_complete_subset(dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    retained_images = [
        image
        for image in dataset["images"]
        if metadata_value(image, ALTITUDE_ALIASES) is not None
        and metadata_value(image, GIMBAL_PITCH_ALIASES) is not None
    ]
    retained_ids = {image["id"] for image in retained_images}
    retained_annotations = [
        annotation
        for annotation in dataset["annotations"]
        if annotation["image_id"] in retained_ids
    ]
    output = {
        key: value
        for key, value in dataset.items()
        if key not in {"images", "annotations"}
    }
    output["images"] = retained_images
    output["annotations"] = retained_annotations
    report = {
        "images_before": len(dataset["images"]),
        "images_after": len(retained_images),
        "annotations_before": len(dataset["annotations"]),
        "annotations_after": len(retained_annotations),
    }
    return output, report


def audit_coco(
    dataset: dict[str, Any],
    metadata_aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    images = dataset.get("images")
    annotations = dataset.get("annotations")
    categories = dataset.get("categories")

    if not isinstance(images, list):
        raise ValueError("COCO JSON must contain an images list")
    if not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain an annotations list")
    if not isinstance(categories, list):
        raise ValueError("COCO JSON must contain a categories list")

    image_ids = [image.get("id") for image in images]
    category_ids = [category.get("id") for category in categories]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    if len(set(image_ids)) != len(image_ids):
        errors.append("duplicate image IDs")
    if len(set(category_ids)) != len(category_ids):
        errors.append("duplicate category IDs")
    if None not in annotation_ids and len(set(annotation_ids)) != len(annotation_ids):
        errors.append("duplicate annotation IDs")

    image_by_id = {image.get("id"): image for image in images}
    category_id_set = set(category_ids)
    class_counts: Counter[int] = Counter()
    semantic_box_counts: Counter[tuple[Any, int, tuple[float, ...]]] = Counter()
    invalid_boxes = 0
    out_of_bounds_boxes = 0

    for index, annotation in enumerate(annotations):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if image_id not in image_by_id:
            errors.append(f"annotation[{index}] references missing image_id={image_id}")
            continue
        if category_id not in category_id_set:
            errors.append(f"annotation[{index}] references missing category_id={category_id}")
        else:
            class_counts[int(category_id)] += 1
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
            continue
        try:
            x, y, width, height = [float(value) for value in bbox]
        except (TypeError, ValueError):
            invalid_boxes += 1
            continue
        if category_id in category_id_set:
            semantic_box_counts[(image_id, int(category_id), (x, y, width, height))] += 1
        if width <= 0 or height <= 0:
            invalid_boxes += 1
            continue
        image = image_by_id[image_id]
        image_width = float(image.get("width", 0) or 0)
        image_height = float(image.get("height", 0) or 0)
        if image_width > 0 and image_height > 0:
            tolerance = 1.0
            outside = (
                x < -tolerance
                or y < -tolerance
                or x + width > image_width + tolerance
                or y + height > image_height + tolerance
            )
            if outside:
                out_of_bounds_boxes += 1

    if invalid_boxes:
        errors.append(f"{invalid_boxes} invalid bounding boxes")
    if out_of_bounds_boxes:
        warnings.append(f"{out_of_bounds_boxes} boxes extend outside image bounds")
    duplicate_box_groups = sum(count > 1 for count in semantic_box_counts.values())
    duplicate_box_extras = sum(count - 1 for count in semantic_box_counts.values() if count > 1)
    if duplicate_box_extras:
        warnings.append(
            f"{duplicate_box_extras} exact duplicate annotations across "
            f"{duplicate_box_groups} image/category/box groups"
        )

    aliases = metadata_aliases or {
        "altitude": ALTITUDE_ALIASES,
        "gimbal_pitch": GIMBAL_PITCH_ALIASES,
    }
    metadata_coverage = {
        name: sum(metadata_value(image, field_aliases) is not None for image in images)
        for name, field_aliases in aliases.items()
    }

    category_names = {
        str(category.get("id")): category.get("name", "") for category in categories
    }
    empty_images = len(images) - len({annotation.get("image_id") for annotation in annotations})

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": category_names,
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "empty_images": empty_images,
        "exact_duplicate_box_groups": duplicate_box_groups,
        "exact_duplicate_annotations": duplicate_box_extras,
        "metadata_coverage": metadata_coverage,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
