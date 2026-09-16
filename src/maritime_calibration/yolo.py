from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .coco import active_categories


def category_maps(dataset: dict[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    """Return COCO-to-YOLO IDs and the contiguous YOLO class names."""
    categories = active_categories(dataset)
    if not categories:
        raise ValueError("no active categories remain after excluding ignored categories")
    coco_to_yolo = {
        int(category["id"]): class_index
        for class_index, category in enumerate(categories)
    }
    names = {
        class_index: str(category["name"])
        for class_index, category in enumerate(categories)
    }
    return coco_to_yolo, names


def annotation_to_yolo(
    annotation: dict[str, Any],
    image: dict[str, Any],
    class_index: int,
) -> str:
    image_width = float(image["width"])
    image_height = float(image["height"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image_id={image.get('id')} has invalid dimensions")

    x, y, width, height = [float(value) for value in annotation["bbox"]]
    if width <= 0 or height <= 0:
        raise ValueError(f"annotation_id={annotation.get('id')} has a non-positive box")
    tolerance = 1e-6
    if (
        x < -tolerance
        or y < -tolerance
        or x + width > image_width + tolerance
        or y + height > image_height + tolerance
    ):
        raise ValueError(f"annotation_id={annotation.get('id')} lies outside its image")

    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    values = (center_x, center_y, normalized_width, normalized_height)
    if any(value < -tolerance or value > 1.0 + tolerance for value in values):
        raise ValueError(f"annotation_id={annotation.get('id')} cannot be normalized safely")
    return f"{class_index} " + " ".join(f"{value:.8f}" for value in values)


def validate_image_files(dataset: dict[str, Any], image_dir: Path) -> None:
    missing_images: list[str] = []
    for image in dataset["images"]:
        file_name = Path(str(image["file_name"]))
        if file_name.is_absolute() or ".." in file_name.parts:
            raise ValueError(f"unsafe image file_name={file_name}")
        image_path = image_dir / file_name
        if not image_path.is_file():
            missing_images.append(str(image_path))
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} referenced images are missing; first: {missing_images[0]}"
        )


def prepare_yolo_split(
    dataset: dict[str, Any],
    labels_dir: Path,
    image_dir: Path | None = None,
) -> dict[str, Any]:
    """Write one YOLO label file per image without copying or changing images."""
    if image_dir is not None:
        validate_image_files(dataset, image_dir)
    coco_to_yolo, names = category_maps(dataset)
    images = {int(image["id"]): image for image in dataset["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped_ignored = 0
    for annotation in dataset["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in coco_to_yolo:
            skipped_ignored += 1
            continue
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"annotation references missing image_id={image_id}")
        annotations_by_image[image_id].append(annotation)

    labels_dir.mkdir(parents=True, exist_ok=False)
    box_count = 0
    for image_id, image in images.items():
        file_name = Path(str(image["file_name"]))
        if file_name.is_absolute() or ".." in file_name.parts:
            raise ValueError(f"unsafe image file_name={file_name}")
        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            class_index = coco_to_yolo[int(annotation["category_id"])]
            lines.append(annotation_to_yolo(annotation, image, class_index))
        box_count += len(lines)
        label_path = labels_dir / file_name.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "images": len(images),
        "boxes": box_count,
        "skipped_ignored_annotations": skipped_ignored,
        "coco_to_yolo": {str(key): value for key, value in coco_to_yolo.items()},
        "names": names,
    }


def write_dataset_yaml(
    output: Path,
    dataset_root: Path,
    names: dict[int, str],
) -> None:
    payload = {
        "path": dataset_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_image_list(dataset: dict[str, Any], image_dir: Path, output: Path) -> None:
    validate_image_files(dataset, image_dir)
    paths = [
        (image_dir / str(image["file_name"])).resolve().as_posix()
        for image in dataset["images"]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + "\n", encoding="utf-8")


def write_detector_yaml(
    output: Path,
    train_list: Path,
    dev_list: Path,
    names: dict[int, str],
) -> None:
    payload = {
        "train": train_list.resolve().as_posix(),
        "val": dev_list.resolve().as_posix(),
        "names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
