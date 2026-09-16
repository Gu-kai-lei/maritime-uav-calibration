from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FRAME_PATTERN = re.compile(r"^(?P<source>.+)_(?P<frame>\d{6})$")
PROJECT_CATEGORIES = [
    {"supercategory": "ignored", "id": 0, "name": "ignored"},
    {"supercategory": "person", "id": 1, "name": "swimmer"},
    {"supercategory": "boat", "id": 2, "name": "boat"},
    {"supercategory": "boat", "id": 3, "name": "jetski"},
    {"supercategory": "object", "id": 4, "name": "life_saving_appliances"},
    {"supercategory": "object", "id": 5, "name": "buoy"},
]


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_frame_name(file_name: str) -> tuple[str, int]:
    match = FRAME_PATTERN.match(Path(file_name).stem)
    if not match:
        raise ValueError(f"cannot parse source video and frame index from {file_name!r}")
    return match.group("source"), int(match.group("frame"))


def normalized_source(value: str) -> str:
    raw = value.split(":", 1)[-1]
    return Path(raw).stem.lower()


def existing_source_names(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for role in payload["coverage"]["roles"].values():
        for row in role["groups"]:
            names.add(normalized_source(str(row["source_group"])))
    return names


def validate_coco(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    image_ids = [int(row["id"]) for row in images]
    file_names = [str(row["file_name"]) for row in images]
    category_ids = [int(row["id"]) for row in categories]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate COCO image id")
    if len(file_names) != len(set(file_names)):
        raise ValueError("duplicate COCO filename")
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("duplicate COCO category id")
    image_by_id = {int(row["id"]): row for row in images}
    category_set = set(category_ids)
    invalid_boxes: list[int] = []
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_by_id:
            raise ValueError(f"annotation {annotation['id']} references missing image {image_id}")
        if category_id not in category_set:
            raise ValueError(
                f"annotation {annotation['id']} references missing category {category_id}"
            )
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            invalid_boxes.append(int(annotation["id"]))
            continue
        x, y, width, height = bbox
        image = image_by_id[image_id]
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > float(image["width"]) + 1e-6
            or y + height > float(image["height"]) + 1e-6
        ):
            invalid_boxes.append(int(annotation["id"]))
    if invalid_boxes:
        raise ValueError(f"invalid bounding boxes: {invalid_boxes[:10]}")
    return {
        "unique_image_ids": True,
        "unique_file_names": True,
        "unique_category_ids": True,
        "missing_annotation_image_references": 0,
        "missing_annotation_category_references": 0,
        "invalid_bounding_boxes": 0,
    }


def select_fixed_rate(
    images: list[dict[str, Any]], prefix: str, stride: int
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for image in images:
        file_name = str(image["file_name"])
        if not file_name.startswith(prefix):
            continue
        source, frame_index = parse_frame_name(file_name)
        grouped[source].append((frame_index, image))
    selected: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda item: (item[0], str(item[1]["file_name"])))
        for position, (frame_index, image) in enumerate(rows):
            if position % stride != 0:
                continue
            selected.append(image)
            metadata[int(image["id"])] = {
                "source_group": source,
                "frame_index": frame_index,
                "position_in_source": position,
                "video_file_expected": f"{source}.MP4",
            }
    return selected, metadata


def write_outputs(args: argparse.Namespace) -> dict[str, Any]:
    annotations_path = args.annotations.resolve()
    actual_md5 = digest(annotations_path, "md5")
    actual_sha256 = digest(annotations_path, "sha256")
    if actual_md5.lower() != args.expected_md5.lower():
        raise ValueError(
            f"annotation MD5 mismatch: expected {args.expected_md5}, got {actual_md5}"
        )
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    validation = validate_coco(payload)
    categories = {str(row["name"]): int(row["id"]) for row in payload["categories"]}
    if args.target_category not in categories:
        raise ValueError(
            f"target category {args.target_category!r} not found; got {sorted(categories)}"
        )
    target_id = categories[args.target_category]
    selected_images, selected_metadata = select_fixed_rate(
        payload["images"], args.test_prefix, args.stride
    )
    selected_ids = {int(row["id"]) for row in selected_images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in selected_ids:
            annotations_by_image[image_id].append(annotation)
    selected_annotations = [
        annotation
        for image_id in sorted(annotations_by_image)
        for annotation in annotations_by_image[image_id]
    ]
    existing = existing_source_names(args.existing_inventory)
    selected_sources = {row["source_group"] for row in selected_metadata.values()}
    overlap = sorted(source for source in selected_sources if normalized_source(source) in existing)

    manifest_rows: list[dict[str, Any]] = []
    target_annotations = 0
    target_positive_images = 0
    target_negative_images = 0
    target_by_source: Counter[str] = Counter()
    for image in sorted(selected_images, key=lambda row: str(row["file_name"])):
        image_id = int(image["id"])
        rows = annotations_by_image.get(image_id, [])
        target_rows = [row for row in rows if int(row["category_id"]) == target_id]
        metadata = selected_metadata[image_id]
        target_annotations += len(target_rows)
        if target_rows:
            target_positive_images += 1
            target_by_source[metadata["source_group"]] += len(target_rows)
        else:
            target_negative_images += 1
        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "source_group": metadata["source_group"],
                "video_file_expected": metadata["video_file_expected"],
                "frame_index": metadata["frame_index"],
                "position_in_source": metadata["position_in_source"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "target_annotation_count": len(target_rows),
                "target_bboxes_xywh": [row["bbox"] for row in target_rows],
                "extracted_path": None,
                "extracted_sha256": None,
            }
        )

    all_test_images = [
        row for row in payload["images"] if str(row["file_name"]).startswith(args.test_prefix)
    ]
    all_test_image_by_id = {int(row["id"]): row for row in all_test_images}
    all_test_ids = {int(row["id"]) for row in all_test_images}
    all_test_sources = {parse_frame_name(str(row["file_name"]))[0] for row in all_test_images}
    all_test_target_rows = [
        row
        for row in payload["annotations"]
        if int(row["image_id"]) in all_test_ids and int(row["category_id"]) == target_id
    ]
    all_test_target_image_ids = {int(row["image_id"]) for row in all_test_target_rows}
    all_test_target_sources = {
        parse_frame_name(str(all_test_image_by_id[image_id]["file_name"]))[0]
        for image_id in all_test_target_image_ids
    }

    ready = (
        bool(target_annotations)
        and bool(target_negative_images)
        and not overlap
        and validation["invalid_bounding_boxes"] == 0
    )
    inventory = {
        "schema_version": 1,
        "status": "ready_for_official_video_archive_acquisition" if ready else "blocked",
        "dataset": {
            "name": "MOBDrone",
            "version": "1.0.0",
            "doi": "10.5281/zenodo.5996890",
            "annotations": str(annotations_path),
            "annotations_bytes": annotations_path.stat().st_size,
            "annotations_md5": actual_md5,
            "annotations_sha256": actual_sha256,
            "categories": categories,
            "images": len(payload["images"]),
            "annotations_count": len(payload["annotations"]),
        },
        "coco_validation": validation,
        "official_recommended_test": {
            "prefix": args.test_prefix,
            "images": len(all_test_images),
            "source_groups": len(all_test_sources),
            "target_category": args.target_category,
            "target_annotations": len(all_test_target_rows),
            "target_positive_images": len(all_test_target_image_ids),
            "target_positive_source_groups": len(all_test_target_sources),
        },
        "fixed_rate_selection": {
            "stride_frames": args.stride,
            "images": len(selected_images),
            "source_groups": len(selected_sources),
            "target_annotations": target_annotations,
            "target_positive_images": target_positive_images,
            "target_negative_images": target_negative_images,
            "target_positive_source_groups": len(target_by_source),
            "target_annotations_by_source": dict(sorted(target_by_source.items())),
            "required_video_files": sorted(
                {row["video_file_expected"] for row in selected_metadata.values()}
            ),
        },
        "source_overlap": {
            "existing_inventory": str(args.existing_inventory.resolve())
            if args.existing_inventory
            else None,
            "normalized_overlap_count": len(overlap),
            "overlap": overlap,
        },
        "frame_pixels_downloaded_or_extracted": False,
        "model_or_prediction_files_accessed": False,
        "training_or_inference_performed": False,
        "ready_for_video_acquisition": ready,
    }
    manifest = {
        "schema_version": 1,
        "status": "pre_extraction_frozen",
        "selection_rule": {
            "test_prefix": args.test_prefix,
            "stride_frames": args.stride,
            "source_unit": "complete MOBDrone video prefix before the terminal frame index",
            "model_result_dependent_sampling": False,
        },
        "annotation_source": {
            "path": str(annotations_path),
            "md5": actual_md5,
            "sha256": actual_sha256,
        },
        "target_category": {"external": args.target_category, "project": "life_saving_appliances"},
        "frames": manifest_rows,
    }
    subset = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": payload["categories"],
    }
    project_target_annotations = []
    for annotation in selected_annotations:
        if int(annotation["category_id"]) != target_id:
            continue
        mapped = dict(annotation)
        mapped["category_id"] = 4
        mapped["external_category_id"] = target_id
        mapped["external_category_name"] = args.target_category
        project_target_annotations.append(mapped)
    project_contract = {
        "images": selected_images,
        "annotations": project_target_annotations,
        "categories": PROJECT_CATEGORIES,
        "external_mapping": {
            args.target_category: "life_saving_appliances",
            "annotation_scope": "target annotations only; non-target external classes are excluded rather than coerced",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "frame_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "instances_fixed_rate_test.json").write_text(
        json.dumps(subset, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (args.output_dir / "instances_fixed_rate_project_contract.json").write_text(
        json.dumps(project_contract, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "frame_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "image_id",
            "file_name",
            "source_group",
            "video_file_expected",
            "frame_index",
            "position_in_source",
            "width",
            "height",
            "target_annotation_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--existing-inventory", type=Path)
    parser.add_argument("--test-prefix", default="DJI_0804")
    parser.add_argument("--target-category", default="life_buoy")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = write_outputs(parse_args())
    print(json.dumps(result["fixed_rate_selection"], indent=2))
    print(f"status={result['status']}")
