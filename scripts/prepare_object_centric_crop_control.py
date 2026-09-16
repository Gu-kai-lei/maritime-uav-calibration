from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.crops import (
    box_to_crop_yolo,
    contains,
    intersects,
    target_context_crop,
    xywh_to_xyxy,
)
from maritime_calibration.yolo import category_maps, file_sha256


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_hash(lines: list[str]) -> str:
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_file_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image file_name={value}")
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic matched-exposure target-centric crop control"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--source-image-root", required=True, type=Path)
    parser.add_argument("--source-train-list", required=True, type=Path)
    parser.add_argument("--original-wrapper-image-root", required=True, type=Path)
    parser.add_argument("--detector-dev-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-train-list", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--context-factors", nargs="+", type=float, default=[8.0, 12.0, 16.0])
    parser.add_argument("--minimum-side", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--forbidden-annotations", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing derived data: {args.output_root}")
    if len(args.context_factors) != 3:
        raise ValueError("exactly three context factors are required for matched Crop4 exposure")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100")

    annotations = load_json(args.annotations)
    coco_to_yolo, names = category_maps(annotations)
    category_by_name = {
        str(category["name"]): int(category["id"])
        for category in active_categories(annotations)
    }
    if args.target not in category_by_name:
        raise ValueError(f"target category is not active: {args.target}")
    target_category_id = category_by_name[args.target]
    target_class_index = coco_to_yolo[target_category_id]

    source_lines = load_lines(args.source_train_list)
    if len(source_lines) != len(set(source_lines)):
        raise ValueError("source training list contains duplicate paths")
    source_by_basename: dict[str, str] = {}
    for line in source_lines:
        basename = Path(line).name
        if basename in source_by_basename:
            raise ValueError(f"ambiguous source basename: {basename}")
        source_by_basename[basename] = line

    images = {int(image["id"]): image for image in annotations["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id in coco_to_yolo:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    annotation_basenames = {safe_file_name(str(image["file_name"])) for image in images.values()}
    if annotation_basenames != set(source_by_basename):
        missing_from_list = sorted(annotation_basenames - set(source_by_basename))
        missing_from_annotations = sorted(set(source_by_basename) - annotation_basenames)
        raise ValueError(
            "annotation/list membership differs: "
            f"missing_from_list={len(missing_from_list)}, "
            f"missing_from_annotations={len(missing_from_annotations)}"
        )

    forbidden_basenames: set[str] = set()
    forbidden_hashes: dict[str, str] = {}
    for forbidden_path in args.forbidden_annotations:
        forbidden = load_json(forbidden_path)
        forbidden_hashes[str(forbidden_path.resolve())] = file_sha256(forbidden_path)
        forbidden_basenames.update(
            safe_file_name(str(image["file_name"])) for image in forbidden["images"]
        )
    overlap = annotation_basenames & forbidden_basenames
    if overlap:
        raise ValueError(f"detector-train overlaps forbidden roles; first={sorted(overlap)[0]}")

    missing_source_images = [
        basename
        for basename in annotation_basenames
        if not (args.source_image_root / basename).is_file()
    ]
    missing_wrapper_images = [
        basename
        for basename in annotation_basenames
        if not (args.original_wrapper_image_root / basename).is_file()
    ]
    if missing_source_images or missing_wrapper_images:
        raise FileNotFoundError(
            f"missing source images={len(missing_source_images)}, "
            f"missing wrapper images={len(missing_wrapper_images)}"
        )

    rare_image_ids = sorted(
        image_id
        for image_id, items in annotations_by_image.items()
        if any(int(item["category_id"]) == target_category_id for item in items)
    )
    expected_added = len(rare_image_ids) * len(args.context_factors)
    temp_root = args.output_root.with_name(f"{args.output_root.name}.building")
    if temp_root.exists():
        raise FileExistsError(f"stale build directory requires inspection: {temp_root}")
    image_output = temp_root / "images" / "train"
    label_output = temp_root / "labels" / "train"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    generated_paths: list[str] = []
    class_box_counts: Counter[int] = Counter()
    target_boxes_added = 0
    boundary_expansion_count = 0

    for image_id in rare_image_ids:
        image_record = images[image_id]
        basename = safe_file_name(str(image_record["file_name"]))
        width = int(image_record["width"])
        height = int(image_record["height"])
        items = sorted(annotations_by_image[image_id], key=lambda item: int(item["id"]))
        active_boxes = [xywh_to_xyxy(item["bbox"]) for item in items]
        target_items = [item for item in items if int(item["category_id"]) == target_category_id]
        target_boxes = [xywh_to_xyxy(item["bbox"]) for item in target_items]
        source_path = args.source_image_root / basename

        with Image.open(source_path) as source_image:
            source_image.load()
            if source_image.size != (width, height):
                raise ValueError(
                    f"metadata/JPEG dimensions differ for {basename}: "
                    f"metadata={(width, height)}, jpeg={source_image.size}"
                )
            source_rgb = source_image.convert("RGB")
            for variant_index, context_factor in enumerate(args.context_factors, start=1):
                crop = target_context_crop(
                    width,
                    height,
                    target_boxes,
                    active_boxes,
                    context_factor,
                    args.minimum_side,
                )
                initial_without_neighbors = target_context_crop(
                    width,
                    height,
                    target_boxes,
                    target_boxes,
                    context_factor,
                    args.minimum_side,
                )
                boundary_expansion_count += int(crop != initial_without_neighbors)
                included: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
                for item, box in zip(items, active_boxes, strict=True):
                    if intersects(box, crop):
                        if not contains(crop, box):
                            raise AssertionError("crop cuts an active annotation")
                        included.append((item, box))
                included_target_ids = {
                    int(item["id"])
                    for item, _ in included
                    if int(item["category_id"]) == target_category_id
                }
                expected_target_ids = {int(item["id"]) for item in target_items}
                if included_target_ids != expected_target_ids:
                    raise AssertionError("crop does not preserve all target instances")

                crop_name = (
                    f"crop4_i{image_id}_v{variant_index}_f{context_factor:g}_{Path(basename).stem}.jpg"
                )
                label_name = Path(crop_name).with_suffix(".txt").name
                crop_path = image_output / crop_name
                label_path = label_output / label_name
                derived = source_rgb.crop(crop)
                derived.save(
                    crop_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                    optimize=False,
                    progressive=False,
                )
                with Image.open(crop_path) as check:
                    check.verify()

                label_lines: list[str] = []
                for item, box in included:
                    class_index = coco_to_yolo[int(item["category_id"])]
                    label_lines.append(box_to_crop_yolo(box, crop, class_index))
                    class_box_counts[class_index] += 1
                    target_boxes_added += int(class_index == target_class_index)
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                records.append(
                    {
                        "source_image_id": image_id,
                        "source_file": basename,
                        "variant": variant_index,
                        "context_factor": context_factor,
                        "crop_xyxy": list(crop),
                        "crop_size": [crop[2] - crop[0], crop[3] - crop[1]],
                        "target_annotation_ids": sorted(expected_target_ids),
                        "included_active_annotations": len(included),
                        "image_file": crop_name,
                        "image_sha256": file_sha256(crop_path),
                        "label_file": label_name,
                        "label_sha256": file_sha256(label_path),
                    }
                )
                generated_paths.append(crop_path.resolve().as_posix())

    if len(records) != expected_added:
        raise AssertionError(f"generated {len(records)} crops, expected {expected_added}")
    expected_target_added = sum(
        1
        for item in annotations["annotations"]
        if int(item["category_id"]) == target_category_id
    ) * len(args.context_factors)
    if target_boxes_added != expected_target_added:
        raise AssertionError(
            f"generated {target_boxes_added} target labels, expected {expected_target_added}"
        )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root.rename(args.output_root)
    final_generated_paths = [
        (args.output_root / "images" / "train" / Path(path).name).resolve().as_posix()
        for path in generated_paths
    ]
    original_wrapper_lines = [
        (args.original_wrapper_image_root / Path(line).name).resolve().as_posix()
        for line in source_lines
    ]
    output_lines = original_wrapper_lines + final_generated_paths
    if len(output_lines) != len(source_lines) + expected_added:
        raise AssertionError("training exposure count mismatch")

    args.output_train_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_train_list.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    detector_dev_lines = load_lines(args.detector_dev_list)
    if len(detector_dev_lines) != len(set(detector_dev_lines)):
        raise ValueError("detector-dev list contains duplicate paths")
    output_yaml = {
        "train": args.output_train_list.resolve().as_posix(),
        "val": args.detector_dev_list.resolve().as_posix(),
        "names": names,
    }
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    recipe_path = args.output_root / "crop_records.jsonl"
    recipe_lines = [json.dumps(record, sort_keys=True, allow_nan=False) for record in records]
    recipe_path.write_text("\n".join(recipe_lines) + "\n", encoding="utf-8")
    target_instances = expected_target_added // len(args.context_factors)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "Crop4-1280 matched-exposure object-centric control",
        "target_category": args.target,
        "target_class_index": target_class_index,
        "context_factors": args.context_factors,
        "minimum_side": args.minimum_side,
        "jpeg_quality": args.jpeg_quality,
        "source_images": len(source_lines),
        "source_unique_images": len(set(source_lines)),
        "target_positive_source_images": len(rare_image_ids),
        "source_target_instances": target_instances,
        "derived_crop_images": len(records),
        "derived_unique_images": len(set(final_generated_paths)),
        "output_training_exposures": len(output_lines),
        "output_unique_files": len(set(output_lines)),
        "derived_target_labels": target_boxes_added,
        "effective_target_instance_exposures": target_instances + target_boxes_added,
        "boundary_expansion_variants": boundary_expansion_count,
        "derived_class_box_counts": {
            names[class_index]: class_box_counts[class_index] for class_index in sorted(names)
        },
        "source_annotations": str(args.annotations.resolve()),
        "source_annotations_sha256": file_sha256(args.annotations),
        "source_train_list": str(args.source_train_list.resolve()),
        "source_train_list_sha256": file_sha256(args.source_train_list),
        "detector_dev_list": str(args.detector_dev_list.resolve()),
        "detector_dev_list_sha256": file_sha256(args.detector_dev_list),
        "forbidden_annotation_hashes": forbidden_hashes,
        "forbidden_overlap_count": 0,
        "output_root": str(args.output_root.resolve()),
        "output_train_list": str(args.output_train_list.resolve()),
        "output_train_list_sha256": file_sha256(args.output_train_list),
        "output_yaml": str(args.output_yaml.resolve()),
        "output_yaml_sha256": file_sha256(args.output_yaml),
        "crop_records": str(recipe_path.resolve()),
        "crop_records_sha256": file_sha256(recipe_path),
        "crop_image_hash_aggregate": aggregate_hash(
            [f"{record['image_file']} {record['image_sha256']}" for record in records]
        ),
        "crop_label_hash_aggregate": aggregate_hash(
            [f"{record['label_file']} {record['label_sha256']}" for record in records]
        ),
        "gates": {
            "exact_three_crops_per_positive_image": True,
            "matched_total_training_exposures_7220": len(output_lines) == 7220,
            "matched_effective_target_instance_exposures_1688": (
                target_instances + target_boxes_added == 1688
            ),
            "all_target_boxes_preserved_in_each_crop": True,
            "no_active_box_cut_by_crop_boundary": True,
            "all_generated_jpegs_verified": True,
            "detector_train_disjoint_from_forbidden_roles": True,
        },
        "source_dataset_modified": False,
        "official_validation_used": False,
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError(f"one or more frozen gates failed: {manifest['gates']}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
