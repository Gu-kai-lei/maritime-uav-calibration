from __future__ import annotations

import argparse
import json
import math
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

from maritime_calibration.coco import load_json
from maritime_calibration.yolo import file_sha256


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the frozen Crop4 dataset")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_root = resolve_manifest_path(manifest["output_root"])
    records_path = resolve_manifest_path(manifest["crop_records"])
    train_list_path = resolve_manifest_path(manifest["output_train_list"])
    yaml_path = resolve_manifest_path(manifest["output_yaml"])
    records = [json.loads(line) for line in load_lines(records_path)]
    train_lines = load_lines(train_list_path)
    yaml_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    if file_sha256(records_path) != manifest["crop_records_sha256"]:
        errors.append("crop record hash differs from manifest")
    if file_sha256(train_list_path) != manifest["output_train_list_sha256"]:
        errors.append("training list hash differs from manifest")
    if file_sha256(yaml_path) != manifest["output_yaml_sha256"]:
        errors.append("dataset YAML hash differs from manifest")

    source_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    factor_counts: Counter[float] = Counter()
    class_counts: Counter[int] = Counter()
    target_labels = 0
    duplicate_label_excess = 0
    target_duplicate_label_excess = 0
    jpeg_failures = 0
    hash_failures = 0
    invalid_labels = 0
    for record in records:
        source_groups[int(record["source_image_id"])].append(record)
        factor_counts[float(record["context_factor"])] += 1
        image_path = output_root / "images" / "train" / record["image_file"]
        label_path = output_root / "labels" / "train" / record["label_file"]
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"missing generated pair: {record['image_file']}")
            continue
        if (
            file_sha256(image_path) != record["image_sha256"]
            or file_sha256(label_path) != record["label_sha256"]
        ):
            hash_failures += 1
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            jpeg_failures += 1
        has_target = False
        label_lines = load_lines(label_path)
        label_counter = Counter(label_lines)
        duplicate_label_excess += sum(count - 1 for count in label_counter.values())
        target_duplicate_label_excess += sum(
            count - 1
            for line, count in label_counter.items()
            if int(line.split()[0]) == int(manifest["target_class_index"])
        )
        for line in label_lines:
            values = line.split()
            if len(values) != 5:
                invalid_labels += 1
                continue
            class_index = int(values[0])
            coordinates = [float(value) for value in values[1:]]
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
                invalid_labels += 1
            class_counts[class_index] += 1
            if class_index == int(manifest["target_class_index"]):
                target_labels += 1
                has_target = True
        if not has_target:
            errors.append(f"crop lacks target label: {record['image_file']}")

    expected_factors = {float(value) for value in manifest["context_factors"]}
    if len(records) != int(manifest["derived_crop_images"]):
        errors.append("derived crop count differs from manifest")
    if len(source_groups) != int(manifest["target_positive_source_images"]):
        errors.append("target-positive source image count differs from manifest")
    for source_id, group in source_groups.items():
        if len(group) != 3 or {float(record["context_factor"]) for record in group} != expected_factors:
            errors.append(f"source image {source_id} lacks the exact three frozen variants")
    if target_labels != int(manifest["derived_target_labels"]):
        errors.append("derived target-label count differs from manifest")
    if len(train_lines) != int(manifest["output_training_exposures"]):
        errors.append("training exposure count differs from manifest")
    if len(train_lines) != len(set(train_lines)):
        errors.append("training list contains duplicate files")
    missing_training_files = sum(not Path(line).is_file() for line in train_lines)
    if yaml_payload.get("train") != train_list_path.as_posix():
        errors.append("dataset YAML train path does not match manifest")
    if yaml_payload.get("val") != resolve_manifest_path(manifest["detector_dev_list"]).as_posix():
        errors.append("dataset YAML val path is not the isolated detector-dev list")

    source_annotations = resolve_manifest_path(manifest["source_annotations"])
    source_dataset = load_json(source_annotations)
    source_names = {Path(str(image["file_name"])).name for image in source_dataset["images"]}
    forbidden_names: set[str] = set()
    for path_value, expected_hash in manifest["forbidden_annotation_hashes"].items():
        path = resolve_manifest_path(path_value)
        if file_sha256(path) != expected_hash:
            errors.append(f"forbidden partition hash differs: {path}")
        forbidden = load_json(path)
        forbidden_names.update(Path(str(image["file_name"])).name for image in forbidden["images"])
    forbidden_overlap = len(source_names & forbidden_names)
    if forbidden_overlap:
        errors.append(f"detector-train overlaps forbidden roles by {forbidden_overlap} images")

    if jpeg_failures:
        errors.append(f"{jpeg_failures} generated JPEGs failed verification")
    if hash_failures:
        errors.append(f"{hash_failures} generated image/label pairs failed hashes")
    if invalid_labels:
        errors.append(f"{invalid_labels} label rows are invalid")
    if target_duplicate_label_excess:
        errors.append(
            f"{target_duplicate_label_excess} target label rows are exact duplicates"
        )
    if missing_training_files:
        errors.append(f"{missing_training_files} training-list files are missing")
    if not all(manifest["gates"].values()):
        errors.append("generation manifest contains a failed gate")

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "records": len(records),
        "source_groups": len(source_groups),
        "factor_counts": {str(key): factor_counts[key] for key in sorted(factor_counts)},
        "class_box_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "target_labels": target_labels,
        "duplicate_label_excess": duplicate_label_excess,
        "target_duplicate_label_excess": target_duplicate_label_excess,
        "training_exposures": len(train_lines),
        "training_unique_files": len(set(train_lines)),
        "forbidden_overlap": forbidden_overlap,
        "jpeg_failures": jpeg_failures,
        "hash_failures": hash_failures,
        "invalid_labels": invalid_labels,
        "missing_training_files": missing_training_files,
        "official_validation_used": False,
        "passed": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
