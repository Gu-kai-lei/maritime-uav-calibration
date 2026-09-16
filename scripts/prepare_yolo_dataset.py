from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from maritime_calibration.coco import load_json
from maritime_calibration.yolo import (
    category_maps,
    file_sha256,
    prepare_yolo_split,
    validate_image_files,
    write_dataset_yaml,
    write_detector_yaml,
    write_image_list,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert verified COCO annotations into YOLO labels without copying images"
    )
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument(
        "--partition-dir",
        type=Path,
        help="Optional directory created by partition_training_coco.py",
    )
    args = parser.parse_args()

    train_images = args.dataset_root / "images" / "train"
    val_images = args.dataset_root / "images" / "val"
    labels_root = args.dataset_root / "labels"
    if labels_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing derived labels: {labels_root}; "
            "remove or rename it explicitly after inspection"
        )

    train = load_json(args.train_annotations)
    val = load_json(args.val_annotations)
    train_map, train_names = category_maps(train)
    val_map, val_names = category_maps(val)
    if (train_map, train_names) != (val_map, val_names):
        raise ValueError("train and validation category schemas differ")

    # Complete both preflight checks before writing any derived label file.
    validate_image_files(train, train_images)
    validate_image_files(val, val_images)
    train_summary = prepare_yolo_split(train, labels_root / "train", train_images)
    val_summary = prepare_yolo_split(val, labels_root / "val", val_images)
    write_dataset_yaml(args.output_yaml, args.dataset_root, train_names)

    partition_outputs = {}
    if args.partition_dir is not None:
        list_dir = args.output_yaml.parent / "lists"
        for role in ("detector_train", "detector_dev", "calibration_fit", "policy_tune"):
            partition_path = args.partition_dir / f"instances_{role}.json"
            if not partition_path.is_file():
                raise FileNotFoundError(f"required partition is missing: {partition_path}")
            partition = load_json(partition_path)
            role_map, role_names = category_maps(partition)
            if (role_map, role_names) != (train_map, train_names):
                raise ValueError(f"partition {role} has a different category schema")
            list_path = list_dir / f"{role}.txt"
            write_image_list(partition, train_images, list_path)
            partition_outputs[role] = {
                "annotations": str(partition_path.resolve()),
                "image_list": str(list_path.resolve()),
                "images": len(partition["images"]),
            }
        detector_yaml = args.output_yaml.with_name("seadronessee_detector.yaml")
        write_detector_yaml(
            detector_yaml,
            Path(partition_outputs["detector_train"]["image_list"]),
            Path(partition_outputs["detector_dev"]["image_list"]),
            train_names,
        )
        partition_outputs["detector_yaml"] = str(detector_yaml.resolve())

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "train_annotations": {
            "path": str(args.train_annotations.resolve()),
            "sha256": file_sha256(args.train_annotations),
        },
        "val_annotations": {
            "path": str(args.val_annotations.resolve()),
            "sha256": file_sha256(args.val_annotations),
        },
        "train": train_summary,
        "val": val_summary,
        "dataset_yaml": str(args.output_yaml.resolve()),
        "partitions": partition_outputs,
    }
    manifest_path = args.output_yaml.with_suffix(".manifest.json")
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
