from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256


def load_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rare_positive_files(
    annotations: dict[str, Any], target_name: str
) -> set[str]:
    categories = {item["name"]: int(item["id"]) for item in active_categories(annotations)}
    if target_name not in categories:
        raise ValueError(f"target category is not active: {target_name}")
    target_id = categories[target_name]
    target_image_ids = {
        item["image_id"]
        for item in annotations["annotations"]
        if int(item["category_id"]) == target_id
    }
    images = {image["id"]: image for image in annotations["images"]}
    return {Path(str(images[image_id]["file_name"])).name for image_id in target_image_ids}


def repeated_training_lines(
    source_lines: list[str], rare_basenames: set[str], repeat_total: int
) -> tuple[list[str], list[str]]:
    if repeat_total < 1:
        raise ValueError("repeat-total must be at least 1")
    basename_to_line: dict[str, str] = {}
    for line in source_lines:
        basename = Path(line).name
        if basename in basename_to_line and basename_to_line[basename] != line:
            raise ValueError(f"ambiguous basename in source list: {basename}")
        basename_to_line[basename] = line
    missing = sorted(rare_basenames - set(basename_to_line))
    if missing:
        raise ValueError(f"{len(missing)} target-positive images are absent from source list")
    rare_lines = [line for line in source_lines if Path(line).name in rare_basenames]
    output = list(source_lines)
    for _ in range(repeat_total - 1):
        output.extend(rare_lines)
    return output, rare_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic repeated rare-positive YOLO training list"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--source-yaml", required=True, type=Path)
    parser.add_argument("--source-train-list", required=True, type=Path)
    parser.add_argument("--source-val-list", type=Path)
    parser.add_argument("--output-val-list", type=Path)
    parser.add_argument(
        "--output-image-root",
        type=Path,
        help="Optional read-only image mapping root used by both derived lists",
    )
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--repeat-total", type=int, default=4)
    parser.add_argument("--output-list", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    source_lines = load_nonempty_lines(args.source_train_list)
    if len(source_lines) != len(set(source_lines)):
        raise ValueError("source training list already contains duplicate image paths")
    rare_basenames = rare_positive_files(annotations, args.target)
    output_lines, rare_lines = repeated_training_lines(
        source_lines, rare_basenames, args.repeat_total
    )

    source_val_lines: list[str] | None = None
    output_val_lines: list[str] | None = None
    if (args.source_val_list is None) != (args.output_val_list is None):
        raise ValueError("source-val-list and output-val-list must be provided together")
    if args.source_val_list is not None:
        source_val_lines = load_nonempty_lines(args.source_val_list)
        if len(source_val_lines) != len(set(source_val_lines)):
            raise ValueError("source validation list contains duplicate image paths")
        output_val_lines = list(source_val_lines)

    if args.output_image_root is not None:
        # Preserve the junction path text so Ultralytics derives an isolated label/cache path.
        image_root = args.output_image_root.absolute()
        output_lines = [(image_root / Path(line).name).as_posix() for line in output_lines]
        if output_val_lines is not None:
            output_val_lines = [
                (image_root / Path(line).name).as_posix() for line in output_val_lines
            ]
        missing = [
            line
            for line in set(output_lines + (output_val_lines or []))
            if not Path(line).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files are missing under output-image-root; first={missing[0]}"
            )

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    if args.output_val_list is not None and output_val_lines is not None:
        args.output_val_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_val_list.write_text(
            "\n".join(output_val_lines) + "\n", encoding="utf-8"
        )
    source_yaml = yaml.safe_load(args.source_yaml.read_text(encoding="utf-8"))
    output_yaml = dict(source_yaml)
    output_yaml["train"] = args.output_list.resolve().as_posix()
    if args.output_val_list is not None:
        output_yaml["val"] = args.output_val_list.resolve().as_posix()
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    rare_instances = sum(
        1
        for item in annotations["annotations"]
        if int(item["category_id"])
        == next(
            int(category["id"])
            for category in active_categories(annotations)
            if category["name"] == args.target
        )
    )
    manifest = {
        "target_category": args.target,
        "repeat_total": args.repeat_total,
        "source_images": len(source_lines),
        "source_unique_images": len(set(source_lines)),
        "rare_positive_images": len(rare_lines),
        "rare_instances": rare_instances,
        "output_exposures": len(output_lines),
        "output_unique_images": len(set(output_lines)),
        "added_exposures": len(output_lines) - len(source_lines),
        "effective_rare_instance_exposures": rare_instances * args.repeat_total,
        "source_annotations": str(args.annotations.resolve()),
        "source_annotations_sha256": file_sha256(args.annotations),
        "source_yaml": str(args.source_yaml.resolve()),
        "source_yaml_sha256": file_sha256(args.source_yaml),
        "source_train_list": str(args.source_train_list.resolve()),
        "source_train_list_sha256": file_sha256(args.source_train_list),
        "output_list": str(args.output_list.resolve()),
        "output_list_sha256": file_sha256(args.output_list),
        "output_yaml": str(args.output_yaml.resolve()),
        "output_yaml_sha256": file_sha256(args.output_yaml),
        "source_val_list": (
            str(args.source_val_list.resolve()) if args.source_val_list is not None else None
        ),
        "source_val_list_sha256": (
            file_sha256(args.source_val_list) if args.source_val_list is not None else None
        ),
        "output_val_list": (
            str(args.output_val_list.resolve()) if args.output_val_list is not None else None
        ),
        "output_val_list_sha256": (
            file_sha256(args.output_val_list) if args.output_val_list is not None else None
        ),
        "output_image_root": (
            str(args.output_image_root.absolute()) if args.output_image_root is not None else None
        ),
        "source_dataset_modified": False,
        "official_validation_used": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
