from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json


SUMMARY_FEATURES = [
    "bbox_width_px",
    "bbox_height_px",
    "relative_area",
    "target_luminance_mean",
    "target_luminance_std",
    "context_luminance_mean",
    "context_luminance_std",
    "absolute_luminance_contrast",
    "standardized_luminance_contrast",
    "target_saturation_mean",
    "target_gradient_mean",
]


def parse_role(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("role must use NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def integer_box(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = [float(value) for value in bbox]
    left = max(0, min(int(np.floor(x)), image_width - 1))
    top = max(0, min(int(np.floor(y)), image_height - 1))
    right = max(left + 1, min(int(np.ceil(x + width)), image_width))
    bottom = max(top + 1, min(int(np.ceil(y + height)), image_height))
    return left, top, right, bottom


def expanded_box(
    bbox: list[float], image_width: int, image_height: int, factor: float
) -> tuple[int, int, int, int]:
    x, y, width, height = [float(value) for value in bbox]
    center_x = x + width / 2
    center_y = y + height / 2
    expanded_width = max(width * factor, width + 8)
    expanded_height = max(height * factor, height + 8)
    return integer_box(
        [
            center_x - expanded_width / 2,
            center_y - expanded_height / 2,
            expanded_width,
            expanded_height,
        ],
        image_width,
        image_height,
    )


def luminance(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def crop_features(
    image: Image.Image, bbox: list[float], context_factor: float
) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    image_height, image_width = rgb.shape[:2]
    left, top, right, bottom = integer_box(bbox, image_width, image_height)
    context_left, context_top, context_right, context_bottom = expanded_box(
        bbox, image_width, image_height, context_factor
    )
    target = rgb[top:bottom, left:right]
    context = rgb[context_top:context_bottom, context_left:context_right]
    target_luminance = luminance(target)
    context_luminance = luminance(context)

    context_mask = np.ones(context_luminance.shape, dtype=bool)
    inner_left = left - context_left
    inner_top = top - context_top
    inner_right = right - context_left
    inner_bottom = bottom - context_top
    context_mask[inner_top:inner_bottom, inner_left:inner_right] = False
    ring_luminance = context_luminance[context_mask]
    if ring_luminance.size == 0:
        ring_luminance = context_luminance.reshape(-1)

    target_mean = float(target_luminance.mean())
    target_std = float(target_luminance.std())
    context_mean = float(ring_luminance.mean())
    context_std = float(ring_luminance.std())
    absolute_contrast = abs(target_mean - context_mean)

    target_float = target.astype(np.float32) / 255.0
    channel_max = target_float.max(axis=2)
    channel_min = target_float.min(axis=2)
    saturation = np.divide(
        channel_max - channel_min,
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 1e-8,
    )
    horizontal_gradient = (
        np.abs(np.diff(target_luminance, axis=1)).mean()
        if target_luminance.shape[1] > 1
        else 0.0
    )
    vertical_gradient = (
        np.abs(np.diff(target_luminance, axis=0)).mean()
        if target_luminance.shape[0] > 1
        else 0.0
    )
    return {
        "target_luminance_mean": target_mean,
        "target_luminance_std": target_std,
        "context_luminance_mean": context_mean,
        "context_luminance_std": context_std,
        "absolute_luminance_contrast": absolute_contrast,
        "standardized_luminance_contrast": absolute_contrast / max(context_std, 1e-6),
        "target_saturation_mean": float(saturation.mean()),
        "target_gradient_mean": float((horizontal_gradient + vertical_gradient) / 2),
    }


def quantile_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, subset in table.groupby("role", sort=False):
        for feature in SUMMARY_FEATURES:
            values = subset[feature].astype(float)
            rows.append(
                {
                    "role": role,
                    "feature": feature,
                    "instances": len(values),
                    "p10": float(values.quantile(0.10)),
                    "median": float(values.median()),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def select_quantile_rows(table: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = table.sort_values(
        ["standardized_luminance_contrast", "annotation_id"]
    ).reset_index(drop=True)
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[indices].reset_index(drop=True)


def make_montage(
    table: pd.DataFrame,
    image_root: Path,
    output_path: Path,
    roles: list[str],
    samples_per_role: int,
    context_factor: float,
) -> None:
    tile_size = (320, 220)
    margin = 14
    header_height = 34
    rows = []
    for role in roles:
        role_rows = select_quantile_rows(table[table["role"] == role], samples_per_role)
        if role_rows.empty:
            continue
        rows.append((role, role_rows))
    if not rows:
        return
    columns = max(len(records) for _, records in rows)
    canvas = Image.new(
        "RGB",
        (
            margin + columns * (tile_size[0] + margin),
            margin + len(rows) * (tile_size[1] + header_height + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (role, records) in enumerate(rows):
        y_base = margin + row_index * (tile_size[1] + header_height + margin)
        draw.text((margin, y_base), f"{role}: low to high local contrast", fill="black")
        for column_index, record in records.iterrows():
            with Image.open(image_root / str(record["file_name"])) as source:
                source = source.convert("RGB")
                context = expanded_box(
                    list(record["bbox"]), source.width, source.height, context_factor
                )
                crop = source.crop(context)
                gt = integer_box(list(record["bbox"]), source.width, source.height)
                relative_gt = (
                    gt[0] - context[0],
                    gt[1] - context[1],
                    gt[2] - context[0],
                    gt[3] - context[1],
                )
                crop_draw = ImageDraw.Draw(crop)
                line_width = max(1, int(max(crop.width, crop.height) / 160))
                crop_draw.rectangle(relative_gt, outline=(255, 40, 40), width=line_width)
                tile = ImageOps.contain(crop, tile_size, Image.Resampling.LANCZOS)
            tile_canvas = Image.new("RGB", tile_size, (235, 238, 242))
            tile_canvas.paste(
                tile,
                ((tile_size[0] - tile.width) // 2, (tile_size[1] - tile.height) // 2),
            )
            x = margin + column_index * (tile_size[0] + margin)
            canvas.paste(tile_canvas, (x, y_base + header_height))
            draw.text(
                (x + 4, y_base + header_height + 4),
                f"z-contrast={record['standardized_luminance_contrast']:.2f}",
                fill=(15, 23, 42),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe rare-class crop visibility across frozen source-aware roles"
    )
    parser.add_argument("--role", action="append", required=True, type=parse_role)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--context-factor", type=float, default=8.0)
    parser.add_argument("--detail-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--montage-output", required=True, type=Path)
    parser.add_argument("--montage-role", action="append", default=[])
    parser.add_argument("--samples-per-role", type=int, default=6)
    args = parser.parse_args()

    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    rows: list[dict[str, Any]] = []
    expected_category_id: int | None = None
    missing_images: list[str] = []
    for role, annotation_path in args.role:
        dataset = load_json(annotation_path)
        categories = {item["name"]: int(item["id"]) for item in active_categories(dataset)}
        if args.target not in categories:
            raise ValueError(f"target category not active in role={role}: {args.target}")
        category_id = categories[args.target]
        if expected_category_id is None:
            expected_category_id = category_id
        elif category_id != expected_category_id:
            raise ValueError("target category id differs between frozen roles")
        images = {image["id"]: image for image in dataset["images"]}
        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in dataset["annotations"]:
            if int(annotation["category_id"]) == category_id:
                annotations_by_image[annotation["image_id"]].append(annotation)
        for image_id, annotations in annotations_by_image.items():
            image_record = images[image_id]
            image_path = args.image_root / str(image_record["file_name"])
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                for annotation in annotations:
                    _, _, bbox_width, bbox_height = [
                        float(value) for value in annotation["bbox"]
                    ]
                    row: dict[str, Any] = {
                        "role": role,
                        "annotation_id": annotation["id"],
                        "image_id": image_id,
                        "file_name": image_record["file_name"],
                        "bbox": annotation["bbox"],
                        "bbox_width_px": bbox_width,
                        "bbox_height_px": bbox_height,
                        "relative_area": bbox_width
                        * bbox_height
                        / max(float(image_record["width"]) * float(image_record["height"]), 1.0),
                    }
                    row.update(crop_features(image, annotation["bbox"], args.context_factor))
                    rows.append(row)
    if missing_images:
        sample = "\n".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} source images missing; first paths:\n{sample}")
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no target annotations found")
    summary = quantile_summary(table)

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    detail_to_write = table.copy()
    detail_to_write["bbox"] = detail_to_write["bbox"].map(json.dumps)
    detail_to_write.to_csv(args.detail_output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    report = {
        "target_category": args.target,
        "target_category_id": expected_category_id,
        "official_validation_used": False,
        "image_root_publicly_exported": False,
        "context_factor": args.context_factor,
        "roles": {
            role: {
                "instances": int(len(subset)),
                "images": int(subset["image_id"].nunique()),
            }
            for role, subset in table.groupby("role", sort=False)
        },
        "summary": summary.to_dict(orient="records"),
        "claim_boundary": (
            "These are descriptive pixel statistics inside annotated boxes and local context. "
            "They do not establish causal domain shift or annotation correctness."
        ),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    montage_roles = args.montage_role or ["detector_train", "detector_dev"]
    make_montage(
        table,
        args.image_root,
        args.montage_output,
        montage_roles,
        args.samples_per_role,
        args.context_factor,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
