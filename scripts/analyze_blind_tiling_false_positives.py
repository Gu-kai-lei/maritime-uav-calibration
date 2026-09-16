from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256
from scripts.export_blind_tiled_predictions import tile_starts


TAXONOMY_ORDER = [
    "true_positive",
    "duplicate_on_matched_target",
    "target_localization_error",
    "overlaps_other_class",
    "near_other_class",
    "background",
]
TARGET_IOU = 0.50
NEAR_IOU = 0.10
NEIGHBOR_IOU = 0.50
OVERLAP_FRACTION = 0.25
SCORE_BINS = [0.001, 0.01, 0.05, 0.10, 0.25, 1.01]
SCORE_LABELS = ["[.001,.01)", "[.01,.05)", "[.05,.10)", "[.10,.25)", "[.25,1]"]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, dtype=float).copy().reshape(-1, 4)
    if output.size:
        output[:, 2] += output[:, 0]
        output[:, 3] += output[:, 1]
    return output


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float).reshape(-1, 4)
    right = np.asarray(right, dtype=float).reshape(-1, 4)
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=float)
    intersection_lt = np.maximum(left[:, None, :2], right[None, :, :2])
    intersection_rb = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(intersection_rb - intersection_lt, 0.0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0.0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0.0, None).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy_match_indices(iou: np.ndarray, threshold: float = TARGET_IOU) -> np.ndarray:
    """Return one matched GT index per prediction using the frozen Ultralytics-style rule."""
    matrix = np.asarray(iou, dtype=float)
    matched_gt = np.full(matrix.shape[1], -1, dtype=int)
    gt_indices, prediction_indices = np.nonzero(matrix >= threshold)
    if len(gt_indices) == 0:
        return matched_gt
    matches = np.column_stack(
        (gt_indices, prediction_indices, matrix[gt_indices, prediction_indices])
    )
    if len(matches) > 1:
        matches = matches[np.argsort(-matches[:, 2], kind="stable")]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.argsort(-matches[:, 2], kind="stable")]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    matched_gt[matches[:, 1].astype(int)] = matches[:, 0].astype(int)
    return matched_gt


def classify_prediction(
    matched_gt_index: int,
    max_target_iou: float,
    max_other_iou: float,
) -> str:
    if matched_gt_index >= 0:
        return "true_positive"
    if max_target_iou >= TARGET_IOU:
        return "duplicate_on_matched_target"
    if max_target_iou >= NEAR_IOU:
        return "target_localization_error"
    if max_other_iou >= TARGET_IOU:
        return "overlaps_other_class"
    if max_other_iou >= NEAR_IOU:
        return "near_other_class"
    return "background"


def internal_tile_boundaries(length: int, tile_size: int) -> list[int]:
    starts = tile_starts(length, tile_size, OVERLAP_FRACTION)
    boundaries = set(starts)
    boundaries.update(min(start + tile_size, length) for start in starts)
    return sorted(value for value in boundaries if 0 < value < length)


def grid_geometry(
    box: np.ndarray, image_width: int, image_height: int, tile_size: int
) -> tuple[float | None, bool]:
    x1, y1, x2, y2 = [float(value) for value in box]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    x_boundaries = internal_tile_boundaries(image_width, tile_size)
    y_boundaries = internal_tile_boundaries(image_height, tile_size)
    distances = [abs(center_x - value) for value in x_boundaries]
    distances.extend(abs(center_y - value) for value in y_boundaries)
    minimum = min(distances) / tile_size if distances else None
    crosses = any(x1 <= value <= x2 for value in x_boundaries) or any(
        y1 <= value <= y2 for value in y_boundaries
    )
    return minimum, crosses


def percentile_summary(values: pd.Series) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return {"p25": None, "p50": None, "p75": None, "p95": None}
    return {
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def parse_conditions(values: list[list[str]]) -> dict[str, tuple[Path, int]]:
    output: dict[str, tuple[Path, int]] = {}
    for name, path_text, tile_text in values:
        if name in output:
            raise ValueError(f"duplicate condition name: {name}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        tile_size = int(tile_text)
        if tile_size <= 0:
            raise ValueError("tile size must be positive")
        output[name] = (path, tile_size)
    return output


def source_group(image: dict[str, Any]) -> str:
    source = image.get("source") or {}
    return str(source.get("video") or source.get("folder_name") or "unknown")


def audit_condition(
    dataset: dict[str, Any],
    predictions: list[dict[str, Any]],
    target_category_id: int,
    condition: str,
    tile_size: int,
    category_names: dict[int, str],
) -> pd.DataFrame:
    images = {item["id"]: item for item in dataset["images"]}
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) in category_names:
            annotations_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["image_id"] not in images:
            raise ValueError(f"prediction references unknown image_id={prediction['image_id']}")
        if int(prediction["category_id"]) != target_category_id:
            raise ValueError("uncapped target export contains a non-target category")
        predictions_by_image[prediction["image_id"]].append(prediction)

    rows: list[dict[str, Any]] = []
    for image_id, image in images.items():
        image_predictions = predictions_by_image.get(image_id, [])
        if not image_predictions:
            continue
        annotations = annotations_by_image.get(image_id, [])
        target_annotations = [
            item for item in annotations if int(item["category_id"]) == target_category_id
        ]
        other_annotations = [
            item for item in annotations if int(item["category_id"]) != target_category_id
        ]
        prediction_boxes = xywh_to_xyxy(
            np.asarray([item["bbox"] for item in image_predictions], dtype=float)
        )
        target_boxes = xywh_to_xyxy(
            np.asarray([item["bbox"] for item in target_annotations], dtype=float).reshape(-1, 4)
        )
        other_boxes = xywh_to_xyxy(
            np.asarray([item["bbox"] for item in other_annotations], dtype=float).reshape(-1, 4)
        )
        target_iou = box_iou(target_boxes, prediction_boxes)
        other_iou = box_iou(other_boxes, prediction_boxes)
        matched_gt = greedy_match_indices(target_iou)
        max_target_iou = (
            target_iou.max(axis=0) if len(target_boxes) else np.zeros(len(prediction_boxes))
        )
        max_other_iou = (
            other_iou.max(axis=0) if len(other_boxes) else np.zeros(len(prediction_boxes))
        )
        max_other_index = (
            other_iou.argmax(axis=0) if len(other_boxes) else np.full(len(prediction_boxes), -1)
        )
        neighbors = box_iou(prediction_boxes, prediction_boxes)
        if len(neighbors):
            np.fill_diagonal(neighbors, -1.0)
            max_neighbor_iou = neighbors.max(axis=1)
        else:
            max_neighbor_iou = np.zeros(0, dtype=float)
        width = int(image["width"])
        height = int(image["height"])
        image_area = float(width * height)
        for index, (prediction, box) in enumerate(zip(image_predictions, prediction_boxes, strict=True)):
            grid_distance, crosses_grid = grid_geometry(box, width, height, tile_size)
            other_index = int(max_other_index[index])
            other_category_id = (
                int(other_annotations[other_index]["category_id"]) if other_index >= 0 else None
            )
            box_width = max(0.0, float(box[2] - box[0]))
            box_height = max(0.0, float(box[3] - box[1]))
            rows.append(
                {
                    "condition": condition,
                    "tile_size": tile_size,
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "source_group": source_group(image),
                    "image_width": width,
                    "image_height": height,
                    "score": float(prediction["score"]),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_area_ratio": box_width * box_height / image_area,
                    "image_has_target_gt": bool(target_annotations),
                    "matched_target_annotation_id": (
                        int(target_annotations[int(matched_gt[index])]["id"])
                        if matched_gt[index] >= 0
                        else None
                    ),
                    "max_target_iou": float(max_target_iou[index]),
                    "max_other_iou": float(max_other_iou[index]),
                    "nearest_other_category_id": other_category_id,
                    "nearest_other_category": (
                        category_names[other_category_id] if other_category_id is not None else None
                    ),
                    "max_target_prediction_neighbor_iou": float(max_neighbor_iou[index]),
                    "nearest_grid_boundary_distance_over_tile": grid_distance,
                    "crosses_internal_grid_boundary": bool(crosses_grid),
                    "taxonomy": classify_prediction(
                        int(matched_gt[index]),
                        float(max_target_iou[index]),
                        float(max_other_iou[index]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_condition(
    records: pd.DataFrame,
    image_count: int,
    target_ground_truth: int,
    source_image_counts: dict[str, int],
) -> dict[str, Any]:
    false_positives = records[records["taxonomy"] != "true_positive"]
    true_positives = records[records["taxonomy"] == "true_positive"]
    taxonomy_counts = records["taxonomy"].value_counts().to_dict()
    fp_taxonomy_counts = false_positives["taxonomy"].value_counts().to_dict()
    source_counts = false_positives["source_group"].value_counts()
    top_group_count = max(1, math.ceil(len(source_counts) * 0.10)) if len(source_counts) else 0
    confusion = false_positives[
        (false_positives["max_target_iou"] < NEAR_IOU)
        & (false_positives["max_other_iou"] >= NEAR_IOU)
    ]["nearest_other_category"].value_counts()
    confusion_by_taxonomy: dict[str, dict[str, int]] = {}
    for taxonomy in ("overlaps_other_class", "near_other_class"):
        counts = false_positives[false_positives["taxonomy"] == taxonomy][
            "nearest_other_category"
        ].value_counts()
        confusion_by_taxonomy[taxonomy] = {
            str(name): int(count) for name, count in counts.items()
        }
    source_rates = [
        {
            "source_group": str(group),
            "false_positives": int(count),
            "images": int(source_image_counts.get(str(group), 0)),
            "false_positives_per_image": safe_fraction(
                int(count), int(source_image_counts.get(str(group), 0))
            ),
        }
        for group, count in source_counts.items()
    ]
    highest_rate_source = max(
        source_rates, key=lambda item: item["false_positives_per_image"], default=None
    )
    score_bins = pd.cut(
        false_positives["score"], bins=SCORE_BINS, labels=SCORE_LABELS, right=False
    ).value_counts(sort=False)
    return {
        "images": image_count,
        "images_with_target_predictions": int(records["image_id"].nunique()),
        "target_ground_truth": target_ground_truth,
        "target_predictions": int(len(records)),
        "true_positives_iou_0_50": int(len(true_positives)),
        "target_recall_iou_0_50": safe_fraction(len(true_positives), target_ground_truth),
        "target_false_positives": int(len(false_positives)),
        "target_false_positives_per_image": safe_fraction(len(false_positives), image_count),
        "false_positives_on_target_negative_images": int(
            (~false_positives["image_has_target_gt"]).sum()
        ),
        "false_positive_share_on_target_negative_images": safe_fraction(
            int((~false_positives["image_has_target_gt"]).sum()), len(false_positives)
        ),
        "taxonomy_counts": {name: int(taxonomy_counts.get(name, 0)) for name in TAXONOMY_ORDER},
        "false_positive_taxonomy_counts": {
            name: int(fp_taxonomy_counts.get(name, 0)) for name in TAXONOMY_ORDER[1:]
        },
        "false_positive_taxonomy_fractions": {
            name: safe_fraction(int(fp_taxonomy_counts.get(name, 0)), len(false_positives))
            for name in TAXONOMY_ORDER[1:]
        },
        "false_positive_score": percentile_summary(false_positives["score"]),
        "true_positive_score": percentile_summary(true_positives["score"]),
        "false_positive_box_area_ratio": percentile_summary(false_positives["box_area_ratio"]),
        "true_positive_box_area_ratio": percentile_summary(true_positives["box_area_ratio"]),
        "false_positive_score_bins": {
            str(label): int(score_bins.get(label, 0)) for label in SCORE_LABELS
        },
        "residual_neighbor_iou_at_least_0_50": int(
            (false_positives["max_target_prediction_neighbor_iou"] >= NEIGHBOR_IOU).sum()
        ),
        "residual_neighbor_fraction": safe_fraction(
            int((false_positives["max_target_prediction_neighbor_iou"] >= NEIGHBOR_IOU).sum()),
            len(false_positives),
        ),
        "crosses_internal_grid_boundary": int(
            false_positives["crosses_internal_grid_boundary"].sum()
        ),
        "crosses_internal_grid_boundary_fraction": safe_fraction(
            int(false_positives["crosses_internal_grid_boundary"].sum()), len(false_positives)
        ),
        "nearest_grid_boundary_distance_over_tile": percentile_summary(
            false_positives["nearest_grid_boundary_distance_over_tile"]
        ),
        "source_groups_with_false_positives": int(len(source_counts)),
        "top_10_percent_source_groups": top_group_count,
        "top_10_percent_source_group_false_positive_share": safe_fraction(
            int(source_counts.head(top_group_count).sum()) if top_group_count else 0,
            len(false_positives),
        ),
        "other_class_associations_at_iou_0_10": {
            str(name): int(count) for name, count in confusion.items()
        },
        "other_class_associations_by_taxonomy": confusion_by_taxonomy,
        "highest_false_positive_rate_source_group": highest_rate_source,
    }


def make_summary_figure(records: pd.DataFrame, summaries: list[dict[str, Any]], output: Path) -> None:
    conditions = [item["condition"] for item in summaries]
    x = np.arange(len(conditions))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    bottom = np.zeros(len(conditions), dtype=float)
    colors = ["#8B5CF6", "#F59E0B", "#0EA5E9", "#22C55E", "#64748B"]
    for taxonomy, color in zip(TAXONOMY_ORDER[1:], colors, strict=True):
        values = np.asarray(
            [item["summary"]["false_positive_taxonomy_fractions"][taxonomy] for item in summaries]
        )
        axes[0, 0].bar(x, values, bottom=bottom, label=taxonomy, color=color)
        bottom += values
    axes[0, 0].set_title("Target-class false-positive taxonomy")
    axes[0, 0].set_ylabel("Fraction of target-class false positives")
    axes[0, 0].set_xticks(x, conditions, rotation=25, ha="right")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].set_ylim(0, 1)

    fp_per_image = [item["summary"]["target_false_positives_per_image"] for item in summaries]
    axes[0, 1].bar(x, fp_per_image, color="#DC2626")
    axes[0, 1].set_title("Target-class false positives at conf >= 0.001")
    axes[0, 1].set_ylabel("False positives / detector-dev image")
    axes[0, 1].set_xticks(x, conditions, rotation=25, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(fp_per_image):
        axes[0, 1].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    boxplot_data: list[np.ndarray] = []
    boxplot_labels: list[str] = []
    for condition in conditions:
        subset = records[records["condition"] == condition]
        for kind, mask in (
            ("TP", subset["taxonomy"] == "true_positive"),
            ("FP", subset["taxonomy"] != "true_positive"),
        ):
            values = subset.loc[mask, "score"].to_numpy(dtype=float)
            if len(values):
                boxplot_data.append(np.log10(np.clip(values, 1e-12, None)))
                boxplot_labels.append(f"{condition}\n{kind}")
    axes[1, 0].boxplot(boxplot_data, tick_labels=boxplot_labels, showfliers=False)
    axes[1, 0].set_title("Target prediction score distributions")
    axes[1, 0].set_ylabel("log10(score)")
    axes[1, 0].tick_params(axis="x", labelrotation=30)
    axes[1, 0].grid(axis="y", alpha=0.25)

    neighbor = [item["summary"]["residual_neighbor_fraction"] for item in summaries]
    grid = [
        item["summary"]["crosses_internal_grid_boundary_fraction"] for item in summaries
    ]
    width = 0.36
    axes[1, 1].bar(x - width / 2, neighbor, width, label="neighbor IoU >= 0.50")
    axes[1, 1].bar(x + width / 2, grid, width, label="crosses grid boundary")
    axes[1, 1].set_title("Residual overlap and grid association")
    axes[1, 1].set_ylabel("Fraction of target-class false positives")
    axes[1, 1].set_xticks(x, conditions, rotation=25, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Blind tiling target-class false-positive mechanism audit", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def choose_montage_rows(records: pd.DataFrame, per_condition: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for condition in (name for name in records["condition"].unique() if name.startswith("crop4")):
        subset = records[
            (records["condition"] == condition) & (records["taxonomy"] != "true_positive")
        ].sort_values("score", ascending=False, kind="stable")
        subset = subset.drop_duplicates("image_id").head(per_condition)
        selected.append(subset)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def make_montage(
    selected: pd.DataFrame,
    dataset: dict[str, Any],
    image_root: Path,
    target_category_id: int,
    output: Path,
) -> None:
    if selected.empty:
        return
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    conditions = list(selected["condition"].unique())
    columns = max(int((selected["condition"] == name).sum()) for name in conditions)
    fig, axes = plt.subplots(
        len(conditions), columns, figsize=(3.2 * columns, 3.4 * len(conditions)), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")
    for row_index, condition in enumerate(conditions):
        condition_rows = selected[selected["condition"] == condition]
        for column_index, record in enumerate(condition_rows.itertuples(index=False)):
            axis = axes[row_index, column_index]
            source = image_root / record.file_name
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                box_width = record.x2 - record.x1
                box_height = record.y2 - record.y1
                side = max(384.0, 4.0 * max(box_width, box_height))
                center_x = (record.x1 + record.x2) / 2.0
                center_y = (record.y1 + record.y2) / 2.0
                crop_x1 = max(0.0, min(center_x - side / 2.0, image.width - side))
                crop_y1 = max(0.0, min(center_y - side / 2.0, image.height - side))
                crop_x2 = min(float(image.width), crop_x1 + side)
                crop_y2 = min(float(image.height), crop_y1 + side)
                crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            axis.imshow(crop)
            axis.add_patch(
                Rectangle(
                    (record.x1 - crop_x1, record.y1 - crop_y1),
                    box_width,
                    box_height,
                    fill=False,
                    edgecolor="#EF4444",
                    linewidth=2,
                )
            )
            for annotation in annotations_by_image.get(record.image_id, []):
                x, y, width, height = [float(value) for value in annotation["bbox"]]
                if x + width < crop_x1 or y + height < crop_y1 or x > crop_x2 or y > crop_y2:
                    continue
                color = (
                    "#22C55E"
                    if int(annotation["category_id"]) == target_category_id
                    else "#06B6D4"
                )
                axis.add_patch(
                    Rectangle(
                        (x - crop_x1, y - crop_y1),
                        width,
                        height,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.4,
                    )
                )
            axis.set_title(
                f"{condition} | image {record.image_id}\n"
                f"score={record.score:.3f} | {record.taxonomy}",
                fontsize=8,
            )
            axis.axis("off")
    fig.suptitle("Top target-class false positives (red); target GT green; other GT cyan")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen blind-tiling target-class false positives without selection"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument(
        "--condition",
        action="append",
        nargs=3,
        metavar=("NAME", "UNCAPPED_TARGET_JSON", "TILE_SIZE"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--montage-per-condition", type=int, default=5)
    args = parser.parse_args()

    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    if args.montage_per_condition <= 0:
        raise ValueError("montage-per-condition must be positive")
    conditions = parse_conditions(args.condition)
    dataset = load_json(args.annotations)
    categories = active_categories(dataset)
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    target_ids = [identifier for identifier, name in category_names.items() if name == args.target]
    if len(target_ids) != 1:
        raise ValueError(f"expected one active target category named {args.target}")
    target_category_id = target_ids[0]
    target_ground_truth = sum(
        int(item["category_id"]) == target_category_id for item in dataset["annotations"]
    )
    source_image_counts = pd.Series(
        [source_group(image) for image in dataset["images"]], dtype="object"
    ).value_counts().to_dict()

    condition_results: list[dict[str, Any]] = []
    all_records: list[pd.DataFrame] = []
    for condition, (path, tile_size) in conditions.items():
        predictions = load_json(path)
        records = audit_condition(
            dataset,
            predictions,
            target_category_id,
            condition,
            tile_size,
            category_names,
        )
        summary = summarize_condition(
            records, len(dataset["images"]), target_ground_truth, source_image_counts
        )
        condition_results.append(
            {
                "condition": condition,
                "tile_size": tile_size,
                "predictions": str(path.resolve()),
                "predictions_sha256": file_sha256(path),
                "summary": summary,
            }
        )
        all_records.append(records)
    records = pd.concat(all_records, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "blind_tiling_target_prediction_audit.csv"
    summary_path = args.output_dir / "blind_tiling_fp_condition_summary.csv"
    confusion_path = args.output_dir / "blind_tiling_fp_other_class_associations.csv"
    source_path = args.output_dir / "blind_tiling_fp_source_groups.csv"
    figure_path = args.output_dir / "blind_tiling_fp_mechanism_audit.png"
    montage_path = args.output_dir / "blind_tiling_fp_top_badcases.png"
    result_path = args.output_dir / "blind_tiling_fp_mechanism_audit.json"

    records.to_csv(records_path, index=False)
    summary_rows = []
    confusion_rows = []
    source_rows = []
    for item in condition_results:
        condition = item["condition"]
        summary = item["summary"]
        summary_rows.append(
            {
                "condition": condition,
                "tile_size": item["tile_size"],
                "target_ground_truth": summary["target_ground_truth"],
                "target_predictions": summary["target_predictions"],
                "true_positives_iou_0_50": summary["true_positives_iou_0_50"],
                "target_recall_iou_0_50": summary["target_recall_iou_0_50"],
                "target_false_positives": summary["target_false_positives"],
                "target_false_positives_per_image": summary[
                    "target_false_positives_per_image"
                ],
                "background_fraction": summary["false_positive_taxonomy_fractions"][
                    "background"
                ],
                "other_class_fraction": summary["false_positive_taxonomy_fractions"][
                    "overlaps_other_class"
                ]
                + summary["false_positive_taxonomy_fractions"]["near_other_class"],
                "residual_neighbor_fraction": summary["residual_neighbor_fraction"],
                "grid_boundary_crossing_fraction": summary[
                    "crosses_internal_grid_boundary_fraction"
                ],
                "fp_share_on_target_negative_images": summary[
                    "false_positive_share_on_target_negative_images"
                ],
                "top_10_percent_source_group_fp_share": summary[
                    "top_10_percent_source_group_false_positive_share"
                ],
            }
        )
        for taxonomy, counts in summary["other_class_associations_by_taxonomy"].items():
            for category, count in counts.items():
                confusion_rows.append(
                    {
                        "condition": condition,
                        "association": taxonomy,
                        "other_category": category,
                        "count": count,
                    }
                )
        condition_records = records[
            (records["condition"] == condition) & (records["taxonomy"] != "true_positive")
        ]
        for group, count in condition_records["source_group"].value_counts().items():
            group_images = int(source_image_counts.get(str(group), 0))
            source_rows.append(
                {
                    "condition": condition,
                    "source_group": group,
                    "detector_dev_images": group_images,
                    "target_false_positives": int(count),
                    "target_false_positives_per_image": safe_fraction(
                        int(count), group_images
                    ),
                    "share": safe_fraction(int(count), len(condition_records)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    make_summary_figure(records, condition_results, figure_path)
    montage_rows = choose_montage_rows(records, args.montage_per_condition)
    make_montage(
        montage_rows,
        dataset,
        args.image_root,
        target_category_id,
        montage_path,
    )

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "post-freeze descriptive target-class false-positive audit",
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "image_root": str(args.image_root.resolve()),
        "target_category": args.target,
        "target_category_id": target_category_id,
        "target_ground_truth": target_ground_truth,
        "official_validation_used": False,
        "calibration_fit_used": False,
        "policy_tune_used": False,
        "training_performed": False,
        "prediction_filtering_or_rescoring_performed": False,
        "threshold_or_tile_size_selection_performed": False,
        "tile_provenance_available": False,
        "grid_diagnostic_note": (
            "Grid-boundary features are reconstructed geometric associations. The original tile "
            "identity of each merged prediction was not retained, so they are not causal evidence."
        ),
        "frozen_rules": {
            "target_match_iou": TARGET_IOU,
            "descriptive_near_iou": NEAR_IOU,
            "residual_neighbor_iou": NEIGHBOR_IOU,
            "overlap_fraction": OVERLAP_FRACTION,
            "score_bins": SCORE_BINS,
            "taxonomy_precedence": TAXONOMY_ORDER,
        },
        "conditions": condition_results,
        "decision": {
            "proposal_or_merge_rule_selected": None,
            "deployment_tile_size_selected": None,
            "supported_use": "mechanism description and future preregistration only",
            "required_next_evidence": (
                "Evaluate one separately registered proposal/merge rule on new untouched or "
                "external source-sequence data."
            ),
        },
        "montage_selection": montage_rows[
            ["condition", "image_id", "score", "taxonomy"]
        ].to_dict(orient="records"),
        "outputs": {
            "prediction_audit_csv": str(records_path.resolve()),
            "condition_summary_csv": str(summary_path.resolve()),
            "other_class_associations_csv": str(confusion_path.resolve()),
            "source_groups_csv": str(source_path.resolve()),
            "figure": str(figure_path.resolve()),
            "badcase_montage": str(montage_path.resolve()),
        },
    }
    result_path.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
