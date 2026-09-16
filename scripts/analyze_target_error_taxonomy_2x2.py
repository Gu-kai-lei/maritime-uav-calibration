from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256
from scripts.analyze_rare_class_factorial_2x2 import factorial_effects
from scripts.audit_rare_class_failure import best_prediction_for_annotation, relative_area

CELL_ORDER = ("natural_640", "repeat4_640", "natural_1280", "repeat4_1280")
CELL_LABELS = ("Natural 640", "Repeat4 640", "Natural 1280", "Repeat4 1280")
COLORS = ("#4C78A8", "#F58518", "#72B7B2", "#E45756")


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use CELL=PATH")
    cell, path = value.split("=", 1)
    if cell not in CELL_ORDER:
        raise argparse.ArgumentTypeError(f"unknown cell {cell}; expected one of {CELL_ORDER}")
    return cell, Path(path)


def localized_counts(
    rows: pd.DataFrame, cell: str, threshold: float, target_name: str
) -> dict[str, int]:
    localized = rows[f"best_any_iou_{cell}"] >= threshold
    same_class = localized & (rows[f"best_any_category_{cell}"] == target_name)
    wrong_class = localized & ~same_class
    return {
        "same_class_localized": int(same_class.sum()),
        "wrong_class_localized": int(wrong_class.sum()),
        "any_class_localized": int(localized.sum()),
        "no_localized_prediction": int((~localized).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose the frozen four-cell target failure into localization and class errors"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    thresholds = sorted(set(args.threshold or [0.10, 0.30, 0.50]))
    if any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("IoU thresholds must be in (0, 1]")
    prediction_paths = dict(args.prediction)
    if set(prediction_paths) != set(CELL_ORDER):
        raise ValueError(f"exactly one prediction file is required for each cell: {CELL_ORDER}")

    dataset = load_json(args.annotations)
    categories = {int(item["id"]): str(item["name"]) for item in active_categories(dataset)}
    target_ids = [category_id for category_id, name in categories.items() if name == args.target]
    if len(target_ids) != 1:
        raise ValueError(f"expected one active target category named {args.target}")
    target_id = target_ids[0]
    images = {image["id"]: image for image in dataset["images"]}
    target_annotations = [
        item for item in dataset["annotations"] if int(item["category_id"]) == target_id
    ]
    if not target_annotations:
        raise ValueError("target category has no annotations")

    prediction_sets: dict[str, list[dict[str, Any]]] = {}
    by_cell_image: dict[str, dict[Any, list[dict[str, Any]]]] = {}
    for cell in CELL_ORDER:
        predictions = load_json(prediction_paths[cell])
        if not isinstance(predictions, list):
            raise ValueError(f"{cell} predictions must be a COCO-style list")
        prediction_sets[cell] = predictions
        by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            by_image[prediction["image_id"]].append(prediction)
        by_cell_image[cell] = by_image

    detail_rows: list[dict[str, Any]] = []
    for annotation in target_annotations:
        image = images[annotation["image_id"]]
        row: dict[str, Any] = {
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "relative_area": relative_area(annotation, image),
        }
        for cell in CELL_ORDER:
            image_predictions = by_cell_image[cell].get(annotation["image_id"], [])
            best_any, best_any_iou = best_prediction_for_annotation(
                annotation, image_predictions
            )
            target_predictions = [
                item for item in image_predictions if int(item["category_id"]) == target_id
            ]
            best_same, best_same_iou = best_prediction_for_annotation(
                annotation, target_predictions
            )
            row[f"best_any_iou_{cell}"] = best_any_iou
            row[f"best_any_category_{cell}"] = (
                categories.get(int(best_any["category_id"]), "unknown")
                if best_any is not None
                else None
            )
            row[f"best_any_score_{cell}"] = (
                float(best_any["score"]) if best_any is not None else None
            )
            row[f"best_same_iou_{cell}"] = best_same_iou
            row[f"best_same_score_{cell}"] = (
                float(best_same["score"]) if best_same is not None else None
            )
        detail_rows.append(row)
    details = pd.DataFrame(detail_rows)

    threshold_rows: list[dict[str, Any]] = []
    cell_summaries: dict[str, Any] = {}
    for cell in CELL_ORDER:
        any_ious = details[f"best_any_iou_{cell}"].astype(float)
        same_ious = details[f"best_same_iou_{cell}"].astype(float)
        target_predictions = [
            item for item in prediction_sets[cell] if int(item["category_id"]) == target_id
        ]
        threshold_summary: dict[str, Any] = {}
        for threshold in thresholds:
            counts = localized_counts(details, cell, threshold, args.target)
            threshold_summary[f"{threshold:.2f}"] = counts
            threshold_rows.append({"cell": cell, "iou_threshold": threshold, **counts})
        best_categories = Counter(
            str(value)
            for value in details[f"best_any_category_{cell}"].dropna().tolist()
        )
        cell_summaries[cell] = {
            "prediction_file": str(prediction_paths[cell]),
            "prediction_sha256": file_sha256(prediction_paths[cell]),
            "all_predictions": len(prediction_sets[cell]),
            "target_class_predictions": len(target_predictions),
            "target_prediction_images": len({item["image_id"] for item in target_predictions}),
            "target_prediction_max_score": (
                max(float(item["score"]) for item in target_predictions)
                if target_predictions
                else None
            ),
            "maximum_best_iou_any_class": float(any_ious.max()),
            "mean_best_iou_any_class": float(any_ious.mean()),
            "median_best_iou_any_class": float(any_ious.median()),
            "maximum_best_iou_same_class": float(same_ious.max()),
            "mean_best_iou_same_class": float(same_ious.mean()),
            "best_overlap_category_counts": dict(best_categories),
            "thresholds": threshold_summary,
        }

    threshold_table = pd.DataFrame(threshold_rows)
    effect_rows: list[dict[str, Any]] = []
    ground_truth = len(details)
    for threshold in thresholds:
        indexed = threshold_table[threshold_table["iou_threshold"] == threshold].set_index("cell")
        for outcome in ("any_class_localized", "same_class_localized", "wrong_class_localized"):
            recalls = tuple(float(indexed.loc[cell, outcome]) / ground_truth for cell in CELL_ORDER)
            effect_rows.append(
                {
                    "iou_threshold": threshold,
                    "outcome": outcome,
                    **{cell: value for cell, value in zip(CELL_ORDER, recalls, strict=True)},
                    **factorial_effects(*recalls),
                }
            )
    effects = pd.DataFrame(effect_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "target_error_taxonomy_per_annotation.csv", index=False)
    threshold_table.to_csv(args.output_dir / "target_error_taxonomy_thresholds.csv", index=False)
    effects.to_csv(args.output_dir / "target_error_taxonomy_effects.csv", index=False)
    report = {
        "analysis": "frozen detector-dev four-cell target error taxonomy",
        "official_validation_used": False,
        "annotations": str(args.annotations),
        "annotations_sha256": file_sha256(args.annotations),
        "target_category": args.target,
        "target_ground_truth": ground_truth,
        "confidence_floor": 0.001,
        "iou_thresholds": thresholds,
        "cells": cell_summaries,
        "factorial_effects": effects.to_dict(orient="records"),
        "claim_boundary": (
            "This is a post-freeze detector-dev error analysis. It does not select a checkpoint, "
            "threshold, or training intervention, and it does not use official validation."
        ),
    }
    (args.output_dir / "target_error_taxonomy_2x2.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for cell, label, color in zip(CELL_ORDER, CELL_LABELS, COLORS, strict=True):
        ordered = np.sort(details[f"best_any_iou_{cell}"].to_numpy(dtype=float))
        axes[0].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label, color=color)
    axes[0].set_xlabel("Best IoU with any predicted class")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_title("All target annotations")
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(thresholds))
    width = 0.19
    for index, (cell, label, color) in enumerate(zip(CELL_ORDER, CELL_LABELS, COLORS, strict=True)):
        cell_rows = threshold_table[threshold_table["cell"] == cell]
        axes[1].bar(
            x + (index - 1.5) * width,
            cell_rows["any_class_localized"],
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, [f"IoU >= {value:.1f}" for value in thresholds])
    axes[1].set_ylabel("Localized target GT")
    axes[1].set_title("Any-class localization")

    max_any = [cell_summaries[cell]["maximum_best_iou_any_class"] for cell in CELL_ORDER]
    max_same = [cell_summaries[cell]["maximum_best_iou_same_class"] for cell in CELL_ORDER]
    x = np.arange(len(CELL_ORDER))
    axes[2].bar(x - 0.18, max_any, 0.36, label="Any class", color="#7A5195")
    axes[2].bar(x + 0.18, max_same, 0.36, label="Target class", color="#EF5675")
    axes[2].set_xticks(x, CELL_LABELS, rotation=25, ha="right")
    axes[2].set_ylabel("Maximum best IoU")
    axes[2].set_title("Best observed overlap")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Detector-dev target error taxonomy across the frozen 2x2 cells")
    figure.tight_layout()
    figure.savefig(args.output_dir / "target_error_taxonomy_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
