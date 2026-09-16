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
from scripts.analyze_rare_class_factorial import role_recall_row
from scripts.audit_rare_class_failure import best_prediction_for_annotation, relative_area

METRICS = ("precision", "recall", "map50", "map50_95")
CELLS = ("repeat4_1280", "crop4_1280")


def metric_rows(
    repeat4: dict[str, Any], crop4: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(repeat4["per_class"]) != set(crop4["per_class"]):
        raise ValueError("evaluation manifests contain different class sets")

    def row(metric: str, repeat_value: float, crop_value: float) -> dict[str, Any]:
        return {
            "metric": metric,
            "repeat4_1280": repeat_value,
            "crop4_1280": crop_value,
            "delta_crop4_minus_repeat4": crop_value - repeat_value,
        }

    aggregate = pd.DataFrame(
        [
            row(
                metric,
                float(repeat4["aggregate"][metric]),
                float(crop4["aggregate"][metric]),
            )
            for metric in METRICS
        ]
    )
    classes = pd.DataFrame(
        [
            {
                "category": category,
                **row(
                    metric,
                    float(repeat4["per_class"][category][metric]),
                    float(crop4["per_class"][category][metric]),
                ),
            }
            for category in repeat4["per_class"]
            for metric in METRICS
        ]
    )
    return aggregate, classes


def role_row(
    role: str,
    annotations: Path,
    repeat_predictions: Path,
    crop_predictions: Path,
    target: str,
    iou: float,
) -> dict[str, Any]:
    raw = role_recall_row(
        role, annotations, repeat_predictions, crop_predictions, target, iou
    )
    return {
        "role": role,
        "target_ground_truth": raw["target_ground_truth"],
        "target_images": raw["target_images"],
        "source_groups": raw["source_groups"],
        "repeat4_1280_matched": raw["natural_640_matched"],
        "crop4_1280_matched": raw["repeat4_640_matched"],
        "repeat4_1280_recall": raw["natural_640_recall"],
        "crop4_1280_recall": raw["repeat4_640_recall"],
        "delta_crop4_minus_repeat4": raw["delta_repeat4_minus_natural"],
        "repeat4_target_predictions": raw["natural_target_predictions"],
        "crop4_target_predictions": raw["repeat4_target_predictions"],
        "repeat4_target_max_score": raw["natural_target_max_score"],
        "crop4_target_max_score": raw["repeat4_target_max_score"],
        "annotations_sha256": raw["annotations_sha256"],
        "repeat4_predictions_sha256": raw["natural_predictions_sha256"],
        "crop4_predictions_sha256": raw["repeat4_predictions_sha256"],
    }


def target_error_taxonomy(
    annotations_path: Path,
    prediction_paths: dict[str, Path],
    target: str,
    thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset = load_json(annotations_path)
    categories = {int(item["id"]): str(item["name"]) for item in active_categories(dataset)}
    target_ids = [category_id for category_id, name in categories.items() if name == target]
    if len(target_ids) != 1:
        raise ValueError(f"expected one active target category named {target}")
    target_id = target_ids[0]
    images = {item["id"]: item for item in dataset["images"]}
    annotations = [
        item for item in dataset["annotations"] if int(item["category_id"]) == target_id
    ]
    if not annotations:
        raise ValueError("target category has no annotations")

    prediction_sets: dict[str, list[dict[str, Any]]] = {}
    by_image: dict[str, dict[Any, list[dict[str, Any]]]] = {}
    for cell in CELLS:
        predictions = load_json(prediction_paths[cell])
        if not isinstance(predictions, list):
            raise ValueError(f"{cell} predictions must be a COCO-style list")
        prediction_sets[cell] = predictions
        indexed: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            indexed[prediction["image_id"]].append(prediction)
        by_image[cell] = indexed

    details: list[dict[str, Any]] = []
    for annotation in annotations:
        image = images[annotation["image_id"]]
        row: dict[str, Any] = {
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "relative_area": relative_area(annotation, image),
        }
        for cell in CELLS:
            candidates = by_image[cell].get(annotation["image_id"], [])
            best_any, best_any_iou = best_prediction_for_annotation(annotation, candidates)
            same_class = [item for item in candidates if int(item["category_id"]) == target_id]
            best_same, best_same_iou = best_prediction_for_annotation(annotation, same_class)
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
        details.append(row)
    detail_table = pd.DataFrame(details)

    threshold_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for cell in CELLS:
        any_ious = detail_table[f"best_any_iou_{cell}"].astype(float)
        same_ious = detail_table[f"best_same_iou_{cell}"].astype(float)
        target_predictions = [
            item for item in prediction_sets[cell] if int(item["category_id"]) == target_id
        ]
        threshold_summary: dict[str, Any] = {}
        for threshold in thresholds:
            localized = any_ious >= threshold
            same = localized & (detail_table[f"best_any_category_{cell}"] == target)
            counts = {
                "same_class_localized": int(same.sum()),
                "wrong_class_localized": int((localized & ~same).sum()),
                "any_class_localized": int(localized.sum()),
                "no_localized_prediction": int((~localized).sum()),
            }
            threshold_summary[f"{threshold:.2f}"] = counts
            threshold_rows.append({"cell": cell, "iou_threshold": threshold, **counts})
        summaries[cell] = {
            "prediction_file": str(prediction_paths[cell].resolve()),
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
            "maximum_best_iou_same_class": float(same_ious.max()),
            "mean_best_iou_same_class": float(same_ious.mean()),
            "best_overlap_category_counts": dict(
                Counter(str(value) for value in detail_table[f"best_any_category_{cell}"].dropna())
            ),
            "thresholds": threshold_summary,
        }
    return detail_table, pd.DataFrame(threshold_rows), summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen Crop4-1280 control")
    parser.add_argument("--repeat-evaluation", required=True, type=Path)
    parser.add_argument("--crop-evaluation", required=True, type=Path)
    parser.add_argument(
        "--role",
        action="append",
        nargs=4,
        metavar=("NAME", "ANNOTATIONS", "REPEAT4", "CROP4"),
        required=True,
    )
    parser.add_argument("--error-annotations", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repeat4 = load_json(args.repeat_evaluation)
    crop4 = load_json(args.crop_evaluation)
    evaluations = (repeat4, crop4)
    if any(item["official_validation_used"] for item in evaluations):
        raise ValueError("object-centric analysis must not use official validation")
    if any(item["image_size"] != 1280 for item in evaluations):
        raise ValueError("both evaluations must use imgsz=1280")
    if any(item["evaluation_split"] != "detector-dev" for item in evaluations):
        raise ValueError("both evaluations must use detector-dev")
    if any(float(item["confidence_floor"]) != 0.001 for item in evaluations):
        raise ValueError("both evaluations must use confidence floor 0.001")
    if any(float(item["nms_iou"]) != 0.70 for item in evaluations):
        raise ValueError("both evaluations must use NMS IoU 0.70")

    aggregate, classes = metric_rows(repeat4, crop4)
    roles = pd.DataFrame(
        [
            role_row(
                role[0], Path(role[1]), Path(role[2]), Path(role[3]), args.target, args.iou
            )
            for role in args.role
        ]
    )
    thresholds = tuple(sorted(set(args.threshold or [0.10, 0.30, 0.50])))
    if any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("IoU thresholds must be in (0, 1]")
    detector_role = next((role for role in args.role if role[0] == "detector-dev"), None)
    if detector_role is None:
        raise ValueError("a detector-dev role is required for target error taxonomy")
    details, taxonomy, taxonomy_summary = target_error_taxonomy(
        args.error_annotations,
        {"repeat4_1280": Path(detector_role[2]), "crop4_1280": Path(detector_role[3])},
        args.target,
        thresholds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    classes.to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    roles.to_csv(args.output_dir / "target_role_recall.csv", index=False)
    details.to_csv(args.output_dir / "target_error_taxonomy_per_annotation.csv", index=False)
    taxonomy.to_csv(args.output_dir / "target_error_taxonomy_thresholds.csv", index=False)

    target_ap_row = classes[
        (classes["category"] == args.target) & (classes["metric"] == "map50_95")
    ]
    if len(target_ap_row) != 1:
        raise ValueError(f"expected one AP50-95 row for {args.target}")
    aggregate_map_row = aggregate[aggregate["metric"] == "map50_95"].iloc[0]
    total_gt = int(roles["target_ground_truth"].sum())
    repeat_matches = int(roles["repeat4_1280_matched"].sum())
    crop_matches = int(roles["crop4_1280_matched"].sum())
    report = {
        "comparison": "Repeat4-1280 whole-frame repetition versus Crop4-1280 object-centric presentation",
        "official_validation_used": False,
        "checkpoint_selection_split": "detector-dev",
        "target_category": args.target,
        "matching_iou": args.iou,
        "confidence_floor": 0.001,
        "nms_iou": 0.70,
        "model_sha256": {
            "repeat4_1280": repeat4["model_sha256"],
            "crop4_1280": crop4["model_sha256"],
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "per_class": classes.to_dict(orient="records"),
        "target_roles": roles.to_dict(orient="records"),
        "target_error_taxonomy": {
            "annotations": str(args.error_annotations.resolve()),
            "annotations_sha256": file_sha256(args.error_annotations),
            "target_ground_truth": len(details),
            "iou_thresholds": list(thresholds),
            "cells": taxonomy_summary,
        },
        "decision": {
            "repeat4_target_matches_across_roles": repeat_matches,
            "crop4_target_matches_across_roles": crop_matches,
            "target_ground_truth_across_roles": total_gt,
            "repeat4_target_proposal_recall_across_roles": repeat_matches / total_gt,
            "crop4_target_proposal_recall_across_roles": crop_matches / total_gt,
            "target_ap50_95_delta": float(
                target_ap_row.iloc[0]["delta_crop4_minus_repeat4"]
            ),
            "aggregate_map50_95_delta": float(
                aggregate_map_row["delta_crop4_minus_repeat4"]
            ),
            "object_centric_presentation_better_on_preregistered_target_endpoints": (
                crop_matches > repeat_matches
                or float(target_ap_row.iloc[0]["delta_crop4_minus_repeat4"]) > 0
            ),
        },
        "claim_boundary": (
            "Calibration-fit and policy-tune are one-time post-freeze diagnostics. They did not "
            "select a checkpoint, threshold, crop recipe, or training setting; official validation "
            "was not used."
        ),
    }
    (args.output_dir / "object_centric_crop_control.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    class_map = classes[classes["metric"] == "map50_95"]
    x = np.arange(len(class_map))
    width = 0.36
    axes[0].bar(x - width / 2, class_map["repeat4_1280"], width, label="Repeat4 1280")
    axes[0].bar(x + width / 2, class_map["crop4_1280"], width, label="Crop4 1280")
    axes[0].set_xticks(x, class_map["category"], rotation=25, ha="right")
    axes[0].set_ylabel("AP50-95")
    axes[0].set_title("Detector-dev per-class AP")
    axes[0].legend(frameon=False)

    x = np.arange(len(roles))
    axes[1].bar(x - width / 2, roles["repeat4_1280_recall"], width, label="Repeat4 1280")
    axes[1].bar(x + width / 2, roles["crop4_1280_recall"], width, label="Crop4 1280")
    axes[1].set_xticks(x, roles["role"], rotation=20, ha="right")
    axes[1].set_ylabel("Target proposal recall at IoU >= 0.50")
    axes[1].set_title("One-time cross-role diagnostics")

    at_half = taxonomy[taxonomy["iou_threshold"] == 0.50].set_index("cell")
    outcomes = ("same_class_localized", "wrong_class_localized", "no_localized_prediction")
    bottom = np.zeros(len(CELLS))
    colors = ("#2A9D8F", "#E9C46A", "#E76F51")
    for outcome, color in zip(outcomes, colors, strict=True):
        values = np.asarray([at_half.loc[cell, outcome] for cell in CELLS], dtype=float)
        axes[2].bar(CELLS, values, bottom=bottom, label=outcome.replace("_", " "), color=color)
        bottom += values
    axes[2].set_ylabel("Target ground-truth annotations")
    axes[2].set_title("Detector-dev error taxonomy at IoU 0.50")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Object-centric crop control: presentation versus whole-frame repetition")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "object_centric_crop_control.png", dpi=200, bbox_inches="tight"
    )
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
