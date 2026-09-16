from __future__ import annotations

import argparse
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, dtype=float).copy().reshape(-1, 4)
    if output.size:
        output[:, 2] += output[:, 0]
        output[:, 3] += output[:, 1]
    return output


def box_iou(labels: np.ndarray, detections: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=float).reshape(-1, 4)
    detections = np.asarray(detections, dtype=float).reshape(-1, 4)
    if len(labels) == 0 or len(detections) == 0:
        return np.zeros((len(labels), len(detections)), dtype=float)
    intersection_lt = np.maximum(labels[:, None, :2], detections[None, :, :2])
    intersection_rb = np.minimum(labels[:, None, 2:], detections[None, :, 2:])
    intersection = np.clip(intersection_rb - intersection_lt, 0.0, None).prod(axis=2)
    label_area = np.clip(labels[:, 2:] - labels[:, :2], 0.0, None).prod(axis=1)
    detection_area = np.clip(detections[:, 2:] - detections[:, :2], 0.0, None).prod(axis=1)
    union = label_area[:, None] + detection_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_predictions(
    detection_classes: np.ndarray,
    label_classes: np.ndarray,
    iou: np.ndarray,
    thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """Match detections exactly like the Ultralytics greedy IoU validator."""
    correct = np.zeros((len(detection_classes), len(thresholds)), dtype=bool)
    class_compatible = label_classes[:, None] == detection_classes[None, :]
    for threshold_index, threshold in enumerate(thresholds):
        label_indices, detection_indices = np.nonzero((iou >= threshold) & class_compatible)
        if not len(label_indices):
            continue
        matches = np.column_stack(
            (label_indices, detection_indices, iou[label_indices, detection_indices])
        )
        if len(matches) > 1:
            matches = matches[np.argsort(-matches[:, 2], kind="stable")]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.argsort(-matches[:, 2], kind="stable")]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def evaluate_predictions(
    dataset: dict[str, Any], predictions: list[dict[str, Any]], target_category_id: int
) -> dict[str, Any]:
    from ultralytics.utils.metrics import ap_per_class

    categories = active_categories(dataset)
    category_to_class = {int(item["id"]): index for index, item in enumerate(categories)}
    class_names = {index: str(item["name"]) for index, item in enumerate(categories)}
    image_ids = {item["id"] for item in dataset["images"]}
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) in category_to_class:
            annotations_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["image_id"] not in image_ids:
            raise ValueError(f"prediction references unknown image_id={prediction['image_id']}")
        if int(prediction["category_id"]) not in category_to_class:
            raise ValueError(
                f"prediction references inactive or unknown category_id={prediction['category_id']}"
            )
        predictions_by_image[prediction["image_id"]].append(prediction)

    all_correct: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_prediction_classes: list[np.ndarray] = []
    all_target_classes: list[np.ndarray] = []
    for image in dataset["images"]:
        annotations = annotations_by_image.get(image["id"], [])
        image_predictions = predictions_by_image.get(image["id"], [])
        label_classes = np.asarray(
            [category_to_class[int(item["category_id"])] for item in annotations], dtype=int
        )
        detection_classes = np.asarray(
            [category_to_class[int(item["category_id"])] for item in image_predictions], dtype=int
        )
        label_boxes = xywh_to_xyxy(
            np.asarray([item["bbox"] for item in annotations], dtype=float).reshape(-1, 4)
        )
        detection_boxes = xywh_to_xyxy(
            np.asarray([item["bbox"] for item in image_predictions], dtype=float).reshape(-1, 4)
        )
        correctness = match_predictions(
            detection_classes, label_classes, box_iou(label_boxes, detection_boxes)
        )
        all_correct.append(correctness)
        all_scores.append(
            np.asarray([float(item["score"]) for item in image_predictions], dtype=float)
        )
        all_prediction_classes.append(detection_classes)
        all_target_classes.append(label_classes)

    correct = np.concatenate(all_correct) if all_correct else np.zeros((0, 10), dtype=bool)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=float)
    prediction_classes = (
        np.concatenate(all_prediction_classes) if all_prediction_classes else np.zeros(0, dtype=int)
    )
    target_classes = (
        np.concatenate(all_target_classes) if all_target_classes else np.zeros(0, dtype=int)
    )
    metric_tuple = ap_per_class(
        correct, scores, prediction_classes, target_classes, plot=False, names=class_names
    )
    true_positives, _, precision, recall, _, ap, unique_classes = metric_tuple[:7]
    class_metrics: list[dict[str, Any]] = []
    for metric_index, class_index in enumerate(unique_classes):
        category = categories[int(class_index)]
        class_metrics.append(
            {
                "class_index": int(class_index),
                "category_id": int(category["id"]),
                "category": str(category["name"]),
                "ground_truth": int((target_classes == class_index).sum()),
                "predictions": int((prediction_classes == class_index).sum()),
                "true_positives_at_max_f1": int(true_positives[metric_index]),
                "precision": float(precision[metric_index]),
                "recall": float(recall[metric_index]),
                "ap50": float(ap[metric_index, 0]),
                "ap50_95": float(ap[metric_index].mean()),
            }
        )
    target_class = category_to_class[target_category_id]
    target_gt = int((target_classes == target_class).sum())
    target_tp = int(correct[prediction_classes == target_class, 0].sum())
    total_tp = int(correct[:, 0].sum())
    return {
        "images": len(dataset["images"]),
        "ground_truth": int(len(target_classes)),
        "predictions": int(len(predictions)),
        "aggregate": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "map50": float(np.mean(ap[:, 0])),
            "map50_95": float(np.mean(ap)),
        },
        "per_class": class_metrics,
        "target": {
            "ground_truth": target_gt,
            "matched_iou_0_50": target_tp,
            "recall_iou_0_50": target_tp / target_gt if target_gt else 0.0,
        },
        "matched_all_iou_0_50": total_tp,
        "false_positives_at_export_floor": int(len(predictions) - total_tp),
        "false_positives_per_image": (len(predictions) - total_tp) / len(dataset["images"]),
    }


def parse_named_paths(values: list[list[str]], label: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name, path_text in values:
        if name in output:
            raise ValueError(f"duplicate {label} name: {name}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = path
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    labels = summary["condition"].tolist()
    x = np.arange(len(labels))
    colors = ["#6B7280" if "full" in label else "#2563EB" if "crop4" in label else "#D97706" for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = [
        ("target_recall_iou_0_50", "Target recall @ IoU 0.50", (0.0, 1.0)),
        ("map50_95", "Aggregate mAP50-95", (0.0, None)),
        ("false_positives_per_image", "False positives / image at conf >= 0.001", (0.0, None)),
        ("latency_p50_ms", "End-to-end latency P50 (ms / image)", (0.0, None)),
    ]
    for axis, (column, title, limits) in zip(axes.flat, panels, strict=True):
        values = summary[column].to_numpy(dtype=float)
        axis.bar(x, np.nan_to_num(values, nan=0.0), color=colors)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if limits[1] is None:
            axis.set_ylim(bottom=limits[0])
        else:
            axis.set_ylim(*limits)
        for index, value in enumerate(values):
            text = "n/a" if np.isnan(value) else f"{value:.3f}"
            axis.text(index, 0.0 if np.isnan(value) else value, text, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Blind tiling inference sensitivity on frozen detector-dev", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen blind tiling prediction exports")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument(
        "--condition", action="append", nargs=2, metavar=("NAME", "PREDICTIONS"), required=True
    )
    parser.add_argument(
        "--uncapped-target", action="append", nargs=2, metavar=("NAME", "PREDICTIONS"), default=[]
    )
    parser.add_argument(
        "--manifest", action="append", nargs=2, metavar=("NAME", "MANIFEST"), default=[]
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    categories = active_categories(dataset)
    target_matches = [int(item["id"]) for item in categories if item["name"] == args.target]
    if len(target_matches) != 1:
        raise ValueError(f"expected one active target category named {args.target}")
    target_category_id = target_matches[0]
    conditions = parse_named_paths(args.condition, "condition")
    uncapped_paths = parse_named_paths(args.uncapped_target, "uncapped target")
    manifest_paths = parse_named_paths(args.manifest, "manifest")
    unknown = (set(uncapped_paths) | set(manifest_paths)) - set(conditions)
    if unknown:
        raise ValueError(f"metadata references unknown conditions: {sorted(unknown)}")

    result_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for condition_name, predictions_path in conditions.items():
        predictions = load_json(predictions_path)
        metrics = evaluate_predictions(dataset, predictions, target_category_id)
        uncapped_metrics = None
        if condition_name in uncapped_paths:
            uncapped_metrics = evaluate_predictions(
                dataset, load_json(uncapped_paths[condition_name]), target_category_id
            )
        manifest = load_json(manifest_paths[condition_name]) if condition_name in manifest_paths else None
        result_records.append(
            {
                "condition": condition_name,
                "predictions": str(predictions_path.resolve()),
                "predictions_sha256": file_sha256(predictions_path),
                "uncapped_target_predictions": (
                    str(uncapped_paths[condition_name].resolve()) if condition_name in uncapped_paths else None
                ),
                "uncapped_target_predictions_sha256": (
                    file_sha256(uncapped_paths[condition_name]) if condition_name in uncapped_paths else None
                ),
                "manifest": manifest,
                "primary_metrics": metrics,
                "uncapped_target_metrics": uncapped_metrics,
            }
        )
        latency = manifest.get("latency_ms_per_image", {}) if manifest else {}
        summary_rows.append(
            {
                "condition": condition_name,
                "images": metrics["images"],
                "predictions": metrics["predictions"],
                "precision": metrics["aggregate"]["precision"],
                "recall": metrics["aggregate"]["recall"],
                "map50": metrics["aggregate"]["map50"],
                "map50_95": metrics["aggregate"]["map50_95"],
                "target_ground_truth": metrics["target"]["ground_truth"],
                "target_matched_iou_0_50": metrics["target"]["matched_iou_0_50"],
                "target_recall_iou_0_50": metrics["target"]["recall_iou_0_50"],
                "uncapped_target_matched_iou_0_50": (
                    uncapped_metrics["target"]["matched_iou_0_50"] if uncapped_metrics else metrics["target"]["matched_iou_0_50"]
                ),
                "uncapped_target_recall_iou_0_50": (
                    uncapped_metrics["target"]["recall_iou_0_50"] if uncapped_metrics else metrics["target"]["recall_iou_0_50"]
                ),
                "false_positives_per_image": metrics["false_positives_per_image"],
                "tiles": manifest.get("tiles") if manifest else None,
                "mean_tiles_per_image": manifest.get("mean_tiles_per_image") if manifest else None,
                "latency_mean_ms": latency.get("mean"),
                "latency_p50_ms": latency.get("p50"),
                "latency_p95_ms": latency.get("p95"),
                "cuda_peak_memory_mib": manifest.get("cuda_peak_memory_mib") if manifest else None,
            }
        )
        for row in metrics["per_class"]:
            per_class_rows.append({"condition": condition_name, **row})

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "blind_tiling_condition_metrics.csv"
    per_class_path = args.output_dir / "blind_tiling_per_class_metrics.csv"
    figure_path = args.output_dir / "blind_tiling_inference_sensitivity.png"
    result_path = args.output_dir / "blind_tiling_inference_sensitivity.json"
    summary.to_csv(summary_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    make_figure(summary, figure_path)

    crop_tiled = summary[
        summary["condition"].str.startswith("crop4") & ~summary["condition"].str.contains("full")
    ]
    target_path_supported = bool((crop_tiled["target_matched_iou_0_50"] > 0).any())
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "target_category": args.target,
        "official_validation_used": False,
        "oracle_locations_used": False,
        "checkpoint_or_threshold_selection_performed": False,
        "metric_implementation": (
            "Ultralytics ap_per_class with same-class greedy IoU matching at 0.50:0.05:0.95"
        ),
        "full_frame_baseline_note": (
            "Full-frame rows are recomputed from the frozen low-threshold prediction JSON files "
            "with the same evaluator used for tiled predictions. They are not the rect-batched "
            "Ultralytics validator manifests."
        ),
        "conditions": result_records,
        "decision": {
            "target_path_supported": target_path_supported,
            "tile_size_selected": None,
            "interpretation": (
                "Crop4 blind tiling restores at least one target proposal on detector-dev."
                if target_path_supported
                else "Crop4 blind tiling remains at zero target proposal recall for both frozen tile sizes."
            ),
        },
        "outputs": {
            "condition_metrics_csv": str(summary_path.resolve()),
            "per_class_metrics_csv": str(per_class_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
    }
    result_path.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
