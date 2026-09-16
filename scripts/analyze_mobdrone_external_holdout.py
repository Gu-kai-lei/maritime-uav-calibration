from __future__ import annotations

import argparse
import json
import re
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
SCRIPT_ROOT = REPO_ROOT / "scripts"
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_blind_tiling_sensitivity import (  # noqa: E402
    box_iou,
    evaluate_predictions,
    match_predictions,
    xywh_to_xyxy,
)
from maritime_calibration.coco import active_categories, load_json  # noqa: E402
from maritime_calibration.yolo import file_sha256  # noqa: E402

FRAME_PATTERN = re.compile(r"^(?P<source>.+)_(?P<frame>\d{6})$")


def frame_source(file_name: str) -> str:
    match = FRAME_PATTERN.match(Path(file_name).stem)
    if not match:
        raise ValueError(f"cannot parse MOBDrone frame source: {file_name}")
    return match.group("source")


def target_only(predictions: list[dict[str, Any]], target_category_id: int) -> list[dict[str, Any]]:
    return [
        row for row in predictions if int(row["category_id"]) == int(target_category_id)
    ]


def target_operating_counts(
    dataset: dict[str, Any], predictions: list[dict[str, Any]], target_category_id: int
) -> dict[str, Any]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_by_id = {int(row["id"]): row for row in dataset["images"]}
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) == target_category_id:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if image_id not in image_by_id:
            raise ValueError(f"prediction references unknown image {image_id}")
        if int(prediction["category_id"]) != target_category_id:
            raise ValueError("target operating counts received a non-target prediction")
        predictions_by_image[image_id].append(prediction)

    source_rows: dict[str, dict[str, Any]] = {}
    totals = {
        "images": len(image_by_id),
        "target_ground_truth": 0,
        "target_predictions": len(predictions),
        "target_matches_iou_0_50": 0,
        "target_false_positives": 0,
        "target_negative_images": 0,
        "target_predictions_on_negative_images": 0,
        "negative_images_with_target_prediction": 0,
    }
    for image_id, image in image_by_id.items():
        source = frame_source(str(image["file_name"]))
        source_row = source_rows.setdefault(
            source,
            {
                "source_group": source,
                "images": 0,
                "target_ground_truth": 0,
                "target_predictions": 0,
                "target_matches_iou_0_50": 0,
                "target_false_positives": 0,
                "target_negative_images": 0,
                "target_predictions_on_negative_images": 0,
                "negative_images_with_target_prediction": 0,
            },
        )
        labels = annotations_by_image.get(image_id, [])
        detections = predictions_by_image.get(image_id, [])
        label_boxes = xywh_to_xyxy(
            np.asarray([row["bbox"] for row in labels], dtype=float).reshape(-1, 4)
        )
        detection_boxes = xywh_to_xyxy(
            np.asarray([row["bbox"] for row in detections], dtype=float).reshape(-1, 4)
        )
        correct = match_predictions(
            np.zeros(len(detections), dtype=int),
            np.zeros(len(labels), dtype=int),
            box_iou(label_boxes, detection_boxes),
            thresholds=np.asarray([0.50]),
        )
        matched = int(correct[:, 0].sum()) if len(correct) else 0
        false_positives = len(detections) - matched
        for row in (totals, source_row):
            row["target_ground_truth"] += len(labels)
            row["target_matches_iou_0_50"] += matched
            row["target_false_positives"] += false_positives
        source_row["images"] += 1
        source_row["target_predictions"] += len(detections)
        if not labels:
            totals["target_negative_images"] += 1
            source_row["target_negative_images"] += 1
            totals["target_predictions_on_negative_images"] += len(detections)
            source_row["target_predictions_on_negative_images"] += len(detections)
            if detections:
                totals["negative_images_with_target_prediction"] += 1
                source_row["negative_images_with_target_prediction"] += 1
    totals["target_recall_iou_0_50"] = (
        totals["target_matches_iou_0_50"] / totals["target_ground_truth"]
        if totals["target_ground_truth"]
        else None
    )
    totals["target_false_positives_per_image"] = totals["target_false_positives"] / totals[
        "images"
    ]
    totals["target_predictions_per_negative_image"] = (
        totals["target_predictions_on_negative_images"] / totals["target_negative_images"]
    )
    totals["negative_image_target_prediction_rate"] = (
        totals["negative_images_with_target_prediction"] / totals["target_negative_images"]
    )
    for row in source_rows.values():
        row["target_recall_iou_0_50"] = (
            row["target_matches_iou_0_50"] / row["target_ground_truth"]
            if row["target_ground_truth"]
            else None
        )
        row["target_false_positives_per_image"] = row["target_false_positives"] / row[
            "images"
        ]
        row["negative_image_target_prediction_rate"] = (
            row["negative_images_with_target_prediction"] / row["target_negative_images"]
            if row["target_negative_images"]
            else None
        )
    return {"overall": totals, "by_source": list(sorted(source_rows.values(), key=lambda r: r["source_group"]))}


def parse_named_paths(values: list[list[str]]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name, path_text in values:
        if name in output:
            raise ValueError(f"duplicate condition {name}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = path
    return output


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    labels = summary["condition"].tolist()
    x = np.arange(len(labels))
    colors = ["#2563EB" if label.startswith("crop4") else "#D97706" for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = [
        ("target_recall_iou_0_50", "External lifebuoy proposal recall @ IoU 0.50"),
        ("target_ap50", "Mapped target AP50"),
        ("target_false_positives_per_image", "Target-class false positives / image"),
        ("negative_image_target_prediction_rate", "Negative images with target prediction"),
    ]
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        values = summary[column].to_numpy(dtype=float)
        axis.bar(x, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("MOBDrone external target-only transfer (no arm selected)", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MOBDrone target-only external holdout")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--condition", action="append", nargs=2, required=True)
    parser.add_argument("--uncapped-target", action="append", nargs=2, default=[])
    parser.add_argument("--manifest", action="append", nargs=2, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    targets = [
        int(row["id"]) for row in active_categories(dataset) if row["name"] == args.target
    ]
    if len(targets) != 1:
        raise ValueError(f"expected one target category named {args.target}")
    target_id = targets[0]
    conditions = parse_named_paths(args.condition)
    uncapped = parse_named_paths(args.uncapped_target) if args.uncapped_target else {}
    manifests = parse_named_paths(args.manifest) if args.manifest else {}
    unknown = (set(uncapped) | set(manifests)) - set(conditions)
    if unknown:
        raise ValueError(f"metadata supplied for unknown conditions: {sorted(unknown)}")

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for name, path in conditions.items():
        predictions = target_only(load_json(path), target_id)
        target_metrics = evaluate_predictions(dataset, predictions, target_id)
        operating = target_operating_counts(dataset, predictions, target_id)
        uncapped_operating = None
        if name in uncapped:
            uncapped_predictions = target_only(load_json(uncapped[name]), target_id)
            uncapped_operating = target_operating_counts(
                dataset, uncapped_predictions, target_id
            )
        per_class = [
            row for row in target_metrics["per_class"] if int(row["category_id"]) == target_id
        ]
        if len(per_class) != 1:
            raise ValueError(f"missing target AP metrics for {name}")
        target_ap = per_class[0]
        overall = operating["overall"]
        uncapped_overall = uncapped_operating["overall"] if uncapped_operating else overall
        manifest = load_json(manifests[name]) if name in manifests else None
        if manifest:
            if float(manifest["confidence_floor"]) != 0.001:
                raise ValueError(f"unexpected confidence floor for {name}")
            nms_value = manifest.get("nms_iou", manifest.get("within_tile_nms_iou"))
            if float(nms_value) != 0.70:
                raise ValueError(f"unexpected NMS IoU for {name}")
        summary_rows.append(
            {
                "condition": name,
                "images": overall["images"],
                "target_ground_truth": overall["target_ground_truth"],
                "target_predictions": overall["target_predictions"],
                "target_matches_iou_0_50": overall["target_matches_iou_0_50"],
                "target_recall_iou_0_50": overall["target_recall_iou_0_50"],
                "uncapped_target_matches_iou_0_50": uncapped_overall[
                    "target_matches_iou_0_50"
                ],
                "uncapped_target_recall_iou_0_50": uncapped_overall[
                    "target_recall_iou_0_50"
                ],
                "target_ap50": target_ap["ap50"],
                "target_ap50_95": target_ap["ap50_95"],
                "target_false_positives_per_image": overall[
                    "target_false_positives_per_image"
                ],
                "target_predictions_per_negative_image": overall[
                    "target_predictions_per_negative_image"
                ],
                "negative_image_target_prediction_rate": overall[
                    "negative_image_target_prediction_rate"
                ],
            }
        )
        source_rows.extend({"condition": name, **row} for row in operating["by_source"])
        records.append(
            {
                "condition": name,
                "predictions": str(path.resolve()),
                "predictions_sha256": file_sha256(path),
                "target_predictions_evaluated": len(predictions),
                "uncapped_target_predictions": str(uncapped[name].resolve())
                if name in uncapped
                else None,
                "uncapped_target_predictions_sha256": file_sha256(uncapped[name])
                if name in uncapped
                else None,
                "manifest": manifest,
                "target_ap_metrics": target_ap,
                "operating_counts": operating,
                "uncapped_operating_counts": uncapped_operating,
            }
        )

    summary = pd.DataFrame(summary_rows)
    sources = pd.DataFrame(source_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "condition_metrics.csv"
    source_path = args.output_dir / "source_metrics.csv"
    figure_path = args.output_dir / "mobdrone_external_holdout.png"
    result_path = args.output_dir / "mobdrone_external_holdout.json"
    summary.to_csv(summary_path, index=False)
    sources.to_csv(source_path, index=False)
    make_figure(summary, figure_path)
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MOBDrone 1.0.0 official recommended DJI_0804 test domain",
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "external_target_mapping": "life_buoy -> life_saving_appliances",
        "images": len(dataset["images"]),
        "target_ground_truth": sum(
            int(row["category_id"]) == target_id for row in dataset["annotations"]
        ),
        "source_groups": len({frame_source(row["file_name"]) for row in dataset["images"]}),
        "conditions": records,
        "claim_boundary": {
            "target_only": True,
            "cross_dataset_aggregate_map_reported": False,
            "checkpoint_threshold_tile_or_merge_rule_selection_performed": False,
            "training_or_calibration_performed": False,
            "official_seadronessee_validation_used": False,
        },
        "outputs": {
            "condition_metrics": str(summary_path.resolve()),
            "source_metrics": str(source_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
