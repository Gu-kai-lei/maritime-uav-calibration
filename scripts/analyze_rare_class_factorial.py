from __future__ import annotations

import argparse
import json
import sys
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

from maritime_calibration.coco import active_categories, load_json, source_group
from maritime_calibration.yolo import file_sha256
from scripts.audit_rare_class_failure import GROUP_FIELDS, matched_target_annotation_ids


def metric_comparison_rows(
    baseline: dict[str, Any], repeat: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    for metric in ("precision", "recall", "map50", "map50_95"):
        baseline_value = float(baseline["aggregate"][metric])
        repeat_value = float(repeat["aggregate"][metric])
        aggregate_rows.append(
            {
                "metric": metric,
                "natural_640": baseline_value,
                "repeat4_640": repeat_value,
                "delta_repeat4_minus_natural": repeat_value - baseline_value,
            }
        )
    class_rows = []
    if set(baseline["per_class"]) != set(repeat["per_class"]):
        raise ValueError("evaluation manifests contain different class sets")
    for category in baseline["per_class"]:
        for metric in ("precision", "recall", "map50", "map50_95"):
            baseline_value = float(baseline["per_class"][category][metric])
            repeat_value = float(repeat["per_class"][category][metric])
            class_rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "natural_640": baseline_value,
                    "repeat4_640": repeat_value,
                    "delta_repeat4_minus_natural": repeat_value - baseline_value,
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(class_rows)


def role_recall_row(
    role: str,
    annotations_path: Path,
    baseline_predictions_path: Path,
    repeat_predictions_path: Path,
    target: str,
    iou: float,
) -> dict[str, Any]:
    dataset = load_json(annotations_path)
    categories = {item["name"]: int(item["id"]) for item in active_categories(dataset)}
    if target not in categories:
        raise ValueError(f"target category not active in {role}: {target}")
    target_id = categories[target]
    target_annotations = [
        item for item in dataset["annotations"] if int(item["category_id"]) == target_id
    ]
    baseline_predictions = load_json(baseline_predictions_path)
    repeat_predictions = load_json(repeat_predictions_path)
    if not isinstance(baseline_predictions, list) or not isinstance(repeat_predictions, list):
        raise ValueError(f"role predictions must be COCO-style lists: {role}")
    baseline_matched = matched_target_annotation_ids(
        target_annotations, baseline_predictions, target_id, iou
    )
    repeat_matched = matched_target_annotation_ids(
        target_annotations, repeat_predictions, target_id, iou
    )
    images = {image["id"]: image for image in dataset["images"]}
    groups = {
        source_group(images[item["image_id"]], GROUP_FIELDS) for item in target_annotations
    }
    baseline_target_predictions = [
        item for item in baseline_predictions if int(item["category_id"]) == target_id
    ]
    repeat_target_predictions = [
        item for item in repeat_predictions if int(item["category_id"]) == target_id
    ]
    ground_truth = len(target_annotations)
    baseline_recall = len(baseline_matched) / ground_truth if ground_truth else None
    repeat_recall = len(repeat_matched) / ground_truth if ground_truth else None
    return {
        "role": role,
        "target_ground_truth": ground_truth,
        "target_images": len({item["image_id"] for item in target_annotations}),
        "source_groups": len(groups),
        "natural_640_matched": len(baseline_matched),
        "repeat4_640_matched": len(repeat_matched),
        "natural_640_recall": baseline_recall,
        "repeat4_640_recall": repeat_recall,
        "delta_repeat4_minus_natural": (
            repeat_recall - baseline_recall
            if baseline_recall is not None and repeat_recall is not None
            else None
        ),
        "natural_target_predictions": len(baseline_target_predictions),
        "repeat4_target_predictions": len(repeat_target_predictions),
        "natural_target_max_score": (
            max(float(item["score"]) for item in baseline_target_predictions)
            if baseline_target_predictions
            else None
        ),
        "repeat4_target_max_score": (
            max(float(item["score"]) for item in repeat_target_predictions)
            if repeat_target_predictions
            else None
        ),
        "annotations_sha256": file_sha256(annotations_path),
        "natural_predictions_sha256": file_sha256(baseline_predictions_path),
        "repeat4_predictions_sha256": file_sha256(repeat_predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the registered rare-class exposure control")
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--repeat-evaluation", required=True, type=Path)
    parser.add_argument(
        "--role",
        action="append",
        nargs=4,
        metavar=("NAME", "ANNOTATIONS", "NATURAL_PREDICTIONS", "REPEAT4_PREDICTIONS"),
        required=True,
    )
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_json(args.baseline_evaluation)
    repeat = load_json(args.repeat_evaluation)
    if baseline["official_validation_used"] or repeat["official_validation_used"]:
        raise ValueError("factorial analysis must not use official validation")
    if baseline["image_size"] != 640 or repeat["image_size"] != 640:
        raise ValueError("both exposure-control evaluations must use imgsz=640")
    aggregate, classes = metric_comparison_rows(baseline, repeat)
    roles = pd.DataFrame(
        [
            role_recall_row(
                role[0], Path(role[1]), Path(role[2]), Path(role[3]), args.target, args.iou
            )
            for role in args.role
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    classes.to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    roles.to_csv(args.output_dir / "target_role_recall.csv", index=False)

    baseline_map = float(baseline["aggregate"]["map50_95"])
    repeat_map = float(repeat["aggregate"]["map50_95"])
    relative_map_change = (repeat_map - baseline_map) / baseline_map
    natural_matches = int(roles["natural_640_matched"].sum())
    repeat_matches = int(roles["repeat4_640_matched"].sum())
    report = {
        "comparison": "natural versus repeat4 rare-positive exposure at 640",
        "official_validation_used": False,
        "target_category": args.target,
        "matching_iou": args.iou,
        "confidence_floor": 0.001,
        "natural_model_sha256": baseline["model_sha256"],
        "repeat4_model_sha256": repeat["model_sha256"],
        "aggregate": aggregate.to_dict(orient="records"),
        "target_roles": roles.to_dict(orient="records"),
        "gate": {
            "natural_target_matches": natural_matches,
            "repeat4_target_matches": repeat_matches,
            "target_localization_improved": repeat_matches > natural_matches,
            "aggregate_map50_95_relative_change": relative_map_change,
            "aggregate_loss_within_10_percent": relative_map_change >= -0.10,
            "repeat4_main_effect_useful": (
                repeat_matches > natural_matches and relative_map_change >= -0.10
            ),
            "natural_1280_remains_independent_next_candidate": True,
            "repeat4_1280_interaction_cell_authorized": False,
        },
        "claim_boundary": (
            "Calibration-fit and policy-tune are one-time post-freeze diagnostics and did not "
            "select the checkpoint or training settings."
        ),
    }
    (args.output_dir / "rare_class_factorial.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    class_map = classes[classes["metric"] == "map50_95"].copy()
    x = np.arange(len(class_map))
    width = 0.36
    axes[0].bar(x - width / 2, class_map["natural_640"], width, label="Natural 640")
    axes[0].bar(x + width / 2, class_map["repeat4_640"], width, label="Repeat4 640")
    axes[0].set_xticks(x, class_map["category"], rotation=25, ha="right")
    axes[0].set_ylabel("AP50-95")
    axes[0].set_title("Detector-dev per-class AP")
    axes[0].legend(frameon=False)
    x = np.arange(len(roles))
    axes[1].bar(x - width / 2, roles["natural_640_recall"], width, label="Natural 640")
    axes[1].bar(x + width / 2, roles["repeat4_640_recall"], width, label="Repeat4 640")
    axes[1].set_xticks(x, roles["role"], rotation=20, ha="right")
    axes[1].set_ylabel("Target proposal recall at IoU >= 0.50")
    axes[1].set_title("Rare-class one-time role diagnostics")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Rare-positive exposure control")
    figure.tight_layout()
    figure.savefig(args.output_dir / "rare_class_factorial.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
