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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import (
    ALTITUDE_ALIASES,
    GIMBAL_PITCH_ALIASES,
    active_categories,
    load_json,
    metadata_value,
    source_group,
)
from maritime_calibration.matching import xywh_iou


GROUP_FIELDS = ["video_id", "source.video", "source.drone", "source.folder_name"]


def best_prediction_for_annotation(
    annotation: dict[str, Any], predictions: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_iou = 0.0
    for prediction in predictions:
        overlap = xywh_iou(annotation["bbox"], prediction["bbox"])
        if overlap > best_iou:
            best = prediction
            best_iou = overlap
    return best, best_iou


def failure_label(
    target_category_id: int,
    best_prediction: dict[str, Any] | None,
    best_iou: float,
    iou_threshold: float,
) -> str:
    if best_prediction is None or best_iou < iou_threshold:
        return "no_localized_prediction"
    if int(best_prediction["category_id"]) == target_category_id:
        return "same_class_localized"
    return "wrong_class_localized"


def relative_area(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    _, _, width, height = [float(value) for value in annotation["bbox"]]
    return width * height / max(float(image["width"]) * float(image["height"]), 1.0)


def optional_median(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(numeric)) if numeric else None


def parse_role(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("role must use NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def matched_target_annotation_ids(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    target_category_id: int,
    iou_threshold: float,
) -> set[Any]:
    """Greedily match same-class predictions to target GT at a fixed IoU."""
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation["category_id"]) == target_category_id:
            annotations_by_image[annotation["image_id"]].append(annotation)
    predictions_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if int(prediction["category_id"]) == target_category_id:
            predictions_by_image[prediction["image_id"]].append(prediction)

    matched: set[Any] = set()
    for image_id, image_predictions in predictions_by_image.items():
        candidates = annotations_by_image.get(image_id, [])
        used_indices: set[int] = set()
        for prediction in sorted(
            image_predictions, key=lambda record: float(record["score"]), reverse=True
        ):
            best_index = -1
            best_iou = 0.0
            for candidate_index, annotation in enumerate(candidates):
                if candidate_index in used_indices:
                    continue
                overlap = xywh_iou(prediction["bbox"], annotation["bbox"])
                if overlap > best_iou:
                    best_index = candidate_index
                    best_iou = overlap
            if best_index >= 0 and best_iou >= iou_threshold:
                used_indices.add(best_index)
                matched.add(candidates[best_index]["id"])
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a persistent rare-class detector failure")
    parser.add_argument("--role", action="append", required=True, type=parse_role)
    parser.add_argument(
        "--role-prediction",
        action="append",
        default=[],
        type=parse_role,
        help="Optional frozen-role prediction file as NAME=PATH",
    )
    parser.add_argument("--detector-dev", required=True, type=Path)
    parser.add_argument("--predictions-640", required=True, type=Path)
    parser.add_argument("--predictions-1280", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--detail-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    role_datasets = {name: load_json(path) for name, path in args.role}
    role_prediction_paths = dict(args.role_prediction)
    unknown_prediction_roles = sorted(set(role_prediction_paths) - set(role_datasets))
    if unknown_prediction_roles:
        raise ValueError(f"prediction roles lack annotations: {unknown_prediction_roles}")
    support_rows = []
    target_category_id: int | None = None
    target_areas_by_role: dict[str, list[float]] = {}
    train_class_areas: dict[str, list[float]] = defaultdict(list)
    train_class_images: dict[str, set[Any]] = defaultdict(set)
    train_class_widths: dict[str, list[float]] = defaultdict(list)
    train_class_heights: dict[str, list[float]] = defaultdict(list)
    train_class_effective_640: dict[str, list[tuple[float, float]]] = defaultdict(list)
    train_class_effective_1280: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for role, dataset in role_datasets.items():
        categories = {item["name"]: int(item["id"]) for item in active_categories(dataset)}
        if args.target not in categories:
            raise ValueError(f"target category not active in role={role}: {args.target}")
        role_target_id = categories[args.target]
        if target_category_id is None:
            target_category_id = role_target_id
        elif target_category_id != role_target_id:
            raise ValueError("target category id differs between frozen roles")
        images = {image["id"]: image for image in dataset["images"]}
        annotations = [
            item for item in dataset["annotations"] if int(item["category_id"]) == role_target_id
        ]
        areas = [relative_area(item, images[item["image_id"]]) for item in annotations]
        target_areas_by_role[role] = areas
        groups = {
            source_group(images[item["image_id"]], GROUP_FIELDS) for item in annotations
        }
        altitudes = [
            metadata_value(images[item["image_id"]], ALTITUDE_ALIASES) for item in annotations
        ]
        gimbals = [
            metadata_value(images[item["image_id"]], GIMBAL_PITCH_ALIASES) for item in annotations
        ]
        support_rows.append(
            {
                "role": role,
                "instances": len(annotations),
                "images": len({item["image_id"] for item in annotations}),
                "source_groups": len(groups),
                "relative_area_p10": float(np.quantile(areas, 0.10)),
                "relative_area_median": float(np.median(areas)),
                "relative_area_p90": float(np.quantile(areas, 0.90)),
                "altitude_median": optional_median(altitudes),
                "gimbal_pitch_median": optional_median(gimbals),
            }
        )
        if role == "detector_train":
            id_to_name = {int(item["id"]): item["name"] for item in active_categories(dataset)}
            for item in dataset["annotations"]:
                category_id = int(item["category_id"])
                if category_id in id_to_name:
                    class_name = id_to_name[category_id]
                    train_class_areas[class_name].append(
                        relative_area(item, images[item["image_id"]])
                    )
                    train_class_images[class_name].add(item["image_id"])
                    _, _, bbox_width, bbox_height = [float(value) for value in item["bbox"]]
                    train_class_widths[class_name].append(bbox_width)
                    train_class_heights[class_name].append(bbox_height)
                    image = images[item["image_id"]]
                    for input_size, destination in (
                        (640, train_class_effective_640),
                        (1280, train_class_effective_1280),
                    ):
                        scale = input_size / max(float(image["width"]), float(image["height"]))
                        destination[class_name].append(
                            (bbox_width * scale, bbox_height * scale)
                        )
    assert target_category_id is not None

    role_recall_rows = []
    for role, prediction_path in role_prediction_paths.items():
        dataset = role_datasets[role]
        predictions = load_json(prediction_path)
        if not isinstance(predictions, list):
            raise ValueError(f"role prediction must be a COCO-style list: {role}")
        target_annotations = [
            item
            for item in dataset["annotations"]
            if int(item["category_id"]) == target_category_id
        ]
        target_predictions = [
            item for item in predictions if int(item["category_id"]) == target_category_id
        ]
        matched = matched_target_annotation_ids(
            target_annotations, predictions, target_category_id, args.iou
        )
        images = {image["id"]: image for image in dataset["images"]}
        source_groups = {
            source_group(images[item["image_id"]], GROUP_FIELDS)
            for item in target_annotations
        }
        matched_source_groups = {
            source_group(images[item["image_id"]], GROUP_FIELDS)
            for item in target_annotations
            if item["id"] in matched
        }
        role_recall_rows.append(
            {
                "role": role,
                "target_ground_truth": len(target_annotations),
                "target_images": len({item["image_id"] for item in target_annotations}),
                "source_groups": len(source_groups),
                "matched_ground_truth": len(matched),
                "proposal_recall_at_iou_0.50": (
                    len(matched) / len(target_annotations) if target_annotations else None
                ),
                "matched_source_groups": len(matched_source_groups),
                "target_class_predictions": len(target_predictions),
                "target_prediction_images": len(
                    {item["image_id"] for item in target_predictions}
                ),
                "target_prediction_max_score": (
                    max(float(item["score"]) for item in target_predictions)
                    if target_predictions
                    else None
                ),
            }
        )

    dev = load_json(args.detector_dev)
    dev_images = {image["id"]: image for image in dev["images"]}
    dev_categories = {int(item["id"]): item["name"] for item in active_categories(dev)}
    dev_target_annotations = [
        item for item in dev["annotations"] if int(item["category_id"]) == target_category_id
    ]
    prediction_sets = {
        "640": load_json(args.predictions_640),
        "1280": load_json(args.predictions_1280),
    }
    predictions_by_resolution_image: dict[str, dict[Any, list[dict[str, Any]]]] = {}
    for resolution, predictions in prediction_sets.items():
        if not isinstance(predictions, list):
            raise ValueError(f"predictions-{resolution} must be a COCO-style list")
        by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            by_image[prediction["image_id"]].append(prediction)
        predictions_by_resolution_image[resolution] = by_image

    detail_rows = []
    summary_rows = []
    confusion: dict[str, Counter[str]] = {"640": Counter(), "1280": Counter()}
    for annotation in dev_target_annotations:
        image = dev_images[annotation["image_id"]]
        row: dict[str, Any] = {
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "source_group": source_group(image, GROUP_FIELDS),
            "relative_area": relative_area(annotation, image),
        }
        for resolution in ("640", "1280"):
            image_predictions = predictions_by_resolution_image[resolution].get(
                annotation["image_id"], []
            )
            best, best_iou = best_prediction_for_annotation(annotation, image_predictions)
            label = failure_label(target_category_id, best, best_iou, args.iou)
            predicted_name = (
                dev_categories.get(int(best["category_id"]), "unknown") if best is not None else None
            )
            row[f"failure_{resolution}"] = label
            row[f"best_iou_{resolution}"] = best_iou
            row[f"best_category_{resolution}"] = predicted_name
            row[f"best_score_{resolution}"] = float(best["score"]) if best is not None else None
            if label == "wrong_class_localized" and predicted_name is not None:
                confusion[resolution][predicted_name] += 1
        detail_rows.append(row)

    details = pd.DataFrame(detail_rows)
    for resolution in ("640", "1280"):
        counts = details[f"failure_{resolution}"].value_counts()
        target_predictions = [
            item
            for item in prediction_sets[resolution]
            if int(item["category_id"]) == target_category_id
        ]
        summary_rows.append(
            {
                "resolution": int(resolution),
                "target_ground_truth": len(details),
                "same_class_localized": int(counts.get("same_class_localized", 0)),
                "wrong_class_localized": int(counts.get("wrong_class_localized", 0)),
                "no_localized_prediction": int(counts.get("no_localized_prediction", 0)),
                "target_class_predictions": len(target_predictions),
                "target_prediction_images": len({item["image_id"] for item in target_predictions}),
                "target_prediction_max_score": (
                    max(float(item["score"]) for item in target_predictions)
                    if target_predictions
                    else None
                ),
            }
        )

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.detail_output, index=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    support = pd.DataFrame(support_rows)
    failures = pd.DataFrame(summary_rows)
    train_class_support = pd.DataFrame(
        [
            {
                "category": name,
                "instances": len(areas),
                "images": len(train_class_images[name]),
                "relative_area_median": float(np.median(areas)),
                "bbox_width_median_px": float(np.median(train_class_widths[name])),
                "bbox_height_median_px": float(np.median(train_class_heights[name])),
                "effective_width_median_640": float(
                    np.median([item[0] for item in train_class_effective_640[name]])
                ),
                "effective_height_median_640": float(
                    np.median([item[1] for item in train_class_effective_640[name]])
                ),
                "effective_width_median_1280": float(
                    np.median([item[0] for item in train_class_effective_1280[name]])
                ),
                "effective_height_median_1280": float(
                    np.median([item[1] for item in train_class_effective_1280[name]])
                ),
            }
            for name, areas in train_class_areas.items()
        ]
    ).sort_values("instances", ascending=False)
    support.to_csv(args.output_dir / "rare_class_role_support.csv", index=False)
    failures.to_csv(args.output_dir / "rare_class_failure_summary.csv", index=False)
    train_class_support.to_csv(
        args.output_dir / "rare_class_train_class_support.csv", index=False
    )
    role_recall = pd.DataFrame(role_recall_rows)
    if not role_recall.empty:
        role_recall.to_csv(args.output_dir / "rare_class_role_recall.csv", index=False)
    localization_diagnostic = {}
    for resolution in ("640", "1280"):
        ious = details[f"best_iou_{resolution}"].astype(float)
        category_counts = details[f"best_category_{resolution}"].value_counts(dropna=True)
        localization_diagnostic[resolution] = {
            "maximum_best_iou_any_class": float(ious.max()),
            "mean_best_iou_any_class": float(ious.mean()),
            "ground_truth_with_any_prediction_iou_at_least_0.10": int((ious >= 0.10).sum()),
            "ground_truth_with_any_prediction_iou_at_least_0.30": int((ious >= 0.30).sum()),
            "most_common_low_overlap_category": (
                str(category_counts.index[0]) if not category_counts.empty else None
            ),
        }
    report = {
        "target_category": args.target,
        "target_category_id": target_category_id,
        "official_validation_used": False,
        "confidence_floor": 0.001,
        "localization_iou": args.iou,
        "role_support": support.to_dict(orient="records"),
        "detector_train_class_support": train_class_support.to_dict(orient="records"),
        "failure_summary": failures.to_dict(orient="records"),
        "role_proposal_recall": role_recall.to_dict(orient="records"),
        "wrong_class_localization_counts": {
            resolution: dict(counts) for resolution, counts in confusion.items()
        },
        "localization_diagnostic": localization_diagnostic,
        "claim_boundary": (
            "The detector-dev target class comes from one source group; this audit diagnoses the "
            "frozen split and cannot establish cross-source generalization."
        ),
    }
    (args.output_dir / "rare_class_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    axes[0].bar(support["role"], support["instances"], color="#2563EB")
    axes[0].set_title("Target instances by frozen role")
    axes[0].tick_params(axis="x", rotation=20)
    names = list(train_class_areas)
    axes[1].boxplot(
        [train_class_areas[name] for name in names],
        tick_labels=names,
        showfliers=False,
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Detector-train GT relative area")
    axes[1].tick_params(axis="x", rotation=25)
    failure_names = [
        "same_class_localized",
        "wrong_class_localized",
        "no_localized_prediction",
    ]
    bottom = np.zeros(len(failures))
    colors = ["#16A34A", "#F59E0B", "#DC2626"]
    for name, color in zip(failure_names, colors, strict=True):
        values = failures[name].to_numpy()
        axes[2].bar(failures["resolution"].astype(str), values, bottom=bottom, label=name, color=color)
        bottom += values
    axes[2].set_title("Target GT failure mode")
    axes[2].legend(fontsize=8, frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Life-saving-appliance failure audit")
    figure.tight_layout()
    figure.savefig(args.output_dir / "rare_class_audit.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
