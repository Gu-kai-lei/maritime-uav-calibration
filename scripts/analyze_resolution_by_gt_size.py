from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from maritime_calibration.coco import active_categories, load_json, source_group
from maritime_calibration.matching import xywh_iou


GROUP_FIELDS = ["video_id", "source.video", "source.drone", "source.folder_name"]
SIZE_LABELS = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]


def matched_annotation_ids(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> set[Any]:
    annotations_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_key[(annotation["image_id"], annotation["category_id"])].append(annotation)
    predictions_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_key[(prediction["image_id"], prediction["category_id"])].append(prediction)

    matched: set[Any] = set()
    for key, key_predictions in predictions_by_key.items():
        candidates = annotations_by_key.get(key, [])
        used_indices: set[int] = set()
        for prediction in sorted(
            key_predictions, key=lambda record: float(record["score"]), reverse=True
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


def grouped_bootstrap_delta(
    table: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    groups = {
        group: subset[["matched_640", "matched_1280"]].to_numpy(dtype=float)
        for group, subset in table.groupby("source_group")
    }
    if len(groups) < 2:
        return float("nan"), float("nan")
    names = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(names, size=len(names), replace=True)
        values = np.concatenate([groups[name] for name in sampled])
        differences.append(float(values[:, 1].mean() - values[:, 0].mean()))
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def summarize_dimension(
    table: pd.DataFrame,
    dimension: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for value, subset in table.groupby(dimension, observed=False):
        lower, upper = grouped_bootstrap_delta(subset, repetitions, seed)
        recall_640 = float(subset["matched_640"].mean())
        recall_1280 = float(subset["matched_1280"].mean())
        rows.append(
            {
                dimension: str(value),
                "ground_truth": len(subset),
                "source_groups": int(subset["source_group"].nunique()),
                "recall_640": recall_640,
                "recall_1280": recall_1280,
                "delta_1280_minus_640": recall_1280 - recall_640,
                "bootstrap_ci_2.5": lower,
                "bootstrap_ci_97.5": upper,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired 640/1280 proposal recall by GT size")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions-640", required=True, type=Path)
    parser.add_argument("--predictions-1280", required=True, type=Path)
    parser.add_argument("--evaluation-640", required=True, type=Path)
    parser.add_argument("--evaluation-1280", required=True, type=Path)
    parser.add_argument("--paired-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    predictions_640 = load_json(args.predictions_640)
    predictions_1280 = load_json(args.predictions_1280)
    if not isinstance(predictions_640, list) or not isinstance(predictions_1280, list):
        raise ValueError("both prediction inputs must be COCO-style lists")
    evaluation_640 = load_json(args.evaluation_640)
    evaluation_1280 = load_json(args.evaluation_1280)
    if evaluation_640["model_sha256"] != evaluation_1280["model_sha256"]:
        raise ValueError("paired evaluations use different model checkpoints")
    if evaluation_640["data_yaml_sha256"] != evaluation_1280["data_yaml_sha256"]:
        raise ValueError("paired evaluations use different detector data definitions")
    if evaluation_640["official_validation_used"] or evaluation_1280["official_validation_used"]:
        raise ValueError("resolution gate must not use official validation")

    matched_640 = matched_annotation_ids(dataset, predictions_640, args.iou)
    matched_1280 = matched_annotation_ids(dataset, predictions_1280, args.iou)
    images = {image["id"]: image for image in dataset["images"]}
    categories = {item["id"]: item["name"] for item in active_categories(dataset)}
    rows = []
    for annotation in dataset["annotations"]:
        image = images[annotation["image_id"]]
        _, _, width, height = [float(value) for value in annotation["bbox"]]
        image_area = max(float(image["width"]) * float(image["height"]), 1.0)
        rows.append(
            {
                "annotation_id": annotation["id"],
                "image_id": annotation["image_id"],
                "category": categories[int(annotation["category_id"])],
                "relative_area": width * height / image_area,
                "source_group": source_group(image, GROUP_FIELDS),
                "matched_640": int(annotation["id"] in matched_640),
                "matched_1280": int(annotation["id"] in matched_1280),
            }
        )
    table = pd.DataFrame(rows)
    if table["source_group"].isna().any():
        raise ValueError("every detector-dev ground truth needs a source group")
    quartiles = np.quantile(table["relative_area"], [0.25, 0.50, 0.75])
    table["size_quartile"] = pd.Categorical(
        [SIZE_LABELS[np.searchsorted(quartiles, value, side="right")] for value in table["relative_area"]],
        categories=SIZE_LABELS,
        ordered=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.paired_output, index=False)
    size = summarize_dimension(table, "size_quartile", args.repetitions, args.seed)
    classes = summarize_dimension(table, "category", args.repetitions, args.seed)
    size.to_csv(args.output_dir / "gt_size_recall.csv", index=False)
    classes.to_csv(args.output_dir / "gt_class_recall.csv", index=False)
    detector_class_rows = []
    for category in evaluation_640["per_class"]:
        if category not in evaluation_1280["per_class"]:
            raise ValueError(f"class missing from 1280 evaluation: {category}")
        for metric in ("precision", "recall", "map50", "map50_95"):
            value_640 = float(evaluation_640["per_class"][category][metric])
            value_1280 = float(evaluation_1280["per_class"][category][metric])
            detector_class_rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "value_640": value_640,
                    "value_1280": value_1280,
                    "delta_1280_minus_640": value_1280 - value_640,
                }
            )
    detector_class_metrics = pd.DataFrame(detector_class_rows)
    detector_class_metrics.to_csv(args.output_dir / "detector_class_metrics.csv", index=False)

    overall_lower, overall_upper = grouped_bootstrap_delta(table, args.repetitions, args.seed)
    speed_640 = sum(
        evaluation_640["speed_ms_per_image"][key]
        for key in ("preprocess", "inference", "postprocess")
    )
    speed_1280 = sum(
        evaluation_1280["speed_ms_per_image"][key]
        for key in ("preprocess", "inference", "postprocess")
    )
    report = {
        "comparison": "paired detector-dev inference with one frozen checkpoint",
        "model_sha256": evaluation_640["model_sha256"],
        "data_yaml_sha256": evaluation_640["data_yaml_sha256"],
        "official_validation_used": False,
        "matching_iou": args.iou,
        "confidence_floor": 0.001,
        "ground_truth": len(table),
        "source_groups": int(table["source_group"].nunique()),
        "relative_area_quartile_boundaries": [float(value) for value in quartiles],
        "aggregate_metrics": {
            "640": evaluation_640["aggregate"],
            "1280": evaluation_1280["aggregate"],
            "map50_95_delta": (
                evaluation_1280["aggregate"]["map50_95"]
                - evaluation_640["aggregate"]["map50_95"]
            ),
        },
        "per_class_metrics": {
            category: {
                "640": evaluation_640["per_class"][category],
                "1280": evaluation_1280["per_class"][category],
            }
            for category in evaluation_640["per_class"]
        },
        "proposal_recall": {
            "640": float(table["matched_640"].mean()),
            "1280": float(table["matched_1280"].mean()),
            "delta": float(table["matched_1280"].mean() - table["matched_640"].mean()),
            "grouped_bootstrap_ci_2.5": overall_lower,
            "grouped_bootstrap_ci_97.5": overall_upper,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "latency_ms_per_image": {
            "640": speed_640,
            "1280": speed_1280,
            "ratio_1280_over_640": speed_1280 / speed_640,
        },
        "claim_boundary": (
            "This evaluates a 640-trained checkpoint at two inference sizes on detector-dev; "
            "it is not evidence about a detector trained at 1280."
        ),
    }
    (args.output_dir / "resolution_size_analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(size))
    width = 0.36
    axes[0].bar(x - width / 2, size["recall_640"], width, label="640")
    axes[0].bar(x + width / 2, size["recall_1280"], width, label="1280")
    axes[0].set_xticks(x, size["size_quartile"], rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Proposal recall by GT relative-area quartile")
    axes[0].legend(frameon=False)
    class_order = list(categories.values())
    class_plot = classes.set_index("category").loc[class_order]
    x = np.arange(len(class_plot))
    axes[1].bar(x - width / 2, class_plot["recall_640"], width)
    axes[1].bar(x + width / 2, class_plot["recall_1280"], width)
    axes[1].set_xticks(x, class_order, rotation=22, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Proposal recall by class")
    for axis in axes:
        axis.set_ylabel("GT matched at IoU >= 0.50")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Paired resolution sensitivity on detector-dev")
    figure.tight_layout()
    figure.savefig(args.output_dir / "resolution_size_recall.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
