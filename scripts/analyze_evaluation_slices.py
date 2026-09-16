from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maritime_calibration.coco import (
    ALTITUDE_ALIASES,
    GIMBAL_PITCH_ALIASES,
    active_categories,
    load_json,
    metadata_value,
    source_group,
)
from maritime_calibration.metrics import calibration_metrics, evaluate_threshold


GROUP_FIELDS = ["video_id", "source.video", "source.drone", "source.folder_name"]
ALTITUDE_BINS = [-np.inf, 25, 50, 100, 150, np.inf]
ALTITUDE_LABELS = ["<25m", "25-50m", "50-100m", "100-150m", ">=150m"]
GIMBAL_BINS = [-np.inf, 0, 30, 60, np.inf]
GIMBAL_LABELS = ["<=0deg", "0-30deg", "30-60deg", ">60deg"]


def groups_from_annotations(path: Path) -> set[str]:
    dataset = load_json(path)
    return {
        group
        for image in dataset["images"]
        if (group := source_group(image, GROUP_FIELDS)) is not None
    }


def image_frame(
    dataset: dict[str, Any],
    calibration_groups: set[str],
    policy_groups: set[str],
) -> pd.DataFrame:
    rows = []
    for image in dataset["images"]:
        group = source_group(image, GROUP_FIELDS)
        if group in calibration_groups:
            role = "calibration_source_seen"
        elif group in policy_groups:
            role = "policy_source_seen"
        else:
            role = "calibrator_source_unseen"
        rows.append(
            {
                "image_id": image["id"],
                "source_group": group,
                "source_role": role,
                "altitude": metadata_value(image, ALTITUDE_ALIASES),
                "gimbal_pitch": metadata_value(image, GIMBAL_PITCH_ALIASES),
            }
        )
    output = pd.DataFrame(rows)
    output["altitude_slice"] = pd.cut(
        output["altitude"], ALTITUDE_BINS, labels=ALTITUDE_LABELS, right=False
    ).astype("string")
    output["gimbal_slice"] = pd.cut(
        output["gimbal_pitch"], GIMBAL_BINS, labels=GIMBAL_LABELS, right=True
    ).astype("string")
    return output


def calibration_slices(
    table: pd.DataFrame,
    dimension: str,
    probability_columns: list[str],
    bins: int,
) -> pd.DataFrame:
    rows = []
    for value, subset in table.groupby(dimension, dropna=False):
        labels = subset["is_tp"].astype(int).to_numpy()
        for column in probability_columns:
            metrics = calibration_metrics(labels, subset[column].to_numpy(), bins=bins)
            rows.append(
                {
                    "dimension": dimension,
                    "slice": str(value),
                    "variant": column.removeprefix("prob_"),
                    "detections": len(subset),
                    "true_positive_rate": float(labels.mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration and fixed-FP operating points by frozen slices"
    )
    parser.add_argument("--probabilities", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--calibration-annotations", required=True, type=Path)
    parser.add_argument("--policy-annotations", required=True, type=Path)
    parser.add_argument("--operating-points", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    categories = {int(item["id"]): item["name"] for item in active_categories(dataset)}
    calibration_groups = groups_from_annotations(args.calibration_annotations)
    policy_groups = groups_from_annotations(args.policy_annotations)
    images = image_frame(dataset, calibration_groups, policy_groups)
    table = pd.read_csv(args.probabilities).merge(images, on="image_id", how="left", validate="many_to_one")
    table["category_slice"] = table["category_id"].map(categories)
    probability_columns = sorted(column for column in table if column.startswith("prob_"))
    if not probability_columns:
        raise ValueError("no prob_* columns found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_outputs = []
    for dimension in ("source_role", "altitude_slice", "gimbal_slice", "category_slice"):
        calibration_outputs.append(
            calibration_slices(table, dimension, probability_columns, args.bins)
        )
    calibration_output = pd.concat(calibration_outputs, ignore_index=True)
    calibration_output.to_csv(args.output_dir / "calibration_metrics_by_slice.csv", index=False)

    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) in categories:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    operating_points = pd.read_csv(args.operating_points)
    operating_rows = []
    for dimension in ("source_role", "altitude_slice", "gimbal_slice"):
        for value, image_subset in images.groupby(dimension, dropna=False):
            image_ids = set(image_subset["image_id"])
            detection_subset = table[table["image_id"].isin(image_ids)]
            ground_truth_count = sum(len(annotations_by_image.get(image_id, [])) for image_id in image_ids)
            if not ground_truth_count:
                continue
            for point in operating_points.itertuples(index=False):
                column = f"prob_{point.variant}"
                if column not in detection_subset:
                    continue
                result = evaluate_threshold(
                    detection_subset,
                    column,
                    float(point.threshold_selected_on_policy),
                    ground_truth_count,
                    len(image_ids),
                )
                operating_rows.append(
                    {
                        "dimension": dimension,
                        "slice": str(value),
                        "variant": point.variant,
                        "target_fp_per_image": point.target_fp_per_image,
                        "threshold_selected_on_policy": point.threshold_selected_on_policy,
                        "images": len(image_ids),
                        "ground_truth": ground_truth_count,
                        **result,
                    }
                )
    pd.DataFrame(operating_rows).to_csv(
        args.output_dir / "operating_points_by_slice.csv", index=False
    )

    overlap_report = {
        "evaluation_groups": int(images["source_group"].nunique()),
        "evaluation_images": len(images),
        "source_role_images": {
            str(key): int(value) for key, value in images["source_role"].value_counts().items()
        },
        "calibration_groups_in_evaluation": sorted(
            set(images["source_group"]) & calibration_groups
        ),
        "policy_groups_in_evaluation": sorted(set(images["source_group"]) & policy_groups),
        "note": "Source-role analysis is diagnostic; official validation is a same-video frame split.",
    }
    (args.output_dir / "source_overlap.json").write_text(
        json.dumps(overlap_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overlap_report, indent=2))


if __name__ == "__main__":
    main()
