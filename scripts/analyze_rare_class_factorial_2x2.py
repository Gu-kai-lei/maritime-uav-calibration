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

from maritime_calibration.coco import load_json
from scripts.analyze_rare_class_factorial import role_recall_row

METRICS = ("precision", "recall", "map50", "map50_95")


def factorial_effects(
    natural_640: float,
    repeat4_640: float,
    natural_1280: float,
    repeat4_1280: float,
) -> dict[str, float]:
    exposure_640 = repeat4_640 - natural_640
    exposure_1280 = repeat4_1280 - natural_1280
    resolution_natural = natural_1280 - natural_640
    resolution_repeat4 = repeat4_1280 - repeat4_640
    return {
        "exposure_effect_640": exposure_640,
        "exposure_effect_1280": exposure_1280,
        "resolution_effect_natural": resolution_natural,
        "resolution_effect_repeat4": resolution_repeat4,
        "difference_in_differences": exposure_1280 - exposure_640,
    }


def four_cell_metric_rows(
    natural_640: dict[str, Any],
    repeat4_640: dict[str, Any],
    natural_1280: dict[str, Any],
    repeat4_1280: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations = (natural_640, repeat4_640, natural_1280, repeat4_1280)
    class_sets = [set(item["per_class"]) for item in evaluations]
    if any(class_set != class_sets[0] for class_set in class_sets[1:]):
        raise ValueError("evaluation manifests contain different class sets")

    def row(metric: str, values: tuple[float, float, float, float]) -> dict[str, Any]:
        return {
            "metric": metric,
            "natural_640": values[0],
            "repeat4_640": values[1],
            "natural_1280": values[2],
            "repeat4_1280": values[3],
            **factorial_effects(*values),
        }

    aggregate_rows = [
        row(metric, tuple(float(item["aggregate"][metric]) for item in evaluations))
        for metric in METRICS
    ]
    class_rows: list[dict[str, Any]] = []
    for category in natural_640["per_class"]:
        for metric in METRICS:
            values = tuple(
                float(item["per_class"][category][metric]) for item in evaluations
            )
            class_rows.append({"category": category, **row(metric, values)})
    return pd.DataFrame(aggregate_rows), pd.DataFrame(class_rows)


def four_cell_role_row(
    role: str,
    annotations: Path,
    natural_640_predictions: Path,
    repeat4_640_predictions: Path,
    natural_1280_predictions: Path,
    repeat4_1280_predictions: Path,
    target: str,
    iou: float,
) -> dict[str, Any]:
    exposure_640 = role_recall_row(
        role, annotations, natural_640_predictions, repeat4_640_predictions, target, iou
    )
    natural_resolution = role_recall_row(
        role, annotations, natural_640_predictions, natural_1280_predictions, target, iou
    )
    repeat4_interaction = role_recall_row(
        role, annotations, natural_640_predictions, repeat4_1280_predictions, target, iou
    )
    recalls = (
        float(exposure_640["natural_640_recall"]),
        float(exposure_640["repeat4_640_recall"]),
        float(natural_resolution["repeat4_640_recall"]),
        float(repeat4_interaction["repeat4_640_recall"]),
    )
    return {
        "role": role,
        "target_ground_truth": exposure_640["target_ground_truth"],
        "target_images": exposure_640["target_images"],
        "source_groups": exposure_640["source_groups"],
        "natural_640_matched": exposure_640["natural_640_matched"],
        "repeat4_640_matched": exposure_640["repeat4_640_matched"],
        "natural_1280_matched": natural_resolution["repeat4_640_matched"],
        "repeat4_1280_matched": repeat4_interaction["repeat4_640_matched"],
        "natural_640_recall": recalls[0],
        "repeat4_640_recall": recalls[1],
        "natural_1280_recall": recalls[2],
        "repeat4_1280_recall": recalls[3],
        **factorial_effects(*recalls),
        "natural_640_target_predictions": exposure_640["natural_target_predictions"],
        "repeat4_640_target_predictions": exposure_640["repeat4_target_predictions"],
        "natural_1280_target_predictions": natural_resolution["repeat4_target_predictions"],
        "repeat4_1280_target_predictions": repeat4_interaction[
            "repeat4_target_predictions"
        ],
        "natural_640_target_max_score": exposure_640["natural_target_max_score"],
        "repeat4_640_target_max_score": exposure_640["repeat4_target_max_score"],
        "natural_1280_target_max_score": natural_resolution["repeat4_target_max_score"],
        "repeat4_1280_target_max_score": repeat4_interaction[
            "repeat4_target_max_score"
        ],
        "annotations_sha256": exposure_640["annotations_sha256"],
        "natural_640_predictions_sha256": exposure_640["natural_predictions_sha256"],
        "repeat4_640_predictions_sha256": exposure_640["repeat4_predictions_sha256"],
        "natural_1280_predictions_sha256": natural_resolution[
            "repeat4_predictions_sha256"
        ],
        "repeat4_1280_predictions_sha256": repeat4_interaction[
            "repeat4_predictions_sha256"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen four-cell 2x2 control")
    parser.add_argument("--natural-640-evaluation", required=True, type=Path)
    parser.add_argument("--repeat4-640-evaluation", required=True, type=Path)
    parser.add_argument("--natural-1280-evaluation", required=True, type=Path)
    parser.add_argument("--repeat4-1280-evaluation", required=True, type=Path)
    parser.add_argument(
        "--role",
        action="append",
        nargs=6,
        metavar=(
            "NAME",
            "ANNOTATIONS",
            "NATURAL640",
            "REPEAT4_640",
            "NATURAL1280",
            "REPEAT4_1280",
        ),
        required=True,
    )
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    evaluations = tuple(
        load_json(path)
        for path in (
            args.natural_640_evaluation,
            args.repeat4_640_evaluation,
            args.natural_1280_evaluation,
            args.repeat4_1280_evaluation,
        )
    )
    if any(item["official_validation_used"] for item in evaluations):
        raise ValueError("factorial analysis must not use official validation")
    if [item["image_size"] for item in evaluations] != [640, 640, 1280, 1280]:
        raise ValueError("expected Natural-640, Repeat4-640, Natural-1280, Repeat4-1280")
    if any(item["evaluation_split"] != "detector-dev" for item in evaluations):
        raise ValueError("all model metrics must use detector-dev")

    aggregate, classes = four_cell_metric_rows(*evaluations)
    roles = pd.DataFrame(
        [
            four_cell_role_row(
                role[0],
                Path(role[1]),
                Path(role[2]),
                Path(role[3]),
                Path(role[4]),
                Path(role[5]),
                args.target,
                args.iou,
            )
            for role in args.role
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "aggregate_metrics_2x2.csv", index=False)
    classes.to_csv(args.output_dir / "per_class_metrics_2x2.csv", index=False)
    roles.to_csv(args.output_dir / "target_role_recall_2x2.csv", index=False)

    target_rows = classes[
        (classes["category"] == args.target) & (classes["metric"] == "map50_95")
    ]
    if len(target_rows) != 1:
        raise ValueError(f"expected one target AP50-95 row for {args.target}")
    target_ap = target_rows.iloc[0]
    total_matches = {
        cell: int(roles[f"{cell}_matched"].sum())
        for cell in ("natural_640", "repeat4_640", "natural_1280", "repeat4_1280")
    }
    total_ground_truth = int(roles["target_ground_truth"].sum())
    report = {
        "comparison": "frozen 2x2 rare-positive exposure by input resolution",
        "official_validation_used": False,
        "checkpoint_selection_split": "detector-dev",
        "target_category": args.target,
        "matching_iou": args.iou,
        "confidence_floor": 0.001,
        "cell_order": ["natural_640", "repeat4_640", "natural_1280", "repeat4_1280"],
        "model_sha256": {
            cell: evaluation["model_sha256"]
            for cell, evaluation in zip(
                ("natural_640", "repeat4_640", "natural_1280", "repeat4_1280"),
                evaluations,
                strict=True,
            )
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "per_class": classes.to_dict(orient="records"),
        "target_roles": roles.to_dict(orient="records"),
        "target_summary": {
            "ground_truth_across_roles": total_ground_truth,
            "matched_across_roles": total_matches,
            "proposal_recall_across_roles": {
                cell: matches / total_ground_truth
                for cell, matches in total_matches.items()
            },
            "detector_dev_ap50_95": {
                cell: float(target_ap[cell])
                for cell in ("natural_640", "repeat4_640", "natural_1280", "repeat4_1280")
            },
        },
        "interaction_definition": (
            "difference_in_differences = (Repeat4_1280 - Natural_1280) - "
            "(Repeat4_640 - Natural_640)"
        ),
        "claim_boundary": (
            "Calibration-fit and policy-tune are one-time post-freeze diagnostics only. "
            "They did not select checkpoints, thresholds, or training settings; official "
            "validation was not used."
        ),
    }
    (args.output_dir / "rare_class_factorial_2x2.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    cells = report["cell_order"]
    labels = ["Natural 640", "Repeat4 640", "Natural 1280", "Repeat4 1280"]
    colors = ["#4C78A8", "#F58518", "#72B7B2", "#E45756"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    map_row = aggregate[aggregate["metric"] == "map50_95"].iloc[0]
    axes[0].bar(labels, [map_row[cell] for cell in cells], color=colors)
    axes[0].set_ylabel("mAP50-95")
    axes[0].set_title("Detector-dev aggregate")
    axes[0].tick_params(axis="x", rotation=25)

    x = np.arange(len(roles))
    width = 0.19
    for index, (cell, label, color) in enumerate(zip(cells, labels, colors, strict=True)):
        axes[1].bar(
            x + (index - 1.5) * width,
            roles[f"{cell}_recall"],
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, roles["role"], rotation=20, ha="right")
    axes[1].set_ylabel("Target proposal recall at IoU >= 0.50")
    axes[1].set_title("Post-freeze role diagnostics")
    axes[1].legend(frameon=False, fontsize=8)

    effects = [
        float(map_row["exposure_effect_640"]),
        float(map_row["exposure_effect_1280"]),
        float(map_row["resolution_effect_natural"]),
        float(map_row["resolution_effect_repeat4"]),
        float(map_row["difference_in_differences"]),
    ]
    effect_labels = ["Exposure\n640", "Exposure\n1280", "Resolution\nnatural", "Resolution\nRepeat4", "DiD"]
    axes[2].bar(effect_labels, effects, color=["#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Absolute mAP50-95 effect")
    axes[2].set_title("2x2 effects")
    axes[2].tick_params(axis="x", labelsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Rare-class resolution-by-exposure factorial")
    figure.tight_layout()
    figure.savefig(args.output_dir / "rare_class_factorial_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
