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


def three_cell_metric_rows(
    natural_640: dict[str, Any],
    repeat4_640: dict[str, Any],
    natural_1280: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_sets = [set(item["per_class"]) for item in (natural_640, repeat4_640, natural_1280)]
    if class_sets[0] != class_sets[1] or class_sets[0] != class_sets[2]:
        raise ValueError("evaluation manifests contain different class sets")
    aggregate_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        baseline = float(natural_640["aggregate"][metric])
        exposure = float(repeat4_640["aggregate"][metric])
        resolution = float(natural_1280["aggregate"][metric])
        aggregate_rows.append(
            {
                "metric": metric,
                "natural_640": baseline,
                "repeat4_640": exposure,
                "natural_1280": resolution,
                "delta_repeat4_minus_natural640": exposure - baseline,
                "delta_natural1280_minus_natural640": resolution - baseline,
            }
        )
    class_rows: list[dict[str, Any]] = []
    for category in natural_640["per_class"]:
        for metric in METRICS:
            baseline = float(natural_640["per_class"][category][metric])
            exposure = float(repeat4_640["per_class"][category][metric])
            resolution = float(natural_1280["per_class"][category][metric])
            class_rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "natural_640": baseline,
                    "repeat4_640": exposure,
                    "natural_1280": resolution,
                    "delta_repeat4_minus_natural640": exposure - baseline,
                    "delta_natural1280_minus_natural640": resolution - baseline,
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(class_rows)


def three_cell_role_row(
    role: str,
    annotations: Path,
    natural_640_predictions: Path,
    repeat4_640_predictions: Path,
    natural_1280_predictions: Path,
    target: str,
    iou: float,
) -> dict[str, Any]:
    exposure = role_recall_row(
        role, annotations, natural_640_predictions, repeat4_640_predictions, target, iou
    )
    resolution = role_recall_row(
        role, annotations, natural_640_predictions, natural_1280_predictions, target, iou
    )
    return {
        **exposure,
        "natural_1280_matched": resolution["repeat4_640_matched"],
        "natural_1280_recall": resolution["repeat4_640_recall"],
        "delta_natural1280_minus_natural640": resolution[
            "delta_repeat4_minus_natural"
        ],
        "natural_1280_target_predictions": resolution["repeat4_target_predictions"],
        "natural_1280_target_max_score": resolution["repeat4_target_max_score"],
        "natural_1280_predictions_sha256": resolution["repeat4_predictions_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the registered three-cell control")
    parser.add_argument("--natural-640-evaluation", required=True, type=Path)
    parser.add_argument("--repeat4-640-evaluation", required=True, type=Path)
    parser.add_argument("--natural-1280-evaluation", required=True, type=Path)
    parser.add_argument(
        "--role",
        action="append",
        nargs=5,
        metavar=("NAME", "ANNOTATIONS", "NATURAL640", "REPEAT4", "NATURAL1280"),
        required=True,
    )
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    natural_640 = load_json(args.natural_640_evaluation)
    repeat4_640 = load_json(args.repeat4_640_evaluation)
    natural_1280 = load_json(args.natural_1280_evaluation)
    evaluations = (natural_640, repeat4_640, natural_1280)
    if any(item["official_validation_used"] for item in evaluations):
        raise ValueError("factorial analysis must not use official validation")
    if [item["image_size"] for item in evaluations] != [640, 640, 1280]:
        raise ValueError("expected Natural-640, Repeat4-640, and Natural-1280 evaluations")
    if any(item["evaluation_split"] != "detector-dev" for item in evaluations):
        raise ValueError("all model metrics must use detector-dev")

    aggregate, classes = three_cell_metric_rows(natural_640, repeat4_640, natural_1280)
    roles = pd.DataFrame(
        [
            three_cell_role_row(
                role[0], Path(role[1]), Path(role[2]), Path(role[3]), Path(role[4]),
                args.target, args.iou
            )
            for role in args.role
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "aggregate_metrics_complete.csv", index=False)
    classes.to_csv(args.output_dir / "per_class_metrics_complete.csv", index=False)
    roles.to_csv(args.output_dir / "target_role_recall_complete.csv", index=False)

    baseline_map = float(natural_640["aggregate"]["map50_95"])
    exposure_map = float(repeat4_640["aggregate"]["map50_95"])
    resolution_map = float(natural_1280["aggregate"]["map50_95"])
    exposure_relative = (exposure_map - baseline_map) / baseline_map
    resolution_relative = (resolution_map - baseline_map) / baseline_map
    baseline_matches = int(roles["natural_640_matched"].sum())
    exposure_matches = int(roles["repeat4_640_matched"].sum())
    resolution_matches = int(roles["natural_1280_matched"].sum())
    exposure_useful = exposure_matches > baseline_matches and exposure_relative >= -0.10
    resolution_useful = resolution_matches > baseline_matches and resolution_relative >= -0.10
    report = {
        "comparison": "Natural-640 versus Repeat4-640 versus Natural-1280",
        "official_validation_used": False,
        "target_category": args.target,
        "matching_iou": args.iou,
        "confidence_floor": 0.001,
        "model_sha256": {
            "natural_640": natural_640["model_sha256"],
            "repeat4_640": repeat4_640["model_sha256"],
            "natural_1280": natural_1280["model_sha256"],
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "target_roles": roles.to_dict(orient="records"),
        "gate": {
            "natural_640_target_matches": baseline_matches,
            "repeat4_640_target_matches": exposure_matches,
            "natural_1280_target_matches": resolution_matches,
            "repeat4_640_map50_95_relative_change": exposure_relative,
            "natural_1280_map50_95_relative_change": resolution_relative,
            "repeat4_main_effect_useful": exposure_useful,
            "resolution_main_effect_useful": resolution_useful,
            "at_least_one_main_effect_useful": exposure_useful or resolution_useful,
            "repeat4_1280_preregistered_gate_met": exposure_useful or resolution_useful,
            "repeat4_1280_launched": False,
            "repeat4_1280_requires_user_decision": True,
        },
        "claim_boundary": (
            "Calibration-fit and policy-tune are one-time post-freeze diagnostics and did not "
            "select checkpoints, thresholds, or training settings."
        ),
    }
    (args.output_dir / "rare_class_factorial_complete.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    class_map = classes[classes["metric"] == "map50_95"].copy()
    x = np.arange(len(class_map))
    width = 0.25
    axes[0].bar(x - width, class_map["natural_640"], width, label="Natural 640")
    axes[0].bar(x, class_map["repeat4_640"], width, label="Repeat4 640")
    axes[0].bar(x + width, class_map["natural_1280"], width, label="Natural 1280")
    axes[0].set_xticks(x, class_map["category"], rotation=25, ha="right")
    axes[0].set_ylabel("AP50-95")
    axes[0].set_title("Detector-dev per-class AP")
    axes[0].legend(frameon=False)
    x = np.arange(len(roles))
    axes[1].bar(x - width, roles["natural_640_recall"], width, label="Natural 640")
    axes[1].bar(x, roles["repeat4_640_recall"], width, label="Repeat4 640")
    axes[1].bar(x + width, roles["natural_1280_recall"], width, label="Natural 1280")
    axes[1].set_xticks(x, roles["role"], rotation=20, ha="right")
    axes[1].set_ylabel("Target proposal recall at IoU >= 0.50")
    axes[1].set_title("Rare-class one-time role diagnostics")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Registered resolution-by-exposure main effects")
    figure.tight_layout()
    figure.savefig(args.output_dir / "rare_class_factorial_complete.png", dpi=200,
                   bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
