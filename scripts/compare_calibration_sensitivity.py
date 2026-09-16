from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CALIBRATION_METRICS = ("ece", "brier", "nll")
DISPLAY_NAMES = {
    "raw": "Raw",
    "global_logistic": "Global Platt",
    "class_logistic": "Class-conditional",
    "altitude": "Altitude",
    "altitude_gimbal": "Altitude + gimbal",
    "full_metadata": "Full metadata",
}


def interval_status(lower: float, upper: float) -> str:
    if upper < 0:
        return "better_than_raw"
    if lower > 0:
        return "worse_than_raw"
    return "inconclusive"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("variant")


def comparison_tables(
    original: Path, sensitivity: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    detector_original = read_json(original / "baseline_detector.json")
    detector_stable = read_json(sensitivity / "baseline_detector.json")
    counts_original = read_json(original / "evaluation_counts.json")
    counts_stable = read_json(sensitivity / "evaluation_counts.json")

    detector_rows = []
    old_detector = detector_original["best_epoch_metrics_from_results_csv"]
    new_detector = detector_stable["best_epoch_metrics_from_results_csv"]
    for metric in ("precision", "recall", "map50", "map50_95"):
        detector_rows.append(
            {
                "metric": metric,
                "original": float(old_detector[metric]),
                "stable_noamp": float(new_detector[metric]),
                "stable_minus_original": float(new_detector[metric] - old_detector[metric]),
            }
        )
    detector_table = pd.DataFrame(detector_rows)

    original_metrics = indexed_metrics(original / "calibration_metrics_complete_case.csv")
    stable_metrics = indexed_metrics(sensitivity / "calibration_metrics_complete_case.csv")
    bootstrap = pd.read_csv(sensitivity / "bootstrap_metric_differences.csv").set_index(
        ["variant", "metric"]
    )
    calibration_rows = []
    for variant in stable_metrics.index:
        for metric in CALIBRATION_METRICS:
            original_value = float(original_metrics.loc[variant, metric])
            stable_value = float(stable_metrics.loc[variant, metric])
            stable_minus_raw = stable_value - float(stable_metrics.loc["raw", metric])
            row: dict[str, Any] = {
                "variant": variant,
                "metric": metric,
                "original": original_value,
                "stable_noamp": stable_value,
                "stable_minus_original": stable_value - original_value,
                "stable_minus_raw": stable_minus_raw,
                "bootstrap_ci_2.5": None,
                "bootstrap_ci_97.5": None,
                "bootstrap_status_vs_raw": "reference" if variant == "raw" else None,
            }
            if variant != "raw":
                interval = bootstrap.loc[(variant, metric)]
                lower = float(interval["ci_2.5"])
                upper = float(interval["ci_97.5"])
                row.update(
                    {
                        "bootstrap_ci_2.5": lower,
                        "bootstrap_ci_97.5": upper,
                        "bootstrap_status_vs_raw": interval_status(lower, upper),
                    }
                )
            calibration_rows.append(row)
    calibration_table = pd.DataFrame(calibration_rows)

    original_operating = pd.read_csv(original / "operating_points_complete_case.csv")
    stable_operating = pd.read_csv(sensitivity / "operating_points_complete_case.csv")
    original_operating = original_operating[
        np.isclose(original_operating["target_fp_per_image"], 0.5)
    ].set_index("variant")
    stable_operating = stable_operating[
        np.isclose(stable_operating["target_fp_per_image"], 0.5)
    ].set_index("variant")
    operating_rows = []
    for variant in stable_operating.index:
        original_recall = float(original_operating.loc[variant, "evaluation_recall"])
        stable_recall = float(stable_operating.loc[variant, "evaluation_recall"])
        operating_rows.append(
            {
                "variant": variant,
                "target_fp_per_image": 0.5,
                "original_evaluation_recall": original_recall,
                "stable_evaluation_recall": stable_recall,
                "stable_minus_original": stable_recall - original_recall,
                "stable_evaluation_fp_per_image": float(
                    stable_operating.loc[variant, "evaluation_fp_per_image"]
                ),
            }
        )
    operating_table = pd.DataFrame(operating_rows)

    coverage: dict[str, Any] = {}
    for population in (
        "metadata_complete_official_validation",
        "all_image_official_validation",
    ):
        old = counts_original[population]
        new = counts_stable[population]
        coverage[population] = {
            "images": int(new["images"]),
            "ground_truth": int(new["ground_truth"]),
            "original_matched_ground_truth": int(old["matched_ground_truth"]),
            "stable_matched_ground_truth": int(new["matched_ground_truth"]),
            "matched_ground_truth_delta": int(
                new["matched_ground_truth"] - old["matched_ground_truth"]
            ),
            "original_coverage": float(old["matched_ground_truth"] / old["ground_truth"]),
            "stable_coverage": float(new["matched_ground_truth"] / new["ground_truth"]),
            "original_predictions": int(old["predictions"]),
            "stable_predictions": int(new["predictions"]),
        }
    return detector_table, calibration_table, operating_table, coverage


def plot_comparison(
    detector: pd.DataFrame,
    operating: pd.DataFrame,
    coverage: dict[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    detector_plot = detector.set_index("metric").loc[["recall", "map50", "map50_95"]]
    x = np.arange(len(detector_plot))
    width = 0.36
    axes[0].bar(x - width / 2, detector_plot["original"], width, label="Original AMP")
    axes[0].bar(x + width / 2, detector_plot["stable_noamp"], width, label="Stable no-AMP")
    axes[0].set_xticks(x, ["Recall", "mAP50", "mAP50-95"])
    axes[0].set_ylim(0, 0.65)
    axes[0].set_title("Detector-dev quality")
    axes[0].legend(frameon=False)

    populations = [
        "metadata_complete_official_validation",
        "all_image_official_validation",
    ]
    labels = ["Metadata complete", "All images"]
    old_coverage = [coverage[name]["original_coverage"] for name in populations]
    new_coverage = [coverage[name]["stable_coverage"] for name in populations]
    x = np.arange(len(populations))
    axes[1].bar(x - width / 2, old_coverage, width, label="Original AMP")
    axes[1].bar(x + width / 2, new_coverage, width, label="Stable no-AMP")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 0.9)
    axes[1].set_title("GT proposal coverage at conf=0.001")

    variants = ["raw", "global_logistic", "altitude_gimbal", "full_metadata"]
    operation = operating.set_index("variant").loc[variants]
    x = np.arange(len(variants))
    axes[2].bar(x - width / 2, operation["original_evaluation_recall"], width)
    axes[2].bar(x + width / 2, operation["stable_evaluation_recall"], width)
    axes[2].set_xticks(x, [DISPLAY_NAMES[name] for name in variants], rotation=18, ha="right")
    axes[2].set_ylim(0, 0.65)
    axes[2].set_title("Recall at policy target 0.5 FP/image")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stable 640 no-AMP detector sensitivity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare stable-detector calibration sensitivity")
    parser.add_argument("--original-results", required=True, type=Path)
    parser.add_argument("--sensitivity-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    detector, calibration, operating, coverage = comparison_tables(
        args.original_results, args.sensitivity_results
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector.to_csv(args.output_dir / "detector_comparison.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_comparison.csv", index=False)
    operating.to_csv(args.output_dir / "fixed_fp_05_comparison.csv", index=False)
    plot_comparison(
        detector,
        operating,
        coverage,
        args.output_dir / "figures" / "comparison_to_original.png",
    )

    global_rows = calibration[calibration["variant"] == "global_logistic"].set_index("metric")
    metadata_rows = calibration[
        calibration["variant"].isin(["altitude_gimbal", "full_metadata"])
    ]
    summary = {
        "comparison": "original frozen AMP baseline versus stable 640 no-AMP sensitivity",
        "official_validation_used_for_checkpoint_selection": False,
        "detector": detector.set_index("metric").to_dict(orient="index"),
        "proposal_coverage": coverage,
        "fixed_fp_05": operating.set_index("variant").to_dict(orient="index"),
        "global_logistic_bootstrap_status": {
            metric: str(global_rows.loc[metric, "bootstrap_status_vs_raw"])
            for metric in CALIBRATION_METRICS
        },
        "metadata_bootstrap_status": {
            variant: subset.set_index("metric")["bootstrap_status_vs_raw"].to_dict()
            for variant, subset in metadata_rows.groupby("variant")
        },
        "claim_boundary": (
            "This is a planned detector-stability sensitivity. It tests whether calibration "
            "conclusions persist under a stable checkpoint; it is not a fresh untouched-validation study."
        ),
    }
    (args.output_dir / "comparison_to_original.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
