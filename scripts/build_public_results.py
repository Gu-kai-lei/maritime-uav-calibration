from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CLASS_ROW = re.compile(
    r"^\s*(all|swimmer|boat|jetski|life_saving_appliances|buoy)\s+"
    r"(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)
DISPLAY_NAMES = {
    "raw": "Raw",
    "global_logistic": "Global Platt",
    "class_logistic": "Class-conditional",
    "altitude": "Altitude",
    "altitude_gimbal": "Altitude + gimbal",
    "full_metadata": "Full metadata",
}
PALETTE = {
    "raw": "#6B7280",
    "global_logistic": "#2563EB",
    "class_logistic": "#8B5CF6",
    "altitude": "#F59E0B",
    "altitude_gimbal": "#EA580C",
    "full_metadata": "#DC2626",
}


def read_metric_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0).reset_index(names="variant")
    return table[["variant", "ece", "mce", "brier", "nll"]]


def parse_training_log(path: Path) -> dict[str, Any]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    batch_match = re.search(r"Using batch-size\s+(\d+)", text)
    early_stop_match = re.search(
        r"Training stopped early.*?Best results observed at epoch\s+(\d+)", text
    )
    duration_match = re.search(r"(\d+) epochs completed in ([0-9.]+) hours", text)
    class_metrics: dict[str, Any] = {}
    for line in text.splitlines():
        match = CLASS_ROW.match(line)
        if not match:
            continue
        name, images, instances, precision, recall, map50, map50_95 = match.groups()
        class_metrics[name] = {
            "images": int(images),
            "instances": int(instances),
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    return {
        "effective_batch_size": int(batch_match.group(1)) if batch_match else None,
        "best_epoch": int(early_stop_match.group(1)) if early_stop_match else None,
        "completed_epochs": int(duration_match.group(1)) if duration_match else None,
        "training_hours": float(duration_match.group(2)) if duration_match else None,
        "detector_dev_metrics": class_metrics,
    }


def first_metric_collapse(results: pd.DataFrame) -> int | None:
    """Return the first epoch at or below 10% of the preceding best AP."""
    best = 0.0
    for row in results.itertuples(index=False, name=None):
        epoch = int(row[results.columns.get_loc("epoch")])
        value = float(row[results.columns.get_loc("metrics/mAP50-95(B)")])
        if best > 0 and value <= 0.1 * best:
            return epoch
        best = max(best, value)
    return None


def detector_summary(run_dir: Path, training_log: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = [column.strip() for column in results.columns]
    metric_column = "metrics/mAP50-95(B)"
    best = results.loc[results[metric_column].idxmax()]
    log_summary = parse_training_log(training_log)
    collapse_epoch = first_metric_collapse(results)
    requested_batch = manifest.get("batch_requested", manifest.get("batch"))
    effective_batch = manifest.get("batch_effective", log_summary["effective_batch_size"])
    if collapse_epoch is None:
        stability_note = (
            "No metric-collapse event occurred under the frozen 10%-of-previous-best definition; "
            "training ended normally or by detector-dev early stopping."
        )
    else:
        stability_note = (
            f"Detector-dev AP collapsed at epoch {collapse_epoch} under the frozen definition; "
            "early stopping preserved the preceding best checkpoint."
        )
    return {
        "run_id": run_dir.name,
        "detector": "YOLOv8n",
        "initialization": "public COCO weights",
        "initialization_sha256": manifest["model_initialization_sha256"],
        "best_checkpoint_sha256": manifest["best_weights_sha256"],
        "data_yaml_sha256": manifest["data_yaml_sha256"],
        "image_size": int(manifest["imgsz"]),
        "seed": int(manifest["seed"]),
        "deterministic_requested": True,
        "epochs_requested": int(manifest["epochs"]),
        "epochs_completed": manifest.get("epochs_completed", log_summary["completed_epochs"]),
        "best_epoch": int(best["epoch"]),
        "early_stopping_patience": manifest.get(
            "patience", manifest.get("early_stopping_patience", 20)
        ),
        "requested_batch": requested_batch,
        "effective_batch_size": effective_batch,
        "amp": manifest.get("amp"),
        "optimizer": manifest.get("optimizer"),
        "training_hours": log_summary["training_hours"],
        "checkpoint_selection_split": "detector-dev",
        "official_validation_used_for_checkpoint_selection": False,
        "best_epoch_metrics_from_results_csv": {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]),
            "map50_95": float(best[metric_column]),
        },
        "final_best_checkpoint_validation": log_summary["detector_dev_metrics"],
        "environment": {
            "python": manifest["python"],
            "platform": manifest["platform"],
            "torch": manifest["torch"],
            "ultralytics": manifest["ultralytics"],
            "cuda_available": manifest["cuda_available"],
            "gpu": manifest["gpu"],
        },
        "metric_collapse_epoch": collapse_epoch,
        "training_stability_note": stability_note,
    }


def save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def plot_reliability(calibration_dir: Path, output: Path) -> None:
    variants = ["raw", "global_logistic", "altitude_gimbal", "full_metadata"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", linewidth=1, label="Ideal")
    for variant in variants:
        table = pd.read_csv(calibration_dir / f"reliability_{variant}.csv")
        ax.plot(
            table["confidence"],
            table["precision"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=PALETTE[variant],
            label=DISPLAY_NAMES[variant],
        )
    ax.set(xlabel="Mean predicted confidence", ylabel="Empirical precision", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Reliability on metadata-complete official validation")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap(table: pd.DataFrame, output: Path) -> None:
    variants = [
        "global_logistic",
        "class_logistic",
        "altitude",
        "altitude_gimbal",
        "full_metadata",
    ]
    metrics = [("ece", "ECE difference"), ("brier", "Brier difference"), ("nll", "NLL difference")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        subset = table[table["metric"] == metric].set_index("variant").loc[variants]
        y = np.arange(len(variants))
        point = subset["point_difference"].to_numpy()
        lower = subset["ci_2.5"].to_numpy()
        upper = subset["ci_97.5"].to_numpy()
        colors = [PALETTE[variant] for variant in variants]
        for row, variant in enumerate(variants):
            ax.errorbar(
                point[row],
                y[row],
                xerr=np.array(
                    [[point[row] - lower[row]], [upper[row] - point[row]]]
                ),
                fmt="o",
                color=PALETTE[variant],
                ecolor=PALETTE[variant],
                markersize=5,
                elinewidth=2,
                capsize=3,
            )
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Candidate minus raw (negative is better)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_yticks(y, [DISPLAY_NAMES[variant] for variant in variants])
        ax.invert_yaxis()
    fig.suptitle("Grouped bootstrap differences across 31 source groups (1,000 resamples)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_operating_points(table: pd.DataFrame, output: Path) -> None:
    variants = ["raw", "global_logistic", "class_logistic", "altitude_gimbal", "full_metadata"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for variant in variants:
        subset = table[table["variant"] == variant].sort_values("target_fp_per_image")
        ax.plot(
            subset["target_fp_per_image"],
            subset["evaluation_recall"],
            marker="o",
            linewidth=1.8,
            linestyle="--" if variant == "raw" else "-",
            markerfacecolor="white" if variant == "raw" else PALETTE[variant],
            color=PALETTE[variant],
            label=DISPLAY_NAMES[variant],
            zorder=4 if variant == "raw" else 3,
        )
    ax.set(
        xlabel="Target false positives per image on policy-tune",
        ylabel="Recall on official validation",
        xlim=(0.08, 1.02),
        ylim=(0, max(0.65, float(table["evaluation_recall"].max()) + 0.04)),
    )
    ax.set_title("Recall at policy-selected false-positive budgets")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_source_roles(table: pd.DataFrame, output: Path) -> None:
    variants = ["raw", "global_logistic", "altitude_gimbal", "full_metadata"]
    roles = ["calibrator_source_unseen", "calibration_source_seen", "policy_source_seen"]
    labels = ["Calibrator-source unseen", "Calibration-source seen", "Policy-source seen"]
    subset = table[
        (table["dimension"] == "source_role")
        & table["variant"].isin(variants)
        & table["slice"].isin(roles)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    x = np.arange(len(roles))
    width = 0.19
    for offset, variant in enumerate(variants):
        ordered = subset[subset["variant"] == variant].set_index("slice").loc[roles]
        for ax, metric, title in zip(axes, ("brier", "nll"), ("Brier score", "Negative log-likelihood"), strict=True):
            ax.bar(
                x + (offset - 1.5) * width,
                ordered[metric],
                width,
                color=PALETTE[variant],
                label=DISPLAY_NAMES[variant],
            )
            ax.set_title(title)
            ax.set_xticks(x, labels, rotation=16, ha="right")
            ax.set_ylabel(f"{title} (lower is better)")
    fig.suptitle("Metadata gains do not transfer to calibrator-source-unseen frames")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_detector_classes(summary: dict[str, Any], output: Path) -> None:
    metrics = summary["final_best_checkpoint_validation"]
    classes = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
    labels = ["Swimmer", "Boat", "Jetski", "Life-saving appliance", "Buoy"]
    values = [metrics[name]["map50"] for name in classes]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(labels, values, color="#2563EB")
    ax.set(xlabel="AP50 on detector-dev", xlim=(0, 1))
    ax.set_title("Detector baseline per-class performance")
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact, versioned tables and figures")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--complete-case-dir", required=True, type=Path)
    parser.add_argument("--all-images-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--metadata-val-summary", required=True, type=Path)
    parser.add_argument("--all-val-summary", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    detector = detector_summary(args.run_dir, args.training_log)
    save_json(detector, args.output_dir / "baseline_detector.json")
    evaluation_counts = {
        "metadata_complete_official_validation": json.loads(
            args.metadata_val_summary.read_text(encoding="utf-8")
        ),
        "all_image_official_validation": json.loads(
            args.all_val_summary.read_text(encoding="utf-8")
        ),
        "prediction_confidence_floor": 0.001,
        "matching_iou": 0.50,
    }
    save_json(evaluation_counts, args.output_dir / "evaluation_counts.json")

    complete_metrics = read_metric_csv(args.complete_case_dir / "calibration_metrics.csv")
    all_metrics = read_metric_csv(args.all_images_dir / "calibration_metrics.csv")
    complete_operating = pd.read_csv(args.complete_case_dir / "operating_points.csv")
    all_operating = pd.read_csv(args.all_images_dir / "operating_points.csv")
    bootstrap = pd.read_csv(args.analysis_dir / "bootstrap_metric_differences.csv")
    slices = pd.read_csv(args.analysis_dir / "calibration_metrics_by_slice.csv")
    slice_operating = pd.read_csv(args.analysis_dir / "operating_points_by_slice.csv")

    complete_metrics.to_csv(args.output_dir / "calibration_metrics_complete_case.csv", index=False)
    all_metrics.to_csv(args.output_dir / "calibration_metrics_all_images.csv", index=False)
    complete_operating.to_csv(args.output_dir / "operating_points_complete_case.csv", index=False)
    all_operating.to_csv(args.output_dir / "operating_points_all_images.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_metric_differences.csv", index=False)
    slices[slices["dimension"] == "source_role"].to_csv(
        args.output_dir / "source_role_calibration_metrics.csv", index=False
    )
    slice_operating[slice_operating["dimension"] == "source_role"].to_csv(
        args.output_dir / "source_role_operating_points.csv", index=False
    )
    shutil.copyfile(args.analysis_dir / "source_overlap.json", args.output_dir / "source_overlap.json")

    plot_reliability(args.complete_case_dir, figures / "reliability_diagram.png")
    plot_bootstrap(bootstrap, figures / "bootstrap_metric_differences.png")
    plot_operating_points(complete_operating, figures / "fixed_fp_recall.png")
    plot_source_roles(slices, figures / "source_role_proper_scores.png")
    plot_detector_classes(detector, figures / "detector_per_class_ap50.png")

    print(f"Public results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
