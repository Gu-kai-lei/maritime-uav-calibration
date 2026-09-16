from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # Support both ``python scripts/...`` and imports from the test suite.
    from scripts.compare_detector_replays import (  # noqa: E402
        METRIC_COLUMN,
        checkpoint_state,
        find_collapse_epoch,
        load_results,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from compare_detector_replays import (  # type: ignore[no-redef]  # noqa: E402
        METRIC_COLUMN,
        checkpoint_state,
        find_collapse_epoch,
        load_results,
    )


PLOT_COLUMNS = {
    METRIC_COLUMN: "Detector-dev mAP50-95",
    "train/box_loss": "Training box loss",
    "train/cls_loss": "Training class loss",
    "train/dfl_loss": "Training DFL loss",
}


def stability_gate(
    collapse_epoch: int | None,
    best_checkpoint: dict[str, Any],
    final_checkpoint: dict[str, Any],
    stderr_bytes: int,
) -> dict[str, Any]:
    checks = {
        "no_metric_collapse": collapse_epoch is None,
        "best_checkpoint_finite": best_checkpoint["nonfinite_values"] == 0,
        "final_checkpoint_finite": final_checkpoint["nonfinite_values"] == 0,
        "stderr_empty": stderr_bytes == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def best_metrics(table: pd.DataFrame) -> dict[str, float | int]:
    row = table.loc[table[METRIC_COLUMN].idxmax()]
    return {
        "epoch": int(row["epoch"]),
        "precision": float(row["metrics/precision(B)"]),
        "recall": float(row["metrics/recall(B)"]),
        "map50": float(row["metrics/mAP50(B)"]),
        "map50_95": float(row[METRIC_COLUMN]),
    }


def plot_control(
    failed: pd.DataFrame,
    control: pd.DataFrame,
    failed_collapse: int | None,
    control_collapse: int | None,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, (column, title) in zip(axes.flat, PLOT_COLUMNS.items(), strict=True):
        ax.plot(
            failed["epoch"],
            failed[column],
            color="#6B7280",
            linewidth=1.8,
            label="AMP primary",
        )
        ax.plot(
            control["epoch"],
            control[column],
            color="#2563EB",
            linewidth=2,
            label="No-AMP control",
        )
        if failed_collapse is not None:
            ax.axvline(
                failed_collapse,
                color="#DC2626",
                linestyle=":",
                linewidth=1.2,
                label="AMP collapse" if column == METRIC_COLUMN else None,
            )
        if control_collapse is not None:
            ax.axvline(
                control_collapse,
                color="#F59E0B",
                linestyle="--",
                linewidth=1.2,
                label="Control collapse" if column == METRIC_COLUMN else None,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("AMP failure versus the frozen no-AMP stability control")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen no-AMP stability control")
    parser.add_argument("--failed-run", required=True, type=Path)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--failed-stderr", required=True, type=Path)
    parser.add_argument("--control-stderr", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    args = parser.parse_args()

    failed = load_results(args.failed_run / "results.csv")
    control = load_results(args.control_run / "results.csv")
    failed_manifest = json.loads(
        (args.failed_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    control_manifest = json.loads(
        (args.control_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    failed_collapse = find_collapse_epoch(failed)
    control_collapse = find_collapse_epoch(control)
    control_best, _ = checkpoint_state(args.control_run / "weights" / "best.pt")
    control_final, _ = checkpoint_state(args.control_run / "weights" / "last.pt")
    stderr_bytes = args.control_stderr.stat().st_size
    gate = stability_gate(control_collapse, control_best, control_final, stderr_bytes)

    output = {
        "comparison": "frozen AMP primary versus single-factor no-AMP stability control",
        "failed_run": args.failed_run.name,
        "control_run": args.control_run.name,
        "frozen_inputs": {
            "initialization_sha256_equal": (
                failed_manifest["model_initialization_sha256"]
                == control_manifest["model_initialization_sha256"]
            ),
            "data_yaml_sha256_equal": (
                failed_manifest["data_yaml_sha256"]
                == control_manifest["data_yaml_sha256"]
            ),
            "image_size_equal": failed_manifest["imgsz"] == control_manifest["imgsz"],
            "seed_equal": failed_manifest["seed"] == control_manifest["seed"],
            "effective_batch_failed": 16,
            "effective_batch_control": control_manifest["batch_effective"],
            "amp_failed": True,
            "amp_control": control_manifest["amp"],
            "official_validation_used_for_selection": False,
        },
        "failed": {
            "epochs_completed": len(failed),
            "collapse_epoch": failed_collapse,
            "best": best_metrics(failed),
            "stderr_bytes": args.failed_stderr.stat().st_size,
        },
        "control": {
            "epochs_completed": len(control),
            "collapse_epoch": control_collapse,
            "best": best_metrics(control),
            "stderr_bytes": stderr_bytes,
            "best_checkpoint": control_best,
            "final_checkpoint": control_final,
        },
        "stability_gate": gate,
        "interpretation": (
            "Passing this one-run control supports an AMP-path explanation but does not establish "
            "a general framework bug; failing it rules out AMP as a sufficient single-factor fix."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_control(failed, control, failed_collapse, control_collapse, args.output_figure)
    print(json.dumps({"stability_gate": gate, "control_best": output["control"]["best"]}, indent=2))


if __name__ == "__main__":
    main()
