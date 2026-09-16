"""Check published table arithmetic and regenerate a CPU-only research summary."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = "results/calibration_metrics_complete_case.csv"
FACTORIAL = "results/sensitivities/rare_class_factorial/aggregate_metrics_2x2.csv"
EXTERNAL = "results/external_holdout/mobdrone/evaluation/condition_metrics.csv"
ARMS = {"repeat4_fullframe", "crop4_fullframe", "repeat4_384", "repeat4_768", "crop4_384", "crop4_768"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def equal(observed: float, expected: float, label: str) -> None:
    if not math.isfinite(observed) or not math.isclose(observed, expected, abs_tol=1e-10):
        raise ValueError(f"{label}: {observed} != {expected}")


def validate_factorial(rows: list[dict]) -> None:
    if len(rows) != 4 or {row["metric"] for row in rows} != {"precision", "recall", "map50", "map50_95"}:
        raise ValueError("expected exactly four factorial metrics")
    for row in rows:
        a, b, c, d = (float(row[key]) for key in ("natural_640", "repeat4_640", "natural_1280", "repeat4_1280"))
        expected = {
            "exposure_effect_640": b - a,
            "exposure_effect_1280": d - c,
            "resolution_effect_natural": c - a,
            "resolution_effect_repeat4": d - b,
            "difference_in_differences": (d - c) - (b - a),
        }
        for key, value in expected.items():
            equal(float(row[key]), value, f"{row['metric']}/{key}")


def validate_external(rows: list[dict]) -> None:
    if len(rows) != 6 or {row["condition"] for row in rows} != ARMS:
        raise ValueError("all six frozen external arms must be present exactly once")
    for row in rows:
        images, gt, predictions, tp = (int(row[key]) for key in (
            "images", "target_ground_truth", "target_predictions", "target_matches_iou_0_50"
        ))
        if (images, gt) != (1264, 350) or not 0 <= tp <= min(gt, predictions):
            raise ValueError("external sample size or matching counts changed")
        equal(float(row["target_recall_iou_0_50"]), tp / gt, "recall")
        equal(float(row["target_false_positives_per_image"]), (predictions - tp) / images, "FP/image")
        equal(float(row["uncapped_target_matches_iou_0_50"]), tp, "uncapped matches")
        for field in ("target_ap50", "target_ap50_95", "negative_image_target_prediction_rate"):
            if not 0 <= float(row[field]) <= 1:
                raise ValueError(f"invalid bounded metric: {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/reproduced_summary")
    args = parser.parse_args()
    calibration = read_rows(ROOT / CALIBRATION)
    factorial = read_rows(ROOT / FACTORIAL)
    external = read_rows(ROOT / EXTERNAL)
    validate_factorial(factorial)
    validate_external(external)
    variants = {row["variant"]: row for row in calibration}
    raw, full = variants["raw"], variants["full_metadata"]
    if not (float(full["ece"]) < float(raw["ece"]) and
            float(full["brier"]) > float(raw["brier"]) and
            float(full["nll"]) > float(raw["nll"])):
        raise ValueError("recorded calibration conclusion no longer matches the table")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    colors = ["#64748b", "#2563eb", "#d97706"]
    shown = [variants[name] for name in ("raw", "global_logistic", "full_metadata")]
    axes[0].bar(["Raw", "Global Platt", "Full metadata"], [float(r["brier"]) for r in shown], color=colors)
    axes[0].set(title="Calibration: lower Brier is better", ylabel="Brier score (official validation)")
    row = next(r for r in factorial if r["metric"] == "map50_95")
    for name, label, color in (("natural", "Natural", "#2563eb"), ("repeat4", "Repeat4", "#d97706")):
        axes[1].plot([640, 1280], [float(row[f"{name}_{size}"]) for size in (640, 1280)], "o-", label=label, color=color)
    axes[1].set(title="Resolution x exposure", xlabel="Input resolution", ylabel="mAP50-95 (detector-dev)", xticks=[640, 1280])
    axes[1].legend()
    for r in external:
        x, y = float(r["target_false_positives_per_image"]), float(r["target_recall_iou_0_50"])
        color = "#d97706" if r["condition"].startswith("crop4") else "#2563eb"
        axes[2].scatter(x, y, color=color)
        axes[2].annotate(r["condition"].replace("fullframe", "full"), (x, y), xytext=(3, 5), textcoords="offset points", fontsize=7)
    axes[2].set(title="External recall has a false-positive cost", xlabel="Target FP/image (MOBDrone)", ylabel="Target recall @ IoU .50", ylim=(-0.04, .65), xlim=(-.5, 20))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "research_summary.png", dpi=180)
    plt.close(fig)
    summary = {
        "scope": "consistency checks on published tables, not independent detector evaluation",
        "inputs_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (CALIBRATION, FACTORIAL, EXTERNAL)},
        "factorial_map50_95": row,
        "external_arms": external,
        "calibration_variants": calibration,
        "checks_passed": True,
    }
    (args.output_dir / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Published evidence arithmetic verified; summary JSON and figure regenerated.")


if __name__ == "__main__":
    main()
