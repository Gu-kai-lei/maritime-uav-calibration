from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from maritime_calibration.calibration import FEATURE_VARIANTS, fit_calibrator
from maritime_calibration.metrics import calibration_metrics, reliability_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate detection calibrators")
    parser.add_argument("--calibration-table", required=True, type=Path)
    parser.add_argument("--policy-table", required=True, type=Path)
    parser.add_argument("--evaluation-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument(
        "--analysis",
        choices=("complete-case", "all-images"),
        default="complete-case",
        help="Primary analysis requires altitude and gimbal metadata for every retained row",
    )
    args = parser.parse_args()

    calibration = pd.read_csv(args.calibration_table)
    policy = pd.read_csv(args.policy_table)
    evaluation = pd.read_csv(args.evaluation_table)
    if args.analysis == "complete-case":
        required_metadata = ["altitude", "gimbal_pitch"]
        calibration = calibration.dropna(subset=required_metadata).copy()
        policy = policy.dropna(subset=required_metadata).copy()
        evaluation = evaluation.dropna(subset=required_metadata).copy()
    if calibration.empty or policy.empty or evaluation.empty:
        raise ValueError(f"analysis={args.analysis} produced an empty table")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fitted = {}
    variants = (
        FEATURE_VARIANTS
        if args.analysis == "complete-case"
        else ("global_logistic", "class_logistic")
    )
    for variant in variants:
        calibrator = fit_calibrator(calibration, variant)
        calibrator.save(str(args.output_dir / f"{variant}.joblib"))
        fitted[variant] = calibrator

    def with_probabilities(table: pd.DataFrame) -> pd.DataFrame:
        output = table.copy()
        output["prob_raw"] = output["score"].astype(float).clip(0, 1)
        for name, calibrator in fitted.items():
            output[f"prob_{name}"] = calibrator.predict(output)
        return output

    policy_output = with_probabilities(policy)
    evaluation_output = with_probabilities(evaluation)
    policy_output.to_csv(args.output_dir / "policy_probabilities.csv", index=False)
    evaluation_output.to_csv(args.output_dir / "evaluation_probabilities.csv", index=False)

    probabilities = {
        column.removeprefix("prob_"): evaluation_output[column].to_numpy()
        for column in evaluation_output.columns
        if column.startswith("prob_")
    }

    labels = evaluation["is_tp"].astype(int).to_numpy()
    reports = {
        name: calibration_metrics(labels, values, bins=args.bins)
        for name, values in probabilities.items()
    }
    pd.DataFrame(reports).T.to_csv(args.output_dir / "calibration_metrics.csv")
    (args.output_dir / "calibration_metrics.json").write_text(
        json.dumps(
            {
                "analysis": args.analysis,
                "calibration_rows": len(calibration),
                "policy_rows": len(policy),
                "evaluation_rows": len(evaluation),
                "metrics": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
    for name, values in probabilities.items():
        table = reliability_table(labels, values, bins=args.bins)
        populated = table[table["count"] > 0]
        table.to_csv(args.output_dir / f"reliability_{name}.csv", index=False)
        axis.plot(populated["confidence"], populated["precision"], marker="o", label=name)
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Empirical precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "reliability_diagram.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "analysis": args.analysis,
                "calibration_rows": len(calibration),
                "policy_rows": len(policy),
                "evaluation_rows": len(evaluation),
                "metrics": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
