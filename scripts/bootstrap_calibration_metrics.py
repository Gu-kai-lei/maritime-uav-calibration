from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from maritime_calibration.metrics import calibration_metrics


METRICS = ("ece", "brier", "nll")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bootstrap confidence intervals for calibration metric differences"
    )
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-column", default="source_group")
    parser.add_argument("--reference", default="prob_raw")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    probability_columns = sorted(column for column in table if column.startswith("prob_"))
    if args.reference not in probability_columns:
        raise ValueError(f"reference column not found: {args.reference}")
    if table[args.group_column].isna().any():
        raise ValueError("grouped bootstrap requires a group for every detection row")
    groups = {
        group: indices.to_numpy()
        for group, indices in table.groupby(args.group_column).groups.items()
    }
    if len(groups) < 2:
        raise ValueError("at least two groups are required for a grouped bootstrap")

    labels = table["is_tp"].astype(int).to_numpy()
    probabilities = {
        column: table[column].astype(float).to_numpy() for column in probability_columns
    }
    point = {
        column: calibration_metrics(labels, values, bins=args.bins)
        for column, values in probabilities.items()
    }
    rng = np.random.default_rng(args.seed)
    group_names = np.array(list(groups), dtype=object)
    differences = {
        (column, metric): []
        for column in probability_columns
        if column != args.reference
        for metric in METRICS
    }
    for _ in range(args.repetitions):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.concatenate([groups[group] for group in sampled_groups])
        sample_labels = labels[indices]
        reference_metrics = calibration_metrics(
            sample_labels, probabilities[args.reference][indices], bins=args.bins
        )
        for column in probability_columns:
            if column == args.reference:
                continue
            candidate_metrics = calibration_metrics(
                sample_labels, probabilities[column][indices], bins=args.bins
            )
            for metric in METRICS:
                differences[(column, metric)].append(
                    candidate_metrics[metric] - reference_metrics[metric]
                )

    rows = []
    for (column, metric), values in differences.items():
        array = np.asarray(values)
        rows.append(
            {
                "variant": column.removeprefix("prob_"),
                "reference": args.reference.removeprefix("prob_"),
                "metric": metric,
                "point_difference": point[column][metric] - point[args.reference][metric],
                "bootstrap_median_difference": float(np.median(array)),
                "ci_2.5": float(np.quantile(array, 0.025)),
                "ci_97.5": float(np.quantile(array, 0.975)),
                "groups": len(groups),
                "repetitions": args.repetitions,
                "interpretation": "negative favors candidate",
            }
        )
    output = pd.DataFrame(rows).sort_values(["metric", "variant"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
