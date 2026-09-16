from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from maritime_calibration.metrics import evaluate_threshold, select_threshold_at_fp_budget


def load_summary(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune thresholds on policy data and evaluate them once on frozen validation"
    )
    parser.add_argument("--policy-table", required=True, type=Path)
    parser.add_argument("--policy-summary", required=True, type=Path)
    parser.add_argument("--evaluation-table", required=True, type=Path)
    parser.add_argument("--evaluation-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fp-per-image",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
    )
    args = parser.parse_args()

    policy = pd.read_csv(args.policy_table)
    evaluation = pd.read_csv(args.evaluation_table)
    policy_summary = load_summary(args.policy_summary)
    evaluation_summary = load_summary(args.evaluation_summary)
    probability_columns = sorted(
        set(column for column in policy if column.startswith("prob_"))
        & set(column for column in evaluation if column.startswith("prob_"))
    )
    if not probability_columns:
        raise ValueError("no shared prob_* columns found")

    rows = []
    for probability_column in probability_columns:
        for budget in args.fp_per_image:
            selected = select_threshold_at_fp_budget(
                policy,
                probability_column,
                int(policy_summary["ground_truth"]),
                int(policy_summary["images"]),
                budget,
            )
            evaluated = evaluate_threshold(
                evaluation,
                probability_column,
                selected["threshold"],
                int(evaluation_summary["ground_truth"]),
                int(evaluation_summary["images"]),
            )
            rows.append(
                {
                    "variant": probability_column.removeprefix("prob_"),
                    "target_fp_per_image": budget,
                    "threshold_selected_on_policy": selected["threshold"],
                    "policy_recall": selected["recall"],
                    "policy_fp_per_image": selected["fp_per_image"],
                    "evaluation_recall": evaluated["recall"],
                    "evaluation_fp_per_image": evaluated["fp_per_image"],
                    "evaluation_true_positives": evaluated["true_positives"],
                    "evaluation_false_positives": evaluated["false_positives"],
                }
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
