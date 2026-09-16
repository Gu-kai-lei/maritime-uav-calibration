from __future__ import annotations

import math

import numpy as np
import pandas as pd


def reliability_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have identical shapes")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": bin_id,
                    "lower": bin_id / bins,
                    "upper": (bin_id + 1) / bins,
                    "count": 0,
                    "confidence": math.nan,
                    "precision": math.nan,
                    "gap": math.nan,
                }
            )
            continue
        confidence = float(probabilities[mask].mean())
        precision = float(labels[mask].mean())
        rows.append(
            {
                "bin": bin_id,
                "lower": bin_id / bins,
                "upper": (bin_id + 1) / bins,
                "count": count,
                "confidence": confidence,
                "precision": precision,
                "gap": abs(confidence - precision),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    table = reliability_table(labels, clipped, bins=bins)
    total = max(int(table["count"].sum()), 1)
    ece = float(((table["count"] / total) * table["gap"].fillna(0)).sum())
    mce = float(table["gap"].max(skipna=True)) if total else math.nan
    brier = float(np.mean((clipped - labels) ** 2))
    nll = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    return {"ece": ece, "mce": mce, "brier": brier, "nll": nll}


def operating_curve(
    table: pd.DataFrame,
    probability_column: str,
    ground_truth_count: int,
    image_count: int,
) -> pd.DataFrame:
    if ground_truth_count <= 0 or image_count <= 0:
        raise ValueError("ground_truth_count and image_count must be positive")
    grouped = (
        table.assign(
            _tp=table["is_tp"].astype(int),
            _fp=1 - table["is_tp"].astype(int),
        )
        .groupby(probability_column, as_index=False)[["_tp", "_fp"]]
        .sum()
        .sort_values(probability_column, ascending=False)
    )
    grouped["cumulative_tp"] = grouped["_tp"].cumsum()
    grouped["cumulative_fp"] = grouped["_fp"].cumsum()
    grouped["recall"] = grouped["cumulative_tp"] / ground_truth_count
    grouped["fp_per_image"] = grouped["cumulative_fp"] / image_count
    return grouped[
        [probability_column, "cumulative_tp", "cumulative_fp", "recall", "fp_per_image"]
    ].rename(columns={probability_column: "threshold"})


def select_threshold_at_fp_budget(
    table: pd.DataFrame,
    probability_column: str,
    ground_truth_count: int,
    image_count: int,
    fp_per_image_budget: float,
) -> dict[str, float]:
    if fp_per_image_budget < 0:
        raise ValueError("fp_per_image_budget must be non-negative")
    curve = operating_curve(table, probability_column, ground_truth_count, image_count)
    eligible = curve[curve["fp_per_image"] <= fp_per_image_budget]
    if eligible.empty:
        return {
            "threshold": float(np.nextafter(1.0, 2.0)),
            "recall": 0.0,
            "fp_per_image": 0.0,
        }
    selected = eligible.iloc[-1]
    return {
        "threshold": float(selected["threshold"]),
        "recall": float(selected["recall"]),
        "fp_per_image": float(selected["fp_per_image"]),
    }


def evaluate_threshold(
    table: pd.DataFrame,
    probability_column: str,
    threshold: float,
    ground_truth_count: int,
    image_count: int,
) -> dict[str, float | int]:
    if ground_truth_count <= 0 or image_count <= 0:
        raise ValueError("ground_truth_count and image_count must be positive")
    retained = table[pd.to_numeric(table[probability_column]) >= threshold]
    true_positives = int(retained["is_tp"].astype(int).sum())
    false_positives = int(len(retained) - true_positives)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": true_positives / ground_truth_count,
        "fp_per_image": false_positives / image_count,
    }
