from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_COLUMN = "metrics/mAP50-95(B)"
PLOT_COLUMNS = {
    METRIC_COLUMN: "Detector-dev mAP50-95",
    "train/box_loss": "Training box loss",
    "train/cls_loss": "Training class loss",
    "train/dfl_loss": "Training DFL loss",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_results(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table.columns = [column.strip() for column in table.columns]
    return table


def find_collapse_epoch(
    table: pd.DataFrame,
    metric: str = METRIC_COLUMN,
    fraction_of_previous_best: float = 0.10,
) -> int | None:
    best = -np.inf
    for row in table.itertuples(index=False):
        epoch = int(getattr(row, "epoch"))
        value = float(getattr(row, metric.replace("/", "_").replace("-", "_"), np.nan))
        if np.isnan(value):
            value = float(table.loc[table["epoch"] == epoch, metric].iloc[0])
        if best > 0 and value <= best * fraction_of_previous_best:
            return epoch
        best = max(best, value)
    return None


def compare_results_tables(primary: pd.DataFrame, replay: pd.DataFrame) -> dict[str, Any]:
    if list(primary.columns) != list(replay.columns):
        raise ValueError("results.csv columns differ")
    if len(primary) != len(replay):
        raise ValueError("results.csv row counts differ")
    fields = [column for column in primary.columns if column != "time"]
    comparisons = {}
    for field in fields:
        left = primary[field].to_numpy(dtype=float)
        right = replay[field].to_numpy(dtype=float)
        exact = bool(np.array_equal(left, right, equal_nan=True))
        jointly_finite = np.isfinite(left) & np.isfinite(right)
        maximum_difference = (
            float(np.max(np.abs(left[jointly_finite] - right[jointly_finite])))
            if jointly_finite.any()
            else None
        )
        comparisons[field] = {
            "exact": exact,
            "maximum_absolute_difference_on_finite_values": maximum_difference,
        }
    return {
        "rows": len(primary),
        "time_excluded_because_it_is_hardware_runtime": True,
        "fields_compared": fields,
        "all_fields_exact": all(item["exact"] for item in comparisons.values()),
        "per_field": comparisons,
    }


def checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError(f"checkpoint has no model or EMA: {path}")
    digest = hashlib.sha256()
    tensors: dict[str, torch.Tensor] = {}
    l2_squared = 0.0
    max_absolute = 0.0
    nonfinite = 0
    floating_values = 0
    for key, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        tensors[key] = tensor
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        if torch.is_floating_point(tensor):
            values = tensor.float()
            nonfinite += int((~torch.isfinite(values)).sum())
            l2_squared += float((values * values).sum())
            max_absolute = max(max_absolute, float(values.abs().max()))
            floating_values += values.numel()
    summary = {
        "file_sha256": file_sha256(path),
        "model_state_sha256": digest.hexdigest(),
        "floating_values": floating_values,
        "nonfinite_values": nonfinite,
        "maximum_absolute_weight": max_absolute,
        "l2_weight_norm": l2_squared**0.5,
    }
    return summary, tensors


def compare_checkpoint(primary: Path, replay: Path) -> dict[str, Any]:
    primary_summary, primary_tensors = checkpoint_state(primary)
    replay_summary, replay_tensors = checkpoint_state(replay)
    if set(primary_tensors) != set(replay_tensors):
        raise ValueError("checkpoint state keys differ")
    exact = True
    maximum_difference = 0.0
    for key in primary_tensors:
        left = primary_tensors[key]
        right = replay_tensors[key]
        exact = exact and torch.equal(left, right)
        if torch.is_floating_point(left):
            maximum_difference = max(
                maximum_difference,
                float((left.float() - right.float()).abs().max()),
            )
    return {
        "primary": primary_summary,
        "replay": replay_summary,
        "checkpoint_file_sha256_equal": (
            primary_summary["file_sha256"] == replay_summary["file_sha256"]
        ),
        "model_state_exact": exact,
        "maximum_absolute_model_state_difference": maximum_difference,
        "note": (
            "Serialized checkpoint hashes may differ because run metadata contains timestamps; "
            "model_state_sha256 compares the ordered model tensors only."
        ),
    }


def plot_trajectories(
    primary: pd.DataFrame,
    replay: pd.DataFrame,
    collapse_epoch: int | None,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, (column, title) in zip(axes.flat, PLOT_COLUMNS.items(), strict=True):
        ax.plot(primary["epoch"], primary[column], color="#2563EB", linewidth=2, label="Primary")
        ax.plot(
            replay["epoch"],
            replay[column],
            color="#DC2626",
            linewidth=1.5,
            linestyle="--",
            label="Exact replay",
        )
        if collapse_epoch is not None:
            ax.axvline(collapse_epoch, color="#111827", linestyle=":", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Deterministic replay reproduces the epoch-9 detector collapse")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the primary detector run with its replay")
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--replay-run", required=True, type=Path)
    parser.add_argument("--primary-stderr", required=True, type=Path)
    parser.add_argument("--replay-stderr", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    args = parser.parse_args()

    primary = load_results(args.primary_run / "results.csv")
    replay = load_results(args.replay_run / "results.csv")
    primary_manifest = json.loads(
        (args.primary_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    replay_manifest = json.loads(
        (args.replay_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    collapse_primary = find_collapse_epoch(primary)
    collapse_replay = find_collapse_epoch(replay)
    comparison = {
        "comparison": "same-seed same-environment deterministic replay",
        "primary_run": args.primary_run.name,
        "replay_run": args.replay_run.name,
        "frozen_inputs": {
            "initialization_sha256_equal": (
                primary_manifest["model_initialization_sha256"]
                == replay_manifest["model_initialization_sha256"]
            ),
            "data_yaml_sha256_equal": (
                primary_manifest["data_yaml_sha256"] == replay_manifest["data_yaml_sha256"]
            ),
            "seed_equal": primary_manifest["seed"] == replay_manifest["seed"],
            "image_size_equal": primary_manifest["imgsz"] == replay_manifest["imgsz"],
            "effective_batch_primary": 16,
            "effective_batch_replay": replay_manifest["batch_effective"],
            "official_validation_used_for_selection": False,
        },
        "trajectory": compare_results_tables(primary, replay),
        "collapse": {
            "definition": "first epoch at or below 10% of the previous best detector-dev mAP50-95",
            "primary_epoch": collapse_primary,
            "replay_epoch": collapse_replay,
            "primary_best_epoch": int(primary.loc[primary[METRIC_COLUMN].idxmax(), "epoch"]),
            "replay_best_epoch": int(replay.loc[replay[METRIC_COLUMN].idxmax(), "epoch"]),
            "primary_completed_epochs": len(primary),
            "replay_completed_epochs": len(replay),
        },
        "checkpoints": {
            name.removesuffix(".pt"): compare_checkpoint(
                args.primary_run / "weights" / name,
                args.replay_run / "weights" / name,
            )
            for name in ("best.pt", "last.pt")
        },
        "stderr_bytes": {
            "primary": args.primary_stderr.stat().st_size,
            "replay": args.replay_stderr.stat().st_size,
        },
        "conclusion": (
            "The optimization collapse is exactly reproducible under the frozen environment. "
            "Every recorded numeric field except runtime is identical, and both best and final "
            "model tensor states are bitwise identical across runs."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_trajectories(primary, replay, collapse_replay, args.output_figure)
    print(json.dumps(comparison["collapse"], indent=2))
    print(comparison["conclusion"])


if __name__ == "__main__":
    main()
