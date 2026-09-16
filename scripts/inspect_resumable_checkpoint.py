from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_detector_replays import checkpoint_state


def tensor_finiteness(value: Any) -> tuple[int, int]:
    """Return floating-value and non-finite counts for a nested checkpoint value."""
    if torch.is_tensor(value):
        if not torch.is_floating_point(value):
            return 0, 0
        tensor = value.detach().float().cpu()
        return tensor.numel(), int((~torch.isfinite(tensor)).sum())
    if isinstance(value, dict):
        counts = [tensor_finiteness(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        counts = [tensor_finiteness(item) for item in value]
    else:
        counts = []
    return sum(item[0] for item in counts), sum(item[1] for item in counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether a YOLO checkpoint is safe to resume")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-stripped",
        action="store_true",
        help="Require finite model tensors but allow a completed optimizer-stripped checkpoint",
    )
    args = parser.parse_args()

    model_summary, _ = checkpoint_state(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    optimizer = checkpoint.get("optimizer")
    optimizer_values, optimizer_nonfinite = tensor_finiteness(optimizer)
    scaler_values, scaler_nonfinite = tensor_finiteness(checkpoint.get("scaler"))
    train_args = checkpoint.get("train_args") or {}
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch_zero_based": epoch,
        "completed_epochs": epoch + 1 if epoch >= 0 else 0,
        "optimizer_present": optimizer is not None,
        "optimizer_floating_values": optimizer_values,
        "optimizer_nonfinite_values": optimizer_nonfinite,
        "scaler_floating_values": scaler_values,
        "scaler_nonfinite_values": scaler_nonfinite,
        "model": model_summary,
        "frozen_args": {
            key: train_args.get(key)
            for key in (
                "data",
                "model",
                "imgsz",
                "epochs",
                "batch",
                "device",
                "workers",
                "seed",
                "deterministic",
                "patience",
                "amp",
                "optimizer",
                "lr0",
                "nbs",
                "fraction",
                "project",
                "name",
            )
        },
        "resumable": (
            epoch >= 0
            and optimizer is not None
            and model_summary["nonfinite_values"] == 0
            and optimizer_nonfinite == 0
            and scaler_nonfinite == 0
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    finite_model = model_summary["nonfinite_values"] == 0
    if not report["resumable"] and not (args.allow_stripped and finite_model):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
