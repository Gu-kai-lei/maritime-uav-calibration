from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.yolo import file_sha256


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def best_epoch_summary(results_csv: Path) -> dict[str, float | int] | None:
    if not results_csv.is_file():
        return None
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        return None
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best[metric]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Ultralytics detector baseline")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="sds_v2_yolov8n_640_seed20260803")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete Ultralytics run from --model checkpoint state",
    )
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    batch: int | float = float(args.batch) if "." in args.batch else int(args.batch)
    project_dir = (
        args.project.resolve()
        if args.project.is_absolute()
        else (REPOSITORY_ROOT / args.project).resolve()
    )
    requested_model = Path(args.model)
    repository_model = REPOSITORY_ROOT / requested_model
    model_source = (
        str(repository_model.resolve())
        if not requested_model.is_absolute() and repository_model.is_file()
        else args.model
    )
    started_at = datetime.now(timezone.utc).isoformat()
    resume_checkpoint_sha256 = (
        file_sha256(Path(model_source)) if args.resume and Path(model_source).is_file() else None
    )
    resume_from_epoch = None
    model = YOLO(model_source)
    if args.resume:
        checkpoint = getattr(model, "ckpt", None) or {}
        checkpoint_epoch = checkpoint.get("epoch")
        resume_from_epoch = (
            int(checkpoint_epoch) + 1 if checkpoint_epoch is not None else None
        )
        results = model.train(
            resume=True,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            plots=True,
        )
    else:
        results = model.train(
            data=str(args.data),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=batch,
            device=args.device,
            workers=args.workers,
            project=str(project_dir),
            name=args.name,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            amp=args.amp,
            optimizer=args.optimizer,
            lr0=args.lr0,
            nbs=args.nbs,
            fraction=args.fraction,
            plots=True,
        )
    best_weights = Path(results.save_dir, "weights", "best.pt")
    trainer = getattr(model, "trainer", None)
    completed_epoch_index = getattr(trainer, "epoch", None)
    effective_batch = getattr(trainer, "batch_size", None)
    effective_args = getattr(trainer, "args", None)

    def effective(name: str, fallback: object) -> object:
        return getattr(effective_args, name, fallback) if effective_args is not None else fallback

    output = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_dir": str(results.save_dir),
        "model": args.model,
        "resolved_model_source": model_source,
        "model_initialization_sha256": (
            file_sha256(Path(model_source))
            if not args.resume and Path(model_source).is_file()
            else None
        ),
        "resumed": args.resume,
        "resume_checkpoint_sha256_before": resume_checkpoint_sha256,
        "resume_from_completed_epochs": resume_from_epoch,
        "best_weights": str(best_weights.resolve()) if best_weights.is_file() else None,
        "best_weights_sha256": file_sha256(best_weights) if best_weights.is_file() else None,
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": file_sha256(args.data),
        "imgsz": effective("imgsz", args.imgsz),
        "epochs": effective("epochs", args.epochs),
        "batch": effective("batch", batch),
        "epochs_requested": effective("epochs", args.epochs),
        "epochs_completed": (
            int(completed_epoch_index) + 1 if completed_epoch_index is not None else None
        ),
        "best_epoch_metrics": best_epoch_summary(Path(results.save_dir, "results.csv")),
        "batch_requested": effective("batch", batch),
        "batch_effective": int(effective_batch) if effective_batch is not None else None,
        "device": args.device,
        "workers": effective("workers", args.workers),
        "seed": effective("seed", args.seed),
        "patience": effective("patience", args.patience),
        "amp": effective("amp", args.amp),
        "optimizer": effective("optimizer", args.optimizer),
        "lr0": effective("lr0", args.lr0),
        "nbs": effective("nbs", args.nbs),
        "fraction": effective("fraction", args.fraction),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
    }
    Path(results.save_dir, "run_manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
