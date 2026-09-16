from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.yolo import file_sha256


def class_metric_rows(
    names: dict[int, str],
    class_indices: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    ap50: np.ndarray,
    ap50_95: np.ndarray,
    instances: np.ndarray | None,
) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for position, class_index_value in enumerate(class_indices):
        class_index = int(class_index_value)
        row: dict[str, float | int] = {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "map50": float(ap50[position]),
            "map50_95": float(ap50_95[position]),
        }
        if instances is not None and class_index < len(instances):
            row["instances"] = int(instances[class_index])
        rows[names[class_index]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector at one frozen input resolution")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    expected_dir = args.project.resolve() / args.name
    if expected_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {expected_dir}")

    import torch
    import ultralytics
    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(str(args.model.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        workers=args.workers,
        half=False,
        augment=False,
        plots=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        verbose=True,
    )
    save_dir = Path(metrics.save_dir)
    box = metrics.box
    instance_counts = getattr(box, "nt_per_class", None)
    if instance_counts is not None:
        instance_counts = np.asarray(instance_counts)
    per_class = class_metric_rows(
        {int(key): str(value) for key, value in metrics.names.items()},
        np.asarray(box.ap_class_index),
        np.asarray(box.p),
        np.asarray(box.r),
        np.asarray(box.ap50),
        np.asarray(box.ap),
        instance_counts,
    )
    manifest: dict[str, Any] = {
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_dir": str(save_dir.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": file_sha256(args.data),
        "evaluation_split": "detector-dev",
        "official_validation_used": False,
        "image_size": args.imgsz,
        "batch": args.batch,
        "confidence_floor": args.conf,
        "nms_iou": args.iou,
        "max_detections": args.max_det,
        "half_precision": False,
        "test_time_augmentation": False,
        "aggregate": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (save_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
