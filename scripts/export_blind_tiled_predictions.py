from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def translate_xyxy(
    boxes: np.ndarray, left: int, top: int, image_width: int, image_height: int
) -> np.ndarray:
    translated = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if translated.size == 0:
        return translated
    translated[:, [0, 2]] += float(left)
    translated[:, [1, 3]] += float(top)
    translated[:, [0, 2]] = translated[:, [0, 2]].clip(0.0, float(image_width))
    translated[:, [1, 3]] = translated[:, [1, 3]].clip(0.0, float(image_height))
    return translated


def class_aware_nms_indices(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float
) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        torch.as_tensor(classes, dtype=torch.int64),
        float(iou),
    )
    return keep.cpu().numpy().astype(np.int64)


def top_score_indices(scores: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    order = np.argsort(-np.asarray(scores), kind="stable")
    return order[:maximum].astype(np.int64)


def xyxy_to_coco(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind tiled YOLO predictions as COCO JSON")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--tile-size", required=True, type=int, choices=(384, 768))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--tile-iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {args.output}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    from ultralytics import YOLO
    import torch

    dataset = load_json(args.annotations)
    categories = active_categories(dataset)
    class_to_category = {index: int(category["id"]) for index, category in enumerate(categories)}
    target_matches = [int(item["id"]) for item in categories if item["name"] == args.target]
    if len(target_matches) != 1:
        raise ValueError(f"expected one active target category named {args.target}")
    target_category_id = target_matches[0]
    image_records = dataset["images"][: args.limit]
    if not image_records:
        raise ValueError("no images selected")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    target_output = args.output.with_name(f"{args.output.stem}.uncapped_target.json")
    started_at = utc_now()
    model = YOLO(str(args.model.resolve()))
    cuda_device = str(args.device).lower() not in {"cpu", "none"} and torch.cuda.is_available()
    if cuda_device:
        torch.cuda.reset_peak_memory_stats()

    capped_predictions: list[dict[str, Any]] = []
    uncapped_target_predictions: list[dict[str, Any]] = []
    per_image_latency_ms: list[float] = []
    total_tiles = 0
    raw_predictions_count = 0
    merged_predictions_count = 0
    started_wall = time.perf_counter()

    for image_index, image_record in enumerate(image_records, start=1):
        source = args.image_root / image_record["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if cuda_device:
            torch.cuda.synchronize()
        image_started = time.perf_counter()
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if (width, height) != (int(image_record["width"]), int(image_record["height"])):
                raise ValueError(f"image dimensions differ from COCO metadata: {source}")
            positions = [
                (left, top)
                for top in tile_starts(height, args.tile_size, args.overlap)
                for left in tile_starts(width, args.tile_size, args.overlap)
            ]
            total_tiles += len(positions)
            for start in range(0, len(positions), args.batch):
                batch_positions = positions[start : start + args.batch]
                crops = [
                    image.crop(
                        (
                            left,
                            top,
                            min(left + args.tile_size, width),
                            min(top + args.tile_size, height),
                        )
                    )
                    for left, top in batch_positions
                ]
                results = model.predict(
                    source=crops,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.tile_iou,
                    max_det=args.max_det,
                    device=args.device,
                    batch=args.batch,
                    half=False,
                    augment=False,
                    save=False,
                    verbose=False,
                )
                if len(results) != len(batch_positions):
                    raise RuntimeError("prediction batch size mismatch")
                for (left, top), result in zip(batch_positions, results, strict=True):
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes = translate_xyxy(
                        result.boxes.xyxy.cpu().numpy(), left, top, width, height
                    )
                    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
                    if any(int(value) not in class_to_category for value in classes):
                        raise ValueError("model class index falls outside the active category mapping")
                    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    all_boxes.append(boxes[valid])
                    all_scores.append(scores[valid])
                    all_classes.append(classes[valid])

        if all_boxes:
            boxes_array = np.concatenate(all_boxes)
            scores_array = np.concatenate(all_scores)
            classes_array = np.concatenate(all_classes)
        else:
            boxes_array = np.empty((0, 4), dtype=np.float32)
            scores_array = np.empty(0, dtype=np.float32)
            classes_array = np.empty(0, dtype=np.int64)
        raw_predictions_count += len(boxes_array)
        kept = class_aware_nms_indices(
            boxes_array, scores_array, classes_array, args.merge_iou
        )
        boxes_array, scores_array, classes_array = (
            boxes_array[kept], scores_array[kept], classes_array[kept]
        )
        merged_predictions_count += len(boxes_array)

        target_indices = np.flatnonzero(
            np.asarray([class_to_category[int(value)] for value in classes_array])
            == target_category_id
        )
        for index in target_indices:
            uncapped_target_predictions.append(
                {
                    "image_id": image_record["id"],
                    "category_id": target_category_id,
                    "bbox": xyxy_to_coco(boxes_array[index]),
                    "score": float(scores_array[index]),
                }
            )
        capped = top_score_indices(scores_array, args.max_det)
        for index in capped:
            capped_predictions.append(
                {
                    "image_id": image_record["id"],
                    "category_id": class_to_category[int(classes_array[index])],
                    "bbox": xyxy_to_coco(boxes_array[index]),
                    "score": float(scores_array[index]),
                }
            )
        if cuda_device:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - image_started) * 1000.0
        per_image_latency_ms.append(latency_ms)

        elapsed = time.perf_counter() - started_wall
        if image_index == 1 or image_index % 10 == 0 or image_index == len(image_records):
            rate = elapsed / image_index
            eta = rate * (len(image_records) - image_index)
            progress = {
                "status": "running" if image_index < len(image_records) else "finalizing",
                "processed_images": image_index,
                "selected_images": len(image_records),
                "tiles": total_tiles,
                "raw_predictions": raw_predictions_count,
                "post_merge_predictions": merged_predictions_count,
                "capped_predictions": len(capped_predictions),
                "uncapped_target_predictions": len(uncapped_target_predictions),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "updated_at_utc": utc_now(),
            }
            progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(progress), flush=True)

    completed_at = utc_now()
    args.output.write_text(json.dumps(capped_predictions) + "\n", encoding="utf-8")
    target_output.write_text(json.dumps(uncapped_target_predictions) + "\n", encoding="utf-8")
    wall_seconds = time.perf_counter() - started_wall
    manifest = {
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "images": len(image_records),
        "dataset_images": len(dataset["images"]),
        "limit": args.limit,
        "performance_claims_allowed": args.limit is None,
        "tiles": total_tiles,
        "mean_tiles_per_image": total_tiles / len(image_records),
        "raw_predictions": raw_predictions_count,
        "post_merge_predictions": merged_predictions_count,
        "capped_predictions": len(capped_predictions),
        "uncapped_target_predictions": len(uncapped_target_predictions),
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "image_root": str(args.image_root.resolve()),
        "target_category": args.target,
        "target_category_id": target_category_id,
        "tile_size": args.tile_size,
        "overlap_fraction": args.overlap,
        "imgsz": args.imgsz,
        "confidence_floor": args.conf,
        "within_tile_nms_iou": args.tile_iou,
        "cross_tile_class_aware_nms_iou": args.merge_iou,
        "max_detections_per_tile": args.max_det,
        "max_detections_per_image": args.max_det,
        "half": False,
        "augment": False,
        "device": args.device,
        "batch": args.batch,
        "official_validation_used": False,
        "oracle_locations_used": False,
        "wall_seconds": wall_seconds,
        "latency_ms_per_image": {
            "mean": float(np.mean(per_image_latency_ms)),
            "p50": percentile(per_image_latency_ms, 50),
            "p95": percentile(per_image_latency_ms, 95),
            "minimum": float(np.min(per_image_latency_ms)),
            "maximum": float(np.max(per_image_latency_ms)),
        },
        "cuda_peak_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 2**20) if cuda_device else None
        ),
        "cuda_peak_reserved_mib": (
            float(torch.cuda.max_memory_reserved() / 2**20) if cuda_device else None
        ),
        "class_to_coco_category": {str(key): value for key, value in class_to_category.items()},
        "primary_predictions": str(args.output.resolve()),
        "uncapped_target_predictions_path": str(target_output.resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    progress_path.write_text(
        json.dumps({"status": "completed", "manifest": str(args.output.with_suffix('.manifest.json').resolve())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
