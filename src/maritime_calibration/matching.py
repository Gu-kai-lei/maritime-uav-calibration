from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .coco import ALTITUDE_ALIASES, GIMBAL_PITCH_ALIASES, metadata_value, source_group


def xywh_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2, ay2 = ax1 + max(aw, 0.0), ay1 + max(ah, 0.0)
    bx2, by2 = bx1 + max(bw, 0.0), by1 + max(bh, 0.0)
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return intersection / union if union > 0 else 0.0


def build_detection_table(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    images = {image["id"]: image for image in ground_truth["images"]}
    ground_truth_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        ground_truth_by_key[(annotation["image_id"], annotation["category_id"])].append(annotation)

    predictions_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_key[(prediction["image_id"], prediction["category_id"])].append(prediction)

    rows: list[dict[str, Any]] = []
    matched_ground_truth = 0
    for key, key_predictions in predictions_by_key.items():
        image_id, category_id = key
        if image_id not in images:
            raise ValueError(f"prediction references unknown image_id={image_id}")
        image = images[image_id]
        candidates = ground_truth_by_key.get(key, [])
        used_indices: set[int] = set()
        for prediction_index, prediction in enumerate(
            sorted(key_predictions, key=lambda record: float(record["score"]), reverse=True)
        ):
            best_index = -1
            best_iou = 0.0
            for candidate_index, annotation in enumerate(candidates):
                if candidate_index in used_indices:
                    continue
                overlap = xywh_iou(prediction["bbox"], annotation["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = candidate_index
            is_true_positive = best_index >= 0 and best_iou >= iou_threshold
            if is_true_positive:
                used_indices.add(best_index)
                matched_ground_truth += 1

            _, _, width, height = [float(value) for value in prediction["bbox"]]
            image_area = max(float(image.get("width", 0)) * float(image.get("height", 0)), 1.0)
            relative_area = max(width * height, 0.0) / image_area
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "prediction_rank": prediction_index,
                    "score": float(prediction["score"]),
                    "is_tp": int(is_true_positive),
                    "match_iou": float(best_iou),
                    "relative_area": relative_area,
                    "altitude": metadata_value(image, ALTITUDE_ALIASES),
                    "gimbal_pitch": metadata_value(image, GIMBAL_PITCH_ALIASES),
                    "source_group": source_group(
                        image,
                        ["video_id", "source.video", "source.drone", "source.folder_name"],
                    ),
                }
            )

    table = pd.DataFrame(rows)
    summary = {
        "images": len(images),
        "ground_truth": len(ground_truth["annotations"]),
        "predictions": len(predictions),
        "matched_ground_truth": matched_ground_truth,
        "false_negatives_at_export_floor": len(ground_truth["annotations"])
        - matched_ground_truth,
    }
    return table, summary
