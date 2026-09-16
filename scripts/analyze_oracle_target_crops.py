from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.crops import target_context_crop, xywh_to_xyxy
from maritime_calibration.matching import xywh_iou
from maritime_calibration.yolo import file_sha256


def crop_relative_xywh(annotation: dict[str, Any], crop: tuple[int, int, int, int]) -> list[float]:
    left, top, right, bottom = crop
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    if x < left or y < top or x + width > right or y + height > bottom:
        raise ValueError("annotation is not fully contained in the crop")
    return [x - left, y - top, width, height]


def greedy_same_class_matches(
    annotations: list[dict[str, Any]], predictions: list[dict[str, Any]], iou: float
) -> set[Any]:
    matched: set[Any] = set()
    used: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: float(item["score"]), reverse=True):
        best_index = -1
        best_overlap = 0.0
        for index, annotation in enumerate(annotations):
            if index in used:
                continue
            overlap = xywh_iou(annotation["bbox"], prediction["bbox"])
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index >= 0 and best_overlap >= iou:
            used.add(best_index)
            matched.add(annotations[best_index]["id"])
    return matched


def best_iou(
    annotation: dict[str, Any], predictions: list[dict[str, Any]]
) -> tuple[float, dict[str, Any] | None]:
    best_overlap = 0.0
    best_prediction = None
    for prediction in predictions:
        overlap = xywh_iou(annotation["bbox"], prediction["bbox"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_prediction = prediction
    return best_overlap, best_prediction


def summarize(details: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, context), group in details.groupby(["model", "context_factor"], sort=False):
        rows.append(
            {
                "model": model,
                "context_factor": float(context),
                "target_ground_truth": len(group),
                "same_class_matched_iou_0_50": int(group["same_class_matched_iou_0_50"].sum()),
                "same_class_recall_iou_0_50": float(group["same_class_matched_iou_0_50"].mean()),
                "same_class_localized_iou_0_10": int((group["best_same_iou"] >= 0.10).sum()),
                "same_class_localized_iou_0_30": int((group["best_same_iou"] >= 0.30).sum()),
                "same_class_localized_iou_0_50": int((group["best_same_iou"] >= 0.50).sum()),
                "any_class_localized_iou_0_10": int((group["best_any_iou"] >= 0.10).sum()),
                "any_class_localized_iou_0_30": int((group["best_any_iou"] >= 0.30).sum()),
                "any_class_localized_iou_0_50": int((group["best_any_iou"] >= 0.50).sum()),
                "maximum_best_same_iou": float(group["best_same_iou"].max()),
                "mean_best_same_iou": float(group["best_same_iou"].mean()),
                "maximum_best_any_iou": float(group["best_any_iou"].max()),
                "mean_best_any_iou": float(group["best_any_iou"].mean()),
                "target_predictions": int(group["target_predictions_in_crop"].sum()),
                "target_prediction_max_score": (
                    float(group["target_prediction_max_score"].max())
                    if group["target_prediction_max_score"].notna().any()
                    else None
                ),
                "median_target_width_at_1280": float(group["target_width_at_1280"].median()),
                "median_target_height_at_1280": float(group["target_height_at_1280"].median()),
                "best_any_category_counts": {
                    str(key): int(value)
                    for key, value in group["best_any_category"].dropna().value_counts().items()
                },
                "target_category": target,
            }
        )
    return pd.DataFrame(rows)


def json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        records.append(
            {
                key: (
                    None
                    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
                    else value.item() if isinstance(value, np.generic) else value
                )
                for key, value in row.items()
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen oracle target-crop diagnostic")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument(
        "--model", action="append", nargs=2, metavar=("NAME", "CHECKPOINT"), required=True
    )
    parser.add_argument("--target", default="life_saving_appliances")
    parser.add_argument("--context", action="append", type=float, default=[])
    parser.add_argument("--minimum-side", type=int, default=384)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    contexts = tuple(sorted(set(args.context or [8.0, 12.0, 16.0])))
    if contexts != (8.0, 12.0, 16.0):
        raise ValueError("the frozen oracle protocol requires contexts 8, 12, and 16")
    dataset = load_json(args.annotations)
    categories = active_categories(dataset)
    category_to_index = {int(item["id"]): index for index, item in enumerate(categories)}
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    target_ids = [category_id for category_id, name in category_names.items() if name == args.target]
    if len(target_ids) != 1:
        raise ValueError(f"expected one active category named {args.target}")
    target_id = target_ids[0]
    if target_id not in category_to_index:
        raise ValueError("target category is not mapped to a model class")
    target_class_index = category_to_index[target_id]

    images = {item["id"]: item for item in dataset["images"]}
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) in category_to_index:
            annotations_by_image[annotation["image_id"]].append(annotation)
    target_images = sorted(
        image_id
        for image_id, annotations in annotations_by_image.items()
        if any(int(item["category_id"]) == target_id for item in annotations)
    )
    if not target_images:
        raise ValueError("detector-dev has no target-positive images")

    crop_records: list[dict[str, Any]] = []
    for context in contexts:
        for image_id in target_images:
            image = images[image_id]
            annotations = annotations_by_image[image_id]
            target_annotations = [item for item in annotations if int(item["category_id"]) == target_id]
            target_boxes = [xywh_to_xyxy(item["bbox"]) for item in target_annotations]
            active_boxes = [xywh_to_xyxy(item["bbox"]) for item in annotations]
            crop = target_context_crop(
                int(image["width"]), int(image["height"]), target_boxes, active_boxes,
                context, args.minimum_side
            )
            crop_records.append(
                {
                    "context_factor": context,
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "crop": crop,
                    "annotations": target_annotations,
                }
            )

    from ultralytics import YOLO

    detail_rows: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}
    for model_name, checkpoint_text in args.model:
        checkpoint = Path(checkpoint_text)
        if model_name in model_hashes:
            raise ValueError(f"duplicate model name: {model_name}")
        model_hashes[model_name] = file_sha256(checkpoint)
        model = YOLO(str(checkpoint.resolve()))
        for start in range(0, len(crop_records), args.batch):
            batch_records = crop_records[start : start + args.batch]
            pil_images: list[Image.Image] = []
            for record in batch_records:
                source = args.image_root / record["file_name"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                with Image.open(source) as image:
                    if image.size != (images[record["image_id"]]["width"], images[record["image_id"]]["height"]):
                        raise ValueError(f"image dimensions differ from COCO metadata: {source}")
                    pil_images.append(image.convert("RGB").crop(record["crop"]))
            results = model.predict(
                source=pil_images,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=args.batch,
                half=False,
                augment=False,
                save=False,
                verbose=False,
            )
            if len(results) != len(batch_records):
                raise RuntimeError("prediction batch size mismatch")
            for record, result in zip(batch_records, results, strict=True):
                predictions: list[dict[str, Any]] = []
                if result.boxes is not None:
                    boxes = result.boxes.xywh.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    for box, score, class_index in zip(boxes, scores, classes, strict=True):
                        if class_index < 0 or class_index >= len(categories):
                            raise ValueError(f"model class index outside active category mapping: {class_index}")
                        predictions.append(
                            {
                                "bbox": [
                                    float(box[0] - box[2] / 2), float(box[1] - box[3] / 2),
                                    float(box[2]), float(box[3])
                                ],
                                "score": float(score),
                                "category_id": int(categories[class_index]["id"]),
                            }
                        )
                target_predictions = [
                    item for item in predictions if int(item["category_id"]) == target_id
                ]
                crop_annotations = [
                    {"id": item["id"], "bbox": crop_relative_xywh(item, record["crop"])}
                    for item in record["annotations"]
                ]
                matched = greedy_same_class_matches(crop_annotations, target_predictions, args.match_iou)
                left, top, right, bottom = record["crop"]
                scale = min(args.imgsz / (right - left), args.imgsz / (bottom - top))
                max_target_score = (
                    max(float(item["score"]) for item in target_predictions)
                    if target_predictions else None
                )
                for source_annotation, crop_annotation in zip(
                    record["annotations"], crop_annotations, strict=True
                ):
                    same_iou, same_prediction = best_iou(crop_annotation, target_predictions)
                    any_iou, any_prediction = best_iou(crop_annotation, predictions)
                    _, _, target_width, target_height = crop_annotation["bbox"]
                    detail_rows.append(
                        {
                            "model": model_name,
                            "context_factor": record["context_factor"],
                            "image_id": record["image_id"],
                            "annotation_id": source_annotation["id"],
                            "crop_left": left,
                            "crop_top": top,
                            "crop_right": right,
                            "crop_bottom": bottom,
                            "crop_width": right - left,
                            "crop_height": bottom - top,
                            "target_width_at_1280": float(target_width * scale),
                            "target_height_at_1280": float(target_height * scale),
                            "target_predictions_in_crop": len(target_predictions),
                            "target_prediction_max_score": max_target_score,
                            "same_class_matched_iou_0_50": source_annotation["id"] in matched,
                            "best_same_iou": same_iou,
                            "best_same_score": (
                                float(same_prediction["score"]) if same_prediction else None
                            ),
                            "best_any_iou": any_iou,
                            "best_any_score": (
                                float(any_prediction["score"]) if any_prediction else None
                            ),
                            "best_any_category": (
                                category_names[int(any_prediction["category_id"])]
                                if any_prediction else None
                            ),
                        }
                    )
            for image in pil_images:
                image.close()

    details = pd.DataFrame(detail_rows)
    summary = summarize(details, args.target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "oracle_target_crop_per_annotation.csv", index=False)
    summary.to_csv(args.output_dir / "oracle_target_crop_summary.csv", index=False)

    crop4 = summary[summary["model"] == "crop4_1280"]
    representation_supported = bool((crop4["same_class_recall_iou_0_50"] > 0).any())
    report = {
        "analysis": "frozen detector-dev oracle target-crop diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_validation_used": False,
        "training_run": False,
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "image_root": str(args.image_root.resolve()),
        "target_category": args.target,
        "target_ground_truth_per_context": int(len(details) / len(model_hashes) / len(contexts)),
        "target_positive_images": len(target_images),
        "contexts": list(contexts),
        "minimum_native_crop_side": args.minimum_side,
        "inference": {
            "imgsz": args.imgsz,
            "confidence_floor": args.conf,
            "nms_iou": args.iou,
            "match_iou": args.match_iou,
            "batch": args.batch,
            "half": False,
            "augment": False,
        },
        "model_sha256": model_hashes,
        "summary": json_records(summary),
        "decision": {
            "crop4_representation_supported_by_positive_oracle_recall": representation_supported,
            "interpretation": (
                "Crop4 contains a recognizable target representation when target location is supplied; "
                "the full-frame failure is principally proposal/scale related."
                if representation_supported
                else "Crop4 still has zero same-class recall when target location and scale are supplied; "
                "the evidence favors target-class representation/domain mismatch over a pure proposal failure."
            ),
        },
        "claim_boundary": (
            "Oracle crops use detector-dev ground-truth locations and cannot be deployed. All contexts "
            "are reported without selecting a checkpoint or threshold; calibration-fit, policy-tune, "
            "and official validation were not used."
        ),
    }
    (args.output_dir / "oracle_target_crop_diagnostic.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    colors = {"repeat4_1280": "#4C78A8", "crop4_1280": "#F58518"}
    for model_name, model_rows in summary.groupby("model", sort=False):
        model_rows = model_rows.sort_values("context_factor")
        color = colors.get(model_name)
        axes[0].plot(
            model_rows["context_factor"], model_rows["same_class_recall_iou_0_50"],
            marker="o", label=model_name, color=color
        )
        axes[1].plot(
            model_rows["context_factor"], model_rows["any_class_localized_iou_0_50"],
            marker="o", label=model_name, color=color
        )
        axes[2].plot(
            model_rows["context_factor"], model_rows["median_target_width_at_1280"],
            marker="o", label=model_name, color=color
        )
    axes[0].set_ylabel("Same-class recall at IoU >= 0.50")
    axes[0].set_title("Oracle target recognition")
    axes[1].set_ylabel("Target GT localized by any class")
    axes[1].set_title("Oracle any-class localization")
    axes[2].set_ylabel("Median target width after resize (px)")
    axes[2].set_title("Effective target scale")
    for axis in axes:
        axis.set_xlabel("Context factor")
        axis.set_xticks(contexts)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle("Detector-dev oracle target-crop diagnostic")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "oracle_target_crop_diagnostic.png", dpi=200, bbox_inches="tight"
    )
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
