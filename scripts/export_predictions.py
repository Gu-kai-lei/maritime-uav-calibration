from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.yolo import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Export low-threshold YOLO predictions as COCO JSON")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from ultralytics import YOLO

    dataset = load_json(args.annotations)
    categories = active_categories(dataset)
    if not categories:
        raise ValueError("no active detection categories remain after excluding ignored categories")
    class_to_category = {index: category["id"] for index, category in enumerate(categories)}
    image_records = dataset["images"]
    sources = [str((args.image_root / image["file_name"]).resolve()) for image in image_records]
    missing = [source for source in sources if not Path(source).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")

    model = YOLO(str(args.model))
    predictions = []
    processed = 0
    # A full list of paths is interpreted by Ultralytics as an in-memory image
    # collection. Explicit chunks prevent all original-resolution frames from
    # being decoded at once before 640 px preprocessing.
    for start in range(0, len(sources), args.batch):
        source_batch = sources[start : start + args.batch]
        image_batch = image_records[start : start + args.batch]
        results = model.predict(
            source=source_batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
            stream=False,
            verbose=False,
        )
        if len(results) != len(image_batch):
            raise RuntimeError(
                f"prediction batch returned {len(results)} results for {len(image_batch)} images"
            )
        for image, result in zip(image_batch, results, strict=True):
            processed += 1
            if result.boxes is None:
                continue
            xywh = result.boxes.xywh.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, class_index in zip(xywh, scores, classes, strict=True):
                if class_index not in class_to_category:
                    raise ValueError(
                        f"model class index {class_index} has no COCO category mapping; "
                        "verify the detector and dataset category schemas"
                    )
                center_x, center_y, width, height = [float(value) for value in box]
                predictions.append(
                    {
                        "image_id": image["id"],
                        "category_id": class_to_category[class_index],
                        "bbox": [center_x - width / 2, center_y - height / 2, width, height],
                        "score": float(score),
                    }
                )
    if processed != len(image_records):
        raise RuntimeError(f"processed {processed} of {len(image_records)} expected images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    manifest = {
        "images": processed,
        "predictions": len(predictions),
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "image_root": str(args.image_root.resolve()),
        "imgsz": args.imgsz,
        "confidence_floor": args.conf,
        "nms_iou": args.iou,
        "device": args.device,
        "batch": args.batch,
        "class_to_coco_category": {str(key): value for key, value in class_to_category.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
