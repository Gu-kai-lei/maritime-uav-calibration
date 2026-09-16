from __future__ import annotations

import argparse
import json
from pathlib import Path

from maritime_calibration.coco import load_json
from maritime_calibration.matching import build_detection_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Match COCO detections and create calibration rows")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    ground_truth = load_json(args.annotations)
    predictions = load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError("predictions JSON must be a COCO-style list")
    table, summary = build_detection_table(ground_truth, predictions, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
