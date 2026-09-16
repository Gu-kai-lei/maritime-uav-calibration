from __future__ import annotations

import argparse
import json
from pathlib import Path

from maritime_calibration.coco import load_json
from maritime_calibration.splitting import split_coco_by_group


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware detector/calibration splits")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--group-field",
        action="append",
        dest="group_fields",
        default=None,
        help="Dotted image-record field; repeat to provide fallbacks",
    )
    args = parser.parse_args()

    group_fields = args.group_fields or [
        "video_id",
        "source.video",
        "source.drone",
        "source.folder_name",
    ]
    detector, calibration, report = split_coco_by_group(
        load_json(args.annotations),
        calibration_fraction=args.calibration_fraction,
        group_fields=group_fields,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "instances_detector_train.json").write_text(
        json.dumps(detector, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "instances_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
