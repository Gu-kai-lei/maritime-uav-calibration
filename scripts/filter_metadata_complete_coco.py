from __future__ import annotations

import argparse
import json
from pathlib import Path

from maritime_calibration.coco import load_json, metadata_complete_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a COCO view containing only altitude-and-gimbal-complete images"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    subset, report = metadata_complete_subset(load_json(args.annotations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
