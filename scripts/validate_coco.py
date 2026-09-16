from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maritime_calibration.coco import audit_coco, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit COCO annotations and metadata coverage")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-metadata", action="store_true")
    args = parser.parse_args()

    report = audit_coco(load_json(args.annotations))
    if args.require_metadata:
        missing = [name for name, count in report["metadata_coverage"].items() if count == 0]
        if missing:
            report["errors"].append(f"required metadata absent: {', '.join(missing)}")
            report["valid"] = False

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
