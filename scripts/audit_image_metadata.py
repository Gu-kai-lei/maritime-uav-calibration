from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from maritime_calibration.coco import load_json


def flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten(value, dotted))
        else:
            output[dotted] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Report every image-level COCO metadata field")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    images = dataset["images"]
    coverage: Counter[str] = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, Any] = {}

    for image in images:
        for key, value in flatten(image).items():
            if value is None or value == "":
                continue
            coverage[key] += 1
            types[key][type(value).__name__] += 1
            examples.setdefault(key, value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    numeric_values[key].append(numeric)

    fields = []
    for key, count in coverage.most_common():
        entry: dict[str, Any] = {
            "field": key,
            "count": count,
            "coverage": count / max(len(images), 1),
            "types": dict(types[key]),
            "example": examples[key],
        }
        values = numeric_values.get(key)
        if values:
            array = np.asarray(values, dtype=float)
            entry["numeric"] = {
                "min": float(array.min()),
                "p25": float(np.quantile(array, 0.25)),
                "median": float(np.median(array)),
                "p75": float(np.quantile(array, 0.75)),
                "max": float(array.max()),
            }
        fields.append(entry)

    report = {"images": len(images), "fields": fields}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
