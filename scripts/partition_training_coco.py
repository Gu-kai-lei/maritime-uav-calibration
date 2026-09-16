from __future__ import annotations

import argparse
import json
from pathlib import Path

from maritime_calibration.coco import active_categories, load_json
from maritime_calibration.splitting import partition_coco_by_group


FRACTIONS = {
    "detector_dev": 0.10,
    "calibration_fit": 0.10,
    "policy_tune": 0.10,
    "detector_train": 0.70,
}
MIN_ACTIVE_BOXES_PER_ROLE = 25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create four leakage-aware roles from the official training COCO file"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--group-field",
        action="append",
        dest="group_fields",
        default=None,
        help="Dotted image-record field; repeat to provide fallbacks",
    )
    args = parser.parse_args()

    dataset = load_json(args.annotations)
    group_fields = args.group_fields or [
        "video_id",
        "source.video",
        "source.drone",
        "source.folder_name",
    ]
    active_ids = {int(category["id"]) for category in active_categories(dataset)}
    partitions = {}
    report = {}
    for search_offset in range(256):
        candidate_seed = args.seed + search_offset
        candidate_partitions, candidate_report = partition_coco_by_group(
            dataset,
            fractions=FRACTIONS,
            group_fields=group_fields,
            seed=candidate_seed,
        )
        has_missing = False
        for details in candidate_report["partitions"].values():
            counts = {
                int(category_id): int(count)
                for category_id, count in details["category_counts"].items()
            }
            present = set(counts)
            missing = sorted(active_ids - present)
            details["missing_active_categories"] = missing
            underrepresented = {
                str(category_id): counts.get(category_id, 0)
                for category_id in sorted(active_ids)
                if counts.get(category_id, 0) < MIN_ACTIVE_BOXES_PER_ROLE
            }
            details["underrepresented_active_categories"] = underrepresented
            has_missing = has_missing or bool(missing) or bool(underrepresented)
        if not has_missing:
            partitions = candidate_partitions
            report = candidate_report
            report["base_seed"] = args.seed
            report["class_coverage_search_offset"] = search_offset
            report["minimum_active_boxes_per_role"] = MIN_ACTIVE_BOXES_PER_ROLE
            break
    if not partitions:
        raise RuntimeError(
            "could not find a four-role, whole-video partition with adequate active-class "
            f"coverage (minimum {MIN_ACTIVE_BOXES_PER_ROLE} boxes per role) "
            "after 256 deterministic candidates"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for role, partition in partitions.items():
        (args.output_dir / f"instances_{role}.json").write_text(
            json.dumps(partition, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (args.output_dir / "partition_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
