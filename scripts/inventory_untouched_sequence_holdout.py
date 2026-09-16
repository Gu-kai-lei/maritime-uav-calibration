from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maritime_calibration.coco import load_json, source_group
from maritime_calibration.yolo import file_sha256


GROUP_FIELDS = ["video_id", "source.video", "source.drone", "source.folder_name"]


def image_group(image: dict[str, Any]) -> str:
    group = source_group(image, GROUP_FIELDS)
    if group is None:
        raise ValueError(f"image_id={image.get('id')} has no source group")
    return group


def group_counts(dataset: dict[str, Any]) -> Counter[str]:
    return Counter(image_group(image) for image in dataset["images"])


def dataset_identity(dataset: dict[str, Any]) -> dict[str, set[Any]]:
    image_ids = [image["id"] for image in dataset["images"]]
    file_names = [str(image["file_name"]) for image in dataset["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("dataset contains duplicate image IDs")
    if len(file_names) != len(set(file_names)):
        raise ValueError("dataset contains duplicate image file names")
    return {
        "image_ids": set(image_ids),
        "file_names": set(file_names),
        "groups": set(group_counts(dataset)),
    }


def analyze_role_coverage(
    full_dataset: dict[str, Any], role_datasets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    full = dataset_identity(full_dataset)
    role_details: dict[str, Any] = {}
    seen_ids: set[Any] = set()
    seen_files: set[str] = set()
    seen_groups: set[str] = set()
    overlaps: list[dict[str, Any]] = []
    for role, dataset in role_datasets.items():
        identity = dataset_identity(dataset)
        unknown_ids = identity["image_ids"] - full["image_ids"]
        unknown_files = identity["file_names"] - full["file_names"]
        if unknown_ids or unknown_files:
            raise ValueError(f"role {role} is not a subset of the official training dataset")
        overlaps.append(
            {
                "role": role,
                "image_id_overlap_with_prior_roles": len(identity["image_ids"] & seen_ids),
                "file_name_overlap_with_prior_roles": len(identity["file_names"] & seen_files),
                "source_group_overlap_with_prior_roles": len(identity["groups"] & seen_groups),
            }
        )
        counts = group_counts(dataset)
        role_details[role] = {
            "images": len(identity["image_ids"]),
            "source_groups": len(identity["groups"]),
            "groups": [
                {"source_group": group, "images": int(counts[group])}
                for group in sorted(counts)
            ],
        }
        seen_ids.update(identity["image_ids"])
        seen_files.update(identity["file_names"])
        seen_groups.update(identity["groups"])
    unused_ids = full["image_ids"] - seen_ids
    unused_files = full["file_names"] - seen_files
    unused_groups = full["groups"] - seen_groups
    return {
        "official_train_images": len(full["image_ids"]),
        "official_train_source_groups": len(full["groups"]),
        "roles": role_details,
        "pairwise_accumulated_overlap_checks": overlaps,
        "role_union_images": len(seen_ids),
        "role_union_file_names": len(seen_files),
        "role_union_source_groups": len(seen_groups),
        "unused_image_ids": sorted(unused_ids),
        "unused_file_names": sorted(unused_files),
        "unused_source_groups": sorted(unused_groups),
        "complete_image_coverage": seen_ids == full["image_ids"],
        "complete_file_name_coverage": seen_files == full["file_names"],
        "complete_source_group_coverage": seen_groups == full["groups"],
    }


def sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_image_inventory(
    image_root: Path, dataset: dict[str, Any], download_manifest: Path | None
) -> dict[str, Any]:
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    expected_names = {str(image["file_name"]) for image in dataset["images"]}
    local_files = {path.name: path.stat().st_size for path in image_root.iterdir() if path.is_file()}
    expected_manifest: dict[str, dict[str, Any]] = {}
    manifest_sha256 = None
    server_inventory_sha256 = None
    if download_manifest is not None:
        manifest = load_json(download_manifest)
        expected_manifest = {str(item["name"]): item for item in manifest["files"]}
        manifest_sha256 = file_sha256(download_manifest)
        server_inventory_sha256 = sha256_lines(
            [
                f"{name}\t{int(item['bytes'])}\t{item.get('etag', '')}"
                for name, item in expected_manifest.items()
            ]
        )
    size_mismatches = [
        {
            "file_name": name,
            "local_bytes": int(local_files[name]),
            "manifest_bytes": int(expected_manifest[name]["bytes"]),
        }
        for name in sorted(expected_names & set(local_files) & set(expected_manifest))
        if int(local_files[name]) != int(expected_manifest[name]["bytes"])
    ]
    local_inventory_sha256 = sha256_lines(
        [f"{name}\t{size}" for name, size in local_files.items()]
    )
    return {
        "image_root": str(image_root.resolve()),
        "annotation_images": len(expected_names),
        "local_files": len(local_files),
        "missing_annotation_images": sorted(expected_names - set(local_files)),
        "unexpected_local_files": sorted(set(local_files) - expected_names),
        "download_manifest": str(download_manifest.resolve()) if download_manifest else None,
        "download_manifest_sha256": manifest_sha256,
        "manifest_missing_annotation_images": sorted(expected_names - set(expected_manifest)),
        "manifest_unexpected_files": sorted(set(expected_manifest) - expected_names),
        "size_mismatches": size_mismatches,
        "local_filename_size_inventory_sha256": local_inventory_sha256,
        "server_name_size_etag_inventory_sha256": server_inventory_sha256,
        "content_sha256_per_image_performed": False,
        "complete": (
            expected_names == set(local_files)
            and (not expected_manifest or expected_names == set(expected_manifest))
            and not size_mismatches
        ),
    }


def protected_inventory(
    protected_dataset: dict[str, Any], train_dataset: dict[str, Any]
) -> dict[str, Any]:
    protected_counts = group_counts(protected_dataset)
    train_groups = set(group_counts(train_dataset))
    protected_groups = set(protected_counts)
    return {
        "images": len(protected_dataset["images"]),
        "source_groups": len(protected_groups),
        "overlap_with_official_train_source_groups": len(protected_groups & train_groups),
        "novel_source_groups_relative_to_official_train": sorted(protected_groups - train_groups),
        "groups": [
            {
                "source_group": group,
                "images": int(protected_counts[group]),
                "overlaps_official_train": group in train_groups,
                "eligible_as_development_holdout": False,
            }
            for group in sorted(protected_counts)
        ],
        "labels_or_model_metrics_summarized": False,
        "policy": "protected official validation; never a development or selection role",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory untouched source-sequence holdout availability without evaluation"
    )
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--train-image-root", required=True, type=Path)
    parser.add_argument("--train-download-manifest", type=Path)
    parser.add_argument(
        "--role", action="append", nargs=2, metavar=("NAME", "ANNOTATIONS"), required=True
    )
    parser.add_argument(
        "--protected",
        nargs=4,
        metavar=("NAME", "ANNOTATIONS", "IMAGE_ROOT", "DOWNLOAD_MANIFEST"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.train_annotations.is_file():
        raise FileNotFoundError(args.train_annotations)
    if args.train_download_manifest and not args.train_download_manifest.is_file():
        raise FileNotFoundError(args.train_download_manifest)
    role_paths: dict[str, Path] = {}
    for name, path_text in args.role:
        if name in role_paths:
            raise ValueError(f"duplicate role: {name}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        role_paths[name] = path

    train_dataset = load_json(args.train_annotations)
    role_datasets = {name: load_json(path) for name, path in role_paths.items()}
    coverage = analyze_role_coverage(train_dataset, role_datasets)
    train_integrity = verify_image_inventory(
        args.train_image_root, train_dataset, args.train_download_manifest
    )

    protected_result = None
    protected_integrity = None
    protected_name = None
    protected_annotations = None
    if args.protected:
        protected_name, annotations_text, image_root_text, manifest_text = args.protected
        protected_annotations = Path(annotations_text)
        protected_image_root = Path(image_root_text)
        protected_manifest = Path(manifest_text)
        for path in (protected_annotations, protected_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        protected_dataset = load_json(protected_annotations)
        protected_result = protected_inventory(protected_dataset, train_dataset)
        protected_integrity = verify_image_inventory(
            protected_image_root, protected_dataset, protected_manifest
        )

    holdout_ready = bool(coverage["unused_source_groups"])
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only local source coverage and integrity inventory",
        "source_group_fields": GROUP_FIELDS,
        "official_train_annotations": str(args.train_annotations.resolve()),
        "official_train_annotations_sha256": file_sha256(args.train_annotations),
        "role_annotation_sha256": {
            name: file_sha256(path) for name, path in role_paths.items()
        },
        "coverage": coverage,
        "train_image_integrity": train_integrity,
        "protected": (
            {
                "name": protected_name,
                "annotations": str(protected_annotations.resolve()),
                "annotations_sha256": file_sha256(protected_annotations),
                "inventory": protected_result,
                "image_integrity": protected_integrity,
            }
            if protected_result is not None
            else None
        ),
        "official_validation_metrics_or_labels_summarized": False,
        "model_or_prediction_files_accessed": False,
        "training_or_inference_performed": False,
        "data_files_modified_or_copied": False,
        "decision": {
            "fresh_source_sequence_holdout_ready": holdout_ready,
            "eligible_local_untouched_source_groups": coverage["unused_source_groups"],
            "official_validation_eligible_for_development": False,
            "status": (
                "ready_internal_untouched_sources_found"
                if holdout_ready
                else "blocked_no_local_untouched_source_sequence"
            ),
            "required_next_action": (
                "Freeze a holdout manifest before labels or metrics are inspected."
                if holdout_ready
                else "Acquire or attach externally sourced, fully annotated maritime sequences "
                "whose source groups are absent from all existing roles."
            ),
        },
        "external_ingest_requirements": [
            "complete source-video grouping metadata",
            "image and annotation files with SHA-256 provenance",
            "life_saving_appliances positive examples",
            "real target-negative maritime scenes",
            "zero source-group overlap with all existing development roles",
            "a frozen manifest before any proposal-rule evaluation",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "untouched_sequence_inventory.json"
    role_csv = args.output_dir / "role_source_coverage.csv"
    protected_csv = args.output_dir / "protected_official_val_sources.csv"
    candidate_csv = args.output_dir / "untouched_candidate_sources.csv"
    role_rows = []
    for role, details in coverage["roles"].items():
        for item in details["groups"]:
            role_rows.append({"role": role, **item, "status": "already_used"})
    write_csv(role_csv, ["role", "source_group", "images", "status"], role_rows)
    protected_rows = protected_result["groups"] if protected_result else []
    write_csv(
        protected_csv,
        [
            "source_group",
            "images",
            "overlaps_official_train",
            "eligible_as_development_holdout",
        ],
        protected_rows,
    )
    write_csv(
        candidate_csv,
        ["source_group", "status"],
        [
            {"source_group": group, "status": "eligible_internal_untouched"}
            for group in coverage["unused_source_groups"]
        ],
    )
    result["outputs"] = {
        "result": str(result_path.resolve()),
        "role_source_coverage": str(role_csv.resolve()),
        "protected_official_val_sources": str(protected_csv.resolve()),
        "untouched_candidate_sources": str(candidate_csv.resolve()),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))
    print(
        f"train images={coverage['official_train_images']} "
        f"groups={coverage['official_train_source_groups']} "
        f"unused_groups={len(coverage['unused_source_groups'])}"
    )


if __name__ == "__main__":
    main()
