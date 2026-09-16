from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

from .coco import source_group


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _closest_group_subset(
    groups: dict[str, list[dict[str, Any]]],
    candidates: set[str],
    target_images: int,
    seed: int,
) -> set[str]:
    """Choose a deterministic whole-group subset closest to an image-count target."""
    ordered = sorted(candidates, key=lambda group: (_stable_fraction(group, seed), group))
    # Each reachable sum stores one deterministic tuple of group names.
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(groups[group])
        additions = {
            current + size: chosen + (group,)
            for current, chosen in list(reachable.items())
            if current + size not in reachable
        }
        reachable.update(additions)
    allowed = [total for total, chosen in reachable.items() if chosen and len(chosen) < len(candidates)]
    if not allowed:
        raise ValueError("cannot form a non-empty group subset while preserving a remainder")
    selected_total = min(
        allowed,
        key=lambda total: (abs(total - target_images), total < target_images, total),
    )
    return set(reachable[selected_total])


def partition_coco_by_group(
    dataset: dict[str, Any],
    fractions: dict[str, float],
    group_fields: Iterable[str],
    seed: int = 20260803,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Partition COCO data into mutually exclusive source-group roles.

    Fractions are applied in insertion order. The final role receives every
    remaining group so no image is dropped.
    """
    if len(fractions) < 2:
        raise ValueError("at least two partition roles are required")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("all partition fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("partition fractions must sum to one")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[Any] = []
    for image in dataset["images"]:
        group = source_group(image, group_fields)
        if group is None:
            missing.append(image.get("id"))
        else:
            groups[group].append(image)
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"{len(missing)} images have no sequence group; first IDs: {preview}. "
            "Do not use an image-level fallback without documenting leakage risk."
        )
    if len(groups) < len(fractions):
        raise ValueError("fewer source groups than requested partition roles")

    total_images = len(dataset["images"])
    remaining = set(groups)
    role_groups: dict[str, set[str]] = {}
    roles = list(fractions)
    for index, role in enumerate(roles[:-1]):
        target = round(total_images * fractions[role])
        selected = _closest_group_subset(groups, remaining, target, seed + index)
        role_groups[role] = selected
        remaining -= selected
    role_groups[roles[-1]] = remaining

    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    partitions: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "seed": seed,
        "fractions_requested": fractions,
        "groups_total": len(groups),
        "images_total": total_images,
        "partitions": {},
    }
    for role in roles:
        image_ids = {
            image["id"]
            for group in role_groups[role]
            for image in groups[group]
        }
        output = {
            key: deepcopy(value)
            for key, value in dataset.items()
            if key not in {"images", "annotations"}
        }
        output["images"] = [
            deepcopy(image) for image in dataset["images"] if image["id"] in image_ids
        ]
        output["annotations"] = [
            deepcopy(annotation)
            for image_id in image_ids
            for annotation in annotations_by_image.get(image_id, [])
        ]
        partitions[role] = output
        counts = Counter(annotation["category_id"] for annotation in output["annotations"])
        report["partitions"][role] = {
            "requested_fraction": fractions[role],
            "actual_image_fraction": len(output["images"]) / total_images,
            "groups": sorted(role_groups[role]),
            "group_count": len(role_groups[role]),
            "images": len(output["images"]),
            "annotations": len(output["annotations"]),
            "category_counts": dict(sorted(counts.items())),
        }

    assigned_ids = [image["id"] for output in partitions.values() for image in output["images"]]
    if len(assigned_ids) != total_images or len(set(assigned_ids)) != total_images:
        raise AssertionError("partitioning lost or duplicated image IDs")
    return partitions, report


def split_coco_by_group(
    dataset: dict[str, Any],
    calibration_fraction: float,
    group_fields: Iterable[str],
    seed: int = 20260803,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[Any] = []
    for image in dataset["images"]:
        group = source_group(image, group_fields)
        if group is None:
            missing.append(image.get("id"))
        else:
            groups[group].append(image)
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"{len(missing)} images have no sequence group; first IDs: {preview}. "
            "Do not use an image-level fallback without documenting leakage risk."
        )
    if len(groups) < 2:
        raise ValueError("at least two source groups are required")

    calibration_groups = {
        group for group in groups if _stable_fraction(group, seed) < calibration_fraction
    }
    if not calibration_groups:
        calibration_groups.add(min(groups, key=lambda group: _stable_fraction(group, seed)))
    if len(calibration_groups) == len(groups):
        calibration_groups.remove(max(groups, key=lambda group: _stable_fraction(group, seed)))

    calibration_image_ids = {
        image["id"] for group in calibration_groups for image in groups[group]
    }
    detector_image_ids = {
        image["id"]
        for group, images in groups.items()
        if group not in calibration_groups
        for image in images
    }

    def subset(image_ids: set[Any]) -> dict[str, Any]:
        output = {
            key: deepcopy(value)
            for key, value in dataset.items()
            if key not in {"images", "annotations"}
        }
        output["images"] = [deepcopy(image) for image in dataset["images"] if image["id"] in image_ids]
        output["annotations"] = [
            deepcopy(annotation)
            for annotation in dataset["annotations"]
            if annotation["image_id"] in image_ids
        ]
        return output

    detector = subset(detector_image_ids)
    calibration = subset(calibration_image_ids)
    report = {
        "seed": seed,
        "requested_calibration_fraction": calibration_fraction,
        "groups_total": len(groups),
        "groups_detector": len(groups) - len(calibration_groups),
        "groups_calibration": len(calibration_groups),
        "images_detector": len(detector["images"]),
        "images_calibration": len(calibration["images"]),
        "annotations_detector": len(detector["annotations"]),
        "annotations_calibration": len(calibration["annotations"]),
        "calibration_groups": sorted(calibration_groups),
        "category_counts_detector": dict(
            sorted(Counter(annotation["category_id"] for annotation in detector["annotations"]).items())
        ),
        "category_counts_calibration": dict(
            sorted(Counter(annotation["category_id"] for annotation in calibration["annotations"]).items())
        ),
    }
    return detector, calibration, report
