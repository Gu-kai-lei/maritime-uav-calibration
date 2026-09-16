from __future__ import annotations

from scripts.inventory_untouched_sequence_holdout import (
    analyze_role_coverage,
    image_group,
    sha256_lines,
)


def dataset(images: list[dict]) -> dict:
    return {"images": images, "annotations": [], "categories": []}


def image(identifier: int, video: str) -> dict:
    return {"id": identifier, "file_name": f"{identifier}.jpg", "source": {"video": video}}


def test_image_group_prefers_source_video() -> None:
    assert image_group(image(1, "flight.mp4")) == "source.video:flight.mp4"


def test_complete_disjoint_role_coverage_has_no_candidate() -> None:
    full = dataset([image(1, "a.mp4"), image(2, "b.mp4")])
    roles = {
        "train": dataset([image(1, "a.mp4")]),
        "dev": dataset([image(2, "b.mp4")]),
    }
    result = analyze_role_coverage(full, roles)
    assert result["complete_image_coverage"] is True
    assert result["complete_source_group_coverage"] is True
    assert result["unused_source_groups"] == []
    assert all(
        item["source_group_overlap_with_prior_roles"] == 0
        for item in result["pairwise_accumulated_overlap_checks"]
    )


def test_unused_source_group_is_reported() -> None:
    full = dataset([image(1, "a.mp4"), image(2, "b.mp4")])
    result = analyze_role_coverage(full, {"train": dataset([image(1, "a.mp4")])})
    assert result["unused_source_groups"] == ["source.video:b.mp4"]
    assert result["complete_image_coverage"] is False


def test_inventory_digest_is_order_independent() -> None:
    assert sha256_lines(["a\t1", "b\t2"]) == sha256_lines(["b\t2", "a\t1"])
