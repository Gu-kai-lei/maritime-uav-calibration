from maritime_calibration.coco import (
    active_categories,
    audit_coco,
    metadata_complete_subset,
    metadata_value,
    source_group,
)


def sample_dataset():
    return {
        "images": [
            {
                "id": 1,
                "file_name": "a.jpg",
                "width": 100,
                "height": 80,
                "meta": {"altitude": 42.5, "gimbal_pitch": 60},
                "source": {"video": "flight_a.mp4"},
            }
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 15]}],
        "categories": [{"id": 1, "name": "object"}],
    }


def test_audit_valid_coco_and_metadata():
    report = audit_coco(sample_dataset())
    assert report["valid"]
    assert report["annotations"] == 1
    assert report["metadata_coverage"] == {"altitude": 1, "gimbal_pitch": 1}


def test_metadata_and_group_extraction():
    image = sample_dataset()["images"][0]
    assert metadata_value(image, ["altitude"]) == 42.5
    assert source_group(image, ["source.video"]) == "source.video:flight_a.mp4"


def test_image_pixel_height_is_not_treated_as_flight_altitude():
    dataset = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 3840, "height": 2160}],
        "annotations": [],
        "categories": [{"id": 1, "name": "target"}],
    }
    report = audit_coco(dataset)
    assert report["metadata_coverage"]["altitude"] == 0


def test_ignored_category_is_not_a_detector_class():
    dataset = {
        "categories": [
            {"id": 0, "name": "ignored"},
            {"id": 1, "name": "swimmer"},
            {"id": 2, "name": "boat"},
        ]
    }
    assert [category["id"] for category in active_categories(dataset)] == [1, 2]


def test_metadata_complete_subset_filters_images_and_annotations_together():
    dataset = sample_dataset()
    dataset["images"].append({"id": 2, "file_name": "b.jpg", "width": 100, "height": 80})
    dataset["annotations"].append(
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]}
    )
    subset, report = metadata_complete_subset(dataset)
    assert [image["id"] for image in subset["images"]] == [1]
    assert [annotation["image_id"] for annotation in subset["annotations"]] == [1]
    assert report == {
        "images_before": 2,
        "images_after": 1,
        "annotations_before": 2,
        "annotations_after": 1,
    }


def test_audit_reports_semantic_duplicate_boxes_without_invalidating_source():
    dataset = sample_dataset()
    duplicate = dict(dataset["annotations"][0])
    duplicate["id"] = 2
    dataset["annotations"].append(duplicate)
    report = audit_coco(dataset)
    assert report["valid"]
    assert report["exact_duplicate_box_groups"] == 1
    assert report["exact_duplicate_annotations"] == 1
    assert "exact duplicate" in report["warnings"][0]
