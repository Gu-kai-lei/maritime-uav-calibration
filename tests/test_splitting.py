from maritime_calibration.splitting import partition_coco_by_group, split_coco_by_group


def test_group_split_has_no_group_overlap():
    images = []
    annotations = []
    annotation_id = 1
    for group_index in range(8):
        for frame_index in range(2):
            image_id = group_index * 10 + frame_index
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{image_id}.jpg",
                    "source": {"video": f"flight_{group_index}.mp4"},
                }
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [0, 0, 2, 2],
                }
            )
            annotation_id += 1
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "target"}],
    }
    detector, calibration, report = split_coco_by_group(
        dataset, 0.25, ["source.video"], seed=42
    )
    detector_groups = {image["source"]["video"] for image in detector["images"]}
    calibration_groups = {image["source"]["video"] for image in calibration["images"]}
    assert detector_groups.isdisjoint(calibration_groups)
    assert report["images_detector"] + report["images_calibration"] == 16


def test_four_role_partition_is_complete_and_group_disjoint():
    images = [
        {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "source": {"video": f"flight_{image_id // 3}.mp4"},
        }
        for image_id in range(30)
    ]
    dataset = {"images": images, "annotations": [], "categories": []}
    fractions = {"dev": 0.1, "calibration": 0.2, "policy": 0.2, "train": 0.5}
    partitions, report = partition_coco_by_group(
        dataset, fractions, ["source.video"], seed=7
    )
    all_ids = [image["id"] for part in partitions.values() for image in part["images"]]
    assert sorted(all_ids) == list(range(30))
    role_groups = [
        {image["source"]["video"] for image in part["images"]}
        for part in partitions.values()
    ]
    for index, left in enumerate(role_groups):
        for right in role_groups[index + 1 :]:
            assert left.isdisjoint(right)
    assert sum(item["images"] for item in report["partitions"].values()) == 30
