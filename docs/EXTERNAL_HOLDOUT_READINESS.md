# External source-sequence holdout readiness

## Status update

The external-data gate described below was satisfied on 2026-08-16 with a preregistered MOBDrone
`DJI_0804` source-sequence holdout. Its target-only evaluation is complete; see the
[MOBDrone external holdout report](MOBDRONE_EXTERNAL_HOLDOUT.md). This page remains the historical
record showing why no local SeaDronesSee role was eligible.

## Local inventory decision

No locally available, untouched source sequence is eligible for the next blind proposal-rule
evaluation. The four official-training-derived roles jointly cover all 8,930 official training
images and all 38 source groups. The next experimental step is therefore blocked on acquiring or
attaching new, fully annotated maritime source sequences.

The official validation split remains protected final-evaluation data. Its source metadata was
inventoried only to test holdout eligibility: all 33 validation source groups also occur in the
official training data, so it is not source-disjoint and cannot be repurposed as a development
holdout. No validation labels, predictions, or metrics were summarized in this inventory.

## Frozen scope

The preregistered read-only procedure is in
[`configs/external_sequence_holdout_readiness.yaml`](../configs/external_sequence_holdout_readiness.yaml).
It permits source-name and file-integrity inventory only. It forbids training, inference,
calibration, threshold search, metric evaluation, dataset mutation, and model or prediction access.

The inventory used complete source groups rather than randomly sampled frames. Source identity was
resolved in this order: `video_id`, `source.video`, `source.drone`, then `source.folder_name`.

## Results

| Inventory item | Result |
|---|---:|
| Official training images | 8,930 |
| Official training source groups | 38 |
| Images covered by the four roles | 8,930 |
| Source groups covered by the four roles | 38 |
| Unused images | 0 |
| Unused source groups | 0 |
| Pairwise role overlap in image IDs, filenames, and source groups | 0 |
| Protected official-validation images | 1,547 |
| Protected official-validation source groups | 33 |
| Validation groups overlapping official train | 33 |
| Validation groups novel to official train | 0 |

The local train and protected-validation image directories match their annotation filenames and
download-manifest byte sizes exactly. Their filename-and-size inventory digests are recorded in the
JSON result. Per-image content SHA-256 was not computed, so the server ETags and aggregate
filename-size digests are reported only as integrity metadata, not as content hashes.

The versioned outputs are:

- [`untouched_sequence_inventory.json`](../results/data_inventory/untouched_sequence_inventory.json)
- [`role_source_coverage.csv`](../results/data_inventory/role_source_coverage.csv)
- [`protected_official_val_sources.csv`](../results/data_inventory/protected_official_val_sources.csv)
- [`untouched_candidate_sources.csv`](../results/data_inventory/untouched_candidate_sources.csv)

The candidate CSV contains only its header because no eligible local source group exists.

## External-data gate used for the follow-up

Before the completed MOBDrone evaluation, the new holdout was required to satisfy all of the
following:

1. Complete source-video grouping metadata is present.
2. Images and annotations have recorded SHA-256 provenance.
3. The data include `life_saving_appliances` positives and realistic maritime target-negative
   scenes.
4. Every source group is absent from detector-train, detector-dev, calibration-fit, policy-tune,
   and any prior diagnostic role.
5. A manifest and evaluation rule are frozen before predictions are generated.

MOBDrone satisfied these acquisition and provenance requirements. Its result supports external
transfer of the Crop4 representation/presentation mechanism, but the measured false-positive load
does not support a deployable proposal-rule claim. No additional training or official-validation
evaluation was authorized by either result.

## Reproduction

```powershell
python scripts\inventory_untouched_sequence_holdout.py `
  --train-annotations D:\dataset\SeaDronesSee_ODv2\compressed\annotations\instances_train.json `
  --train-image-root D:\dataset\SeaDronesSee_ODv2\compressed\images\train `
  --train-download-manifest D:\dataset\SeaDronesSee_ODv2\manifests\train_webdav_manifest.json `
  --role detector_train artifacts\partitions\seed20260803\instances_detector_train.json `
  --role detector_dev artifacts\partitions\seed20260803\instances_detector_dev.json `
  --role calibration_fit artifacts\partitions\seed20260803\instances_calibration_fit.json `
  --role policy_tune artifacts\partitions\seed20260803\instances_policy_tune.json `
  --protected official_validation D:\dataset\SeaDronesSee_ODv2\compressed\annotations\instances_val.json D:\dataset\SeaDronesSee_ODv2\compressed\images\val D:\dataset\SeaDronesSee_ODv2\manifests\val_webdav_manifest.json `
  --output-dir results\data_inventory
```
