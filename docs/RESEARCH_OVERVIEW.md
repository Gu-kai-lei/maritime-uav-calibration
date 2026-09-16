# Research overview and evidence map

## Research contribution

This project asks whether maritime UAV detections can be made more trustworthy by
calibrating their confidence and by identifying why rare rescue targets fail.
It uses an established YOLOv8n detector. The original work is in source-aware experimental
control, calibration ablations, failure diagnosis, and reproducible evaluation tooling.

The investigation links four levels of evidence:
1. **Probability quality:** ECE alongside proper scores and fixed-FP recall.
2. **Detector validity:** optimization replay and an AMP intervention.
3. **Mechanism:** resolution/exposure factorial, matched crops, oracle and blind presentation.
4. **Transfer cost:** independent-domain recall measured together with false positives.

## Completed studies

| Study | Intervention / comparison | Main evidence | Report |
|---|---|---|---|
| Calibration | Raw, global, class-conditional and metadata logistic models | Metadata improves binned ECE but worsens Brier/NLL and fixed-budget recall | [Results](RESULTS.md) |
| Stability | Same-seed replay, then no AMP | Epoch-9 collapse reproduces; no-AMP control is stable | [Stability](STABILITY_REPLICATION.md) |
| Sensitivity | Repeat calibration on stable 640 detector | Metadata negative result remains | [Stable sensitivity](STABLE_640_SENSITIVITY.md) |
| Inference resolution | Paired 640/1280 prediction | Gain concentrates in smallest GT-size quartile | [Resolution](RESOLUTION_INFERENCE_SENSITIVITY.md) |
| Rare-class audit | Support, effective scale, source context | Target is rare and approximately 5.50 × 4.83 px at median 640 resize | [Audit](RARE_CLASS_AUDIT.md) |
| 2×2 factorial | Natural/Repeat4 × 640/1280 | Resolution drives aggregate AP; target AP remains zero | [Factorial](RARE_CLASS_FACTORIAL.md) |
| Target taxonomy | Localization versus class confusion | No target-class box overlaps detector-dev targets | [Taxonomy](TARGET_ERROR_TAXONOMY_2X2.md) |
| Crop4 | Matched exposure with object-centered crops | Slight aggregate gain; target proposal total falls 73/501 → 39/501 | [Crop control](OBJECT_CENTRIC_CROP_CONTROL.md) |
| Oracle crops | Supply target location at inference | Up to 23/64 targets recovered | [Oracle](ORACLE_TARGET_CROP_DIAGNOSTIC.md) |
| Blind tiles | Two frozen tile sizes, no target locations | Crop4 recovers 13/64 and 14/64 with high FP load | [Tiling](BLIND_TILING_INFERENCE_SENSITIVITY.md) |
| FP mechanism | Duplicates, other-class associations, background | Both background and class confusion matter | [FP audit](BLIND_TILING_FP_AUDIT.md) |
| External holdout | Six frozen arms on MOBDrone | Representation transfers, with prohibitive FP burden | [MOBDrone](MOBDRONE_EXTERNAL_HOLDOUT.md) |

The full factorial separates the exposure effect at each resolution from the resolution effect
under each exposure condition. The interaction is the difference-in-differences, not a comparison
of the two best-looking cells. All six external arms remain in the headline result.

## Evaluation roles and inference limits

- Official training data are grouped into detector-train, detector-dev, calibration-fit,
  and policy-tune. Source groups are indivisible within this partition.
- Detector checkpoints use detector-dev. Calibrators use calibration-fit; original operating
  thresholds use policy-tune before evaluation.
- Official validation was evaluated after freezing in the original calibration study and its
  documented sensitivity. It was not used for subsequent rare-class checkpoint selection.
- The official train/validation split shares sources. Official-validation calibration results
  should not be advertised as independent-source generalization.
- Detector-dev was repeatedly diagnosed. The later diagnostic reports are mechanistic evidence,
  not independent confirmatory tests.
- Calibration-fit and policy-tune were later examined as frozen, non-selecting diagnostics;
  those observations do not justify retuning the detector.
- MOBDrone is an external, target-only endpoint. It is now closed to method selection.
  A lack of normalized source-name overlap is a recorded provenance check, not a universal
  proof against all possible duplicate imagery.

The low-confidence proposal endpoint is recall at an export floor of 0.001. It is not the recall
of a deployed alert system. The oracle experiment uses ground-truth locations and is a diagnostic
upper-bound-style intervention, not an operational model.

## What remains unproven

There is no claim of state-of-the-art performance, deployment readiness, multi-seed robustness,
or a published peer-reviewed method. AMP causality is bounded to the recorded environment and
control. The completed optimizer-stripped weights are not resumable training checkpoints.
Recorded protocols were written locally before the corresponding runs; they were not publicly
timestamped by this first GitHub commit.

## Next work: proposal gate v1 (planned)

Develop a target-versus-hard-negative proposal gate using detector-train only, with source-grouped
development. Define its features, operating objective and acceptance criteria before evaluation.
Retain both 384/768 inference arms. Detector-dev can provide a documented development diagnostic,
but a new external set is required for independent confirmation. Do not select features,
thresholds, checkpoints or tiling parameters using the closed MOBDrone result.

No proposal-gate implementation or result is included in this release. Suggested acceptance
targets in prior discussion are not a registered experiment until written into a new protocol.

## Authorship and tools

GU Kailei owns the research project. Development and documentation were assisted by Codex;
claims should be assessed against the committed code, tests and evidence. Ultralytics supplies
the detector architecture and training engine. Source datasets and papers are credited in
[related work](RELATED_WORK.md), [dataset notes](DATASET.md), and the external report.
