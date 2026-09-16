# Frozen Experiment Protocol

Version: 0.1 — drafted before downloading data or running experiments.

## Hypotheses

- H1: Raw detector confidence is less calibrated for high-altitude and very small targets.
- H2: A global logistic calibrator improves overall calibration metrics but leaves systematic
  altitude-dependent error.
- H3: Adding altitude, gimbal pitch, and predicted relative box area improves calibration and
  recall at a fixed false-positive-per-image budget on the frozen validation split.

## Outcome definition

A prediction is labelled correct when it is greedily matched to an unmatched ground-truth box
of the same category at IoU >= 0.50. Predictions must be exported at confidence 0.001 before
calibration. Changing this floor after viewing validation results is not allowed.

## Data partitions

1. Start from the official training and validation annotations.
2. Partition official training data by source video into detector-train (70%), detector-dev
   (10%), calibration-fit (10%), and policy-tune (10%). Whole videos are indivisible, so the
   implementation minimizes image-count deviation while preserving strict source separation.
   Trinity RGB and multispectral images lack a video key and are conservatively grouped together
   by drone source rather than split by sensor folder.
   Every role must contain at least 25 ground-truth boxes from each active class; the deterministic
   seed search and its offset are recorded in the partition report.
3. Use detector-dev only for detector early stopping and checkpoint selection.
4. Fit calibrators on calibration-fit and choose operating thresholds on policy-tune.
5. Keep the official validation split untouched until the full pipeline is frozen.
6. Use official test-server submission only as an optional final confirmation.

The primary metadata-aware analysis uses only images with both altitude and gimbal pitch. A
secondary all-image sensitivity analysis is limited to raw, global, and class-conditional variants
that do not require flight metadata; it must not be mixed with the complete-case result. Verified
coverage is 6,212/8,930 training images and 1,093/1,547 validation images.

If v2 annotations do not expose a reliable sequence key, document the available fields before
choosing a fallback. An image-level fallback must be explicitly labelled as a leakage risk.

## Detector baseline

- Model: YOLOv8n public COCO initialization.
- Input: 640 px.
- Seed: 20260803.
- Primary run: 100 epochs, early stopping patience 20.
- Select weights using the calibration-independent detector development metric defined after
  inspecting the dataset structure. Never select using the frozen official validation split.

## Calibration variants

- Raw: detector confidence without fitting.
- Global: logistic calibration using score logit only.
- Class-conditional: score logit plus one-hot target category, controlling for class difficulty.
- Altitude: class-conditional baseline plus log altitude.
- Altitude + gimbal: add sine/cosine encoding of gimbal pitch.
- Full metadata: add log predicted relative box area.

The primary comparison fits and evaluates every variant on the same metadata-complete subset.
An all-image raw/global/class-conditional sensitivity analysis is reported separately. All fitted
models use median imputation, standardized numerical features, and one-hot target category. Logistic
regression is the primary calibrator because its coefficients and ablations are interpretable.

## Primary metrics

- Detection: AP50-95, AP50, precision, recall.
- Calibration: ECE, maximum calibration error, Brier score, negative log-likelihood.
- Operational: recall at fixed false positives per image.
- Slices: altitude, gimbal pitch, predicted size, ground-truth size, and category.
- Runtime: preprocessing, detector inference, postprocessing, and calibration latency.

## Analysis rules

- Report confidence intervals where practical, preferably by grouped bootstrap over source
  sequence rather than individual frames.
- Treat source-role slices on official validation as diagnostic: the official benchmark partitions
  frames, and validation videos overlap the official training videos. Do not call this a
  source-generalization test.
- Report every planned variant, including negative results.
- Separate detector quality from calibration quality: calibration cannot recover a target that
  was never proposed above the low export threshold.
- Do not claim deployment readiness from benchmark results.
- Do not claim novelty until a focused literature review is completed.

## Stop conditions

The first public release is complete when dataset integrity is verified, one detector baseline
is reproducible, all planned calibration variants are evaluated on frozen validation data,
ablation and reliability figures are generated, and limitations are documented.
