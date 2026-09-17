# Reproducibility

Run commands from the repository root. Python >=3.10 is declared; CI checks 3.10 and 3.12.
Historical training used Python 3.12.7, PyTorch 2.6.0+cu124 and Ultralytics 8.4.51;
see [environment notes](ENVIRONMENT.md). Broad package bounds are not an exact environment lock.

## Level 1: inspect and regenerate compact evidence (CPU)

```bash
# CI installs the CPU-only PyTorch 2.6 wheel first; follow the official
# PyTorch selector if your platform needs a different build.
python -m pip install -e ".[dev]"
python -m pytest
python scripts/reproduce_published_results.py
python scripts/verify_publication.py
```

This uses only committed CSV/JSON files. The first script checks the factorial arithmetic,
external TP/FP accounting and the calibration claim, then writes a summary JSON and figure to
`artifacts/reproduced_summary/`. The second checks the public artifact hashes and tracked-file
publication rules. Neither runs inference, downloads weights, or re-evaluates a holdout.

The artifact catalog is generated at publication and detects changed or missing public result
files. It does not attest to the original correctness of those results. Synthetic tests cover
matching, splitting, crops, calibration, tiling, external extraction and analysis.

## Level 2: reconstruct the data roles

Obtain the original datasets from their official hosts. Do not use the downloaded official
validation set for detector early stopping. Use a separate derived workspace for labels/caches.
The optional WebDAV image downloader reads `SEADRONESSEE_SHARE_TOKEN` from the local process
environment; obtain that value from the official host and never commit it.
The commands below assume an untouched extracted SeaDronesSee tree under
`data/SeaDronesSee_ODv2/compressed/`; replace that prefix to match your machine.

```bash
python scripts/validate_coco.py --annotations data/SeaDronesSee_ODv2/compressed/annotations/instances_train.json --require-metadata
python scripts/partition_training_coco.py --annotations data/SeaDronesSee_ODv2/compressed/annotations/instances_train.json --output-dir artifacts/partitions/seed20260803 --seed 20260803
python scripts/prepare_yolo_dataset.py --train-annotations data/SeaDronesSee_ODv2/compressed/annotations/instances_train.json --val-annotations data/SeaDronesSee_ODv2/compressed/annotations/instances_val.json --dataset-root data/SeaDronesSee_ODv2/compressed --output-yaml artifacts/yolo/seadronessee_odv2.yaml --partition-dir artifacts/partitions/seed20260803
```

The label-conversion script refuses to overwrite an existing labels directory.
It creates a separate `seadronessee_detector.yaml` whose train and val entries are
detector-train and detector-dev. Compare partition counts and seed search offsets with
[the recorded partition](PARTITION_SNAPSHOT_20260803.md).
Historical YAML files retain dataset-drive examples. Adapt paths in a working copy; do not
silently change protocol parameters.

## Level 3: rerun training and inference (data + GPU)

Install the appropriate PyTorch build for your device, then
`python -m pip install -e ".[train,dev]"`. Reproducing the historic environment additionally
requires the versions recorded in the manifests. The commands here are instructions for a
deliberate rerun, not commands executed by CI.

Example stable natural-640 run, in a new output directory:

```bash
python scripts/train_yolo.py --data artifacts/yolo/seadronessee_detector.yaml --imgsz 640 --batch 16 --nbs 64 --no-amp --seed 20260803 --workers 4 --epochs 100 --patience 20 --name reproduction_natural640
```

1280 experiments use batch 4 and nbs 64. Repeat4 changes rare-positive exposure; Crop4 uses
6,251 originals plus 969 derived crops. Follow the respective
[factorial](RARE_CLASS_FACTORIAL.md) and [crop](OBJECT_CENTRIC_CROP_CONTROL.md) protocols,
including derived dataset isolation. Do not compare a newly trained checkpoint to historical
results without recording its identity and role.

Example low-threshold export:

```bash
python scripts/export_predictions.py --model runs/detect/reproduction_natural640/weights/best.pt --annotations artifacts/partitions/seed20260803/instances_detector_dev.json --image-root data/SeaDronesSee_ODv2/compressed/images/train --output artifacts/reproduction/predictions_detector_dev.json --imgsz 640 --conf 0.001 --iou 0.70
```

Use `export_blind_tiled_predictions.py --help` for the frozen tiled exporter: tile sizes 384
and 768, overlap .25, input 1280, tile/merge IoU .70, max-det 300, plus an uncapped target stream.
Its output guard prevents overwriting existing predictions.

## External provenance and analysis

[The MOBDrone report](MOBDRONE_EXTERNAL_HOLDOUT.md) and its
[configuration](../configs/mobdrone_external_holdout.yaml) are the definitive contract.
The preparation, range downloader, frame extractor and analysis scripts expose `--help`.
Prepare fixed-rate manifests from official annotations, verify selected video entries and
frame geometry, export all six frozen arms, and analyze the comparable target class only.

Original dataset annotations, images, prediction streams and model weights are omitted from
Git. Consequently a clone can reproduce table-level checks but cannot immediately reproduce
the historic GPU inference. The checkpoint catalog identifies the required weights by SHA-256.
A deliberate exact rerun is permissible as reproduction; the completed external set remains
closed to selection and new method development.
