# Maritime UAV Detection: Calibration, Rare Targets & Transfer

[![Tests](https://github.com/Gu-kai-lei/maritime-uav-calibration/actions/workflows/tests.yml/badge.svg)](https://github.com/Gu-kai-lei/maritime-uav-calibration/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/original_code-MIT-blue.svg)](LICENSE)
[中文导读](README.zh-CN.md) · [Research map](docs/RESEARCH_OVERVIEW.md) · [Reproduce](docs/REPRODUCIBILITY.md) · [Results](docs/RESULTS.md) · [Artifacts](docs/ARTIFACTS.md)

**When a maritime UAV detector misses a rare rescue target, is the problem confidence,
optimization, exposure, scale, or the way the target is presented?**

This research repository follows that question from metadata-aware confidence calibration
to controlled rare-class experiments and a frozen external-domain evaluation. It includes
reusable evaluation code, experiment protocols, failure analyses, machine-readable results,
and synthetic tests. It reports negative findings alongside improvements.

The detector is **Ultralytics YOLOv8n**; the contribution is the experimental design,
calibration/evaluation pipeline, and evidence explaining failure modes—not a new detector architecture.
The original calibration study is retained in full.

## Findings at a glance

| Question | Evidence | Supported conclusion |
|---|---|---|
| Does flight metadata improve calibration? | Full metadata reduces ECE from 0.02224 to 0.01123, but worsens Brier from 0.04238 to 0.04811 | Lower binned ECE alone does not establish better calibration |
| Was the weak detector an optimization artifact? | Same-seed replay reproduces epoch-9 collapse; a no-AMP control remains stable | Stability must be checked before interpreting calibration |
| Does resolution or rare-positive exposure help? | Natural 640 → 1280 increases detector-dev mAP50–95 from 0.27981 to 0.36570; target AP remains zero in all four factorial cells | Aggregate gains conceal an unresolved rare-class failure |
| Did object-centered crops teach a representation? | Oracle crops recover up to 23/64 targets; blind Crop4 tiles recover 13/64 and 14/64 | Presentation can expose a learned representation |
| Does it transfer to independent videos? | On MOBDrone, Crop4 tiles recover 188/350 and 132/350 targets, with 7.98 and 3.41 target FP/image | Representation transfers; false positives prevent a deployment claim |

The calibration row uses **SeaDronesSee official validation after freezing**. The factorial
and diagnostic rows use **detector-dev**. The external row uses **MOBDrone target-only evaluation**.
These endpoints are not interchangeable.

![External target-only transfer and false-positive burden](results/external_holdout/mobdrone/evaluation/mobdrone_external_holdout.png)

## External evaluation: report all six arms

1,264 fixed-rate frames from 26 MOBDrone videos; 350 lifebuoys and 915 target-negative frames.
Only `life_buoy → life_saving_appliances` is mapped. Confidence floor 0.001; NMS IoU 0.70.
Tiles use 25% overlap and are resized to 1280 for inference.

| Frozen arm | Target recall @ IoU .50 | Target AP50 | Target FP/image | Negative frames triggered |
|---|---:|---:|---:|---:|
| Repeat4 full frame | 0.1257 | 0.0791 | 0.0854 | 2.62% |
| Crop4 full frame | 0.0857 | 0.0346 | 0.3544 | 10.82% |
| Repeat4 tile 384 | 0.0000 | 0.0000 | 15.8402 | 62.95% |
| Repeat4 tile 768 | 0.0057 | 0.0013 | 1.0941 | 26.45% |
| Crop4 tile 384 | 0.5371 | 0.3055 | 7.9794 | 81.09% |
| Crop4 tile 768 | 0.3771 | 0.2186 | 3.4098 | 60.11% |

[Source table](results/external_holdout/mobdrone/evaluation/condition_metrics.csv) ·
[Protocol and limitations](docs/MOBDRONE_EXTERNAL_HOLDOUT.md).
No checkpoint, threshold, tile size, or inference arm was selected from this holdout.

## Run the CPU evidence check

No dataset, weights, GPU, or model download is needed:

```bash
git clone https://github.com/Gu-kai-lei/maritime-uav-calibration.git
cd maritime-uav-calibration
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python scripts/reproduce_published_results.py
python scripts/verify_publication.py
```

The reproduction script validates arithmetic in the committed tables and regenerates
a three-panel summary under `artifacts/reproduced_summary/`. This checks published evidence;
it does not independently rerun detector inference or validate source annotations.
See [full reproduction instructions](docs/REPRODUCIBILITY.md) for those requirements.

## What is implemented

- COCO integrity and metadata auditing; active/ignored class contracts.
- Source-grouped detector-train / detector-dev / calibration-fit / policy-tune partitions.
- Greedy IoU matching, ECE/Brier/NLL, fixed-FP operating points, grouped bootstrap.
- Global and metadata-aware logistic calibration with explicit ablations.
- Training manifests, checkpoint hashes, deterministic replay and finite-weight checks.
- A resolution × exposure 2×2 design, matched object-centered crops, and target error taxonomy.
- Oracle crop and location-blind tiling diagnostics with class-aware merge and uncapped target audit.
- External video acquisition by ZIP byte range, CRC32/SHA-256 checks, deterministic frame extraction,
  target-only transfer evaluation, and source-level error analysis.

## Navigate the evidence

| Stage | Read |
|---|---|
| Original protocol and calibration | [Protocol](protocols/experiment_protocol.md), [results](docs/RESULTS.md), [related work](docs/RELATED_WORK.md) |
| Optimization stability | [Replication](docs/STABILITY_REPLICATION.md), [stable calibration sensitivity](docs/STABLE_640_SENSITIVITY.md) |
| Scale and rare-class failure | [Resolution](docs/RESOLUTION_INFERENCE_SENSITIVITY.md), [rare-class audit](docs/RARE_CLASS_AUDIT.md) |
| Controlled interventions | [2×2 factorial](docs/RARE_CLASS_FACTORIAL.md), [target taxonomy](docs/TARGET_ERROR_TAXONOMY_2X2.md), [Crop4](docs/OBJECT_CENTRIC_CROP_CONTROL.md) |
| Mechanism and failure cost | [Oracle crops](docs/ORACLE_TARGET_CROP_DIAGNOSTIC.md), [blind tiling](docs/BLIND_TILING_INFERENCE_SENSITIVITY.md), [FP audit](docs/BLIND_TILING_FP_AUDIT.md) |
| Independent-domain evidence | [Holdout readiness](docs/EXTERNAL_HOLDOUT_READINESS.md), [MOBDrone](docs/MOBDRONE_EXTERNAL_HOLDOUT.md) |

```text
src/maritime_calibration/  Reusable matching, metrics, calibration, splitting, crop and YOLO utilities
scripts/                  Data preparation, training, inference and analysis CLIs
configs/                  Recorded experiment definitions and outcomes
protocols/                Original research protocol
results/                  Compact tables, figures, provenance and integrity catalog
tests/                    Synthetic tests and evidence consistency checks
docs/                     Stage reports, reproduction guide and claim boundaries
```

## Status and limitations

Completed: calibration, stability controls, four-cell factorial, Crop4 control, oracle/blind
diagnostics, FP audit, and external MOBDrone evaluation. **Proposal gate v1 is planned, not implemented.**

The experiments use a single training seed. The official SeaDronesSee frame split has source
overlap; it is not an independent-video generalization test. Detector-dev has been reused for
diagnosis and is not an untouched confirmation set. MOBDrone is now closed to selection.
Historical protocol files record local preregistration decisions; the initial GitHub publication
does not establish independently timestamped preregistration.

Weights, images, original annotations, logs, and full prediction streams are not distributed.
Their provenance, hashes where recorded, reconstruction steps, and availability limits are
documented in [ARTIFACTS](docs/ARTIFACTS.md). Historical configuration paths must be adapted on
another machine. The public repository does not claim one-command bitwise reproduction.

## Attribution

Research implementation and analysis: **GU Kailei**.
See [CITATION.cff](CITATION.cff), [contribution guidance](CONTRIBUTING.md), and
[research overview](docs/RESEARCH_OVERVIEW.md).

Original repository code is MIT-licensed. Ultralytics, pretrained weights, SeaDronesSee and
MOBDrone retain their own licenses and citation requirements; the repository license does not
relicense them. Obtain data from the
[SeaDronesSee official host](https://seadronessee.cs.uni-tuebingen.de/dataset) and
[MOBDrone official host](https://aimh.isti.cnr.it/dataset/mobdrone/).
