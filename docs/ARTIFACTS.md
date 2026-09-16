# Artifact availability and publication policy

The public repository preserves code, protocols, historical stage reports, compact CSV/JSON
results, analytical figures, acquisition manifests and checkpoint identities.
`results/artifact_catalog.json` lists the public result files with byte sizes and SHA-256.
`results/checkpoint_catalog.json` identifies locally available training checkpoints and records
whether each on-disk best.pt agrees with its run manifest at publication time.

## Included and omitted material

| Material | Availability | Reconstruction / evidence |
|---|---|---|
| Code, tests, configs, protocols | Git | Package and CLI sources |
| Aggregate and source-level metrics, analytical figures | Git | Results directories and integrity catalog |
| Training provenance and per-epoch metrics | Git, under results/training_runs | Copied from completed local run manifests and results.csv |
| Model best.pt / last.pt | Local only | Checkpoint catalog and training recipes; no download URL claimed |
| Original images, videos, official annotations | Obtain from original hosts | Dataset/version links and recorded hashes |
| Derived MOBDrone COCO annotation exports | Local only | prepare_mobdrone_external_holdout.py |
| Full per-image detection streams and 24 MB FP audit table | Local only | Export and analysis scripts |
| FP contact sheet containing dataset pixels | Local only | analyze_blind_tiling_false_positives.py |
| Original logs, local environments, conversation exports | Local only | Not required for CPU checks |

Omission from Git does not delete local evidence. Dataset-derived contact sheets and original
annotations are not bundled as repository-code assets. Published per-annotation diagnostic
tables contain evaluation measurements rather than a redistributed source annotation package.

The historical FP report references a locally generated contact sheet. The historical MOBDrone
report references locally regenerated project-contract annotations. Follow their script recipes
if these local-only artifacts are needed; their absence from Git is intentional.

## Paths and preserved provenance

Before the first public commit, absolute paths containing a personal Windows home were converted
to repository-relative paths or portable location placeholders. Original affected files were
backed up under ignored `artifacts/publication_originals/`.
Metrics, seeds, dates, hashes of original inputs and experiment settings were retained.
Checksums recorded inside historical JSON still describe their original source artifacts; the
publication catalog separately hashes the sanitized published files.

The original historical configs include example `D:/dataset/` paths. These identify the prior
layout and are not promised to exist on another machine. Use local working copies for new paths.
The earliest experimental runs predate any Git commit, and their null git_commit fields are
retained. This repository does not manufacture a historical commit trail.

## Licenses and citation

MIT applies to original repository code. External dependencies, weights and datasets retain their
own terms. In particular, do not infer that YOLO model weights have been MIT-relicensed by this
repository. Refer to the upstream projects before redistributing models or derived imagery.

- SeaDronesSee: https://seadronessee.cs.uni-tuebingen.de/dataset
- MOBDrone: https://aimh.isti.cnr.it/dataset/mobdrone/
- MOBDrone archived version: https://doi.org/10.5281/zenodo.5996890
- Ultralytics: https://github.com/ultralytics/ultralytics

Use the repository citation metadata and cite the source datasets when using their data.
