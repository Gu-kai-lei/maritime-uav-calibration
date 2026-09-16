# Changelog

## 0.2.0 — 2026-09-16

First public research snapshot, assembling experiments recorded locally in August 2026.
This release date is not the date the experiments were run.

- Published the calibration pipeline, original protocol, proper-score ablations and source-grouped
  uncertainty estimates, including the metadata calibration negative result.
- Included deterministic collapse replication, no-AMP controls, stable calibration sensitivity,
  resolution/exposure factorial, matched Crop4 control and target-error taxonomy.
- Included oracle crops, blind tiling, false-positive mechanism analysis and the completed
  six-arm MOBDrone external target-only evaluation.
- Added bilingual navigation, a research contribution/limitations map, reproduction levels,
  an artifact policy, checkpoint identities, historical run manifests and per-epoch metrics.
- Added CPU-only evidence regeneration, public artifact checksums and publication guard tests.
- Configured Python 3.10/3.12 CI. Local verification: 85 tests pass; compileall and compact
  evidence regeneration pass. Hosted CI status is available in GitHub Actions.

No new detector training or external evaluation was performed for this publication.
Original affected provenance files were backed up locally before personal-path normalization.
Proposal gate v1 remains planned.
