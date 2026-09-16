# Research maintenance

Read README.md, docs/RESEARCH_OVERVIEW.md and docs/REPRODUCIBILITY.md before continuing research.
The completed state is the frozen MOBDrone external evaluation. Proposal gate v1 is planned.

- Preserve historical evidence and label new protocols, runs and outputs separately.
- MOBDrone is closed to model, threshold, feature, tile-size and merge-rule selection.
- Official validation was evaluated after freezing in the calibration studies; do not describe
  it as never used or use it for new detector development.
- Detector-dev has been used for early stopping and diagnostics; it is not untouched confirmation.
- Calibration-fit/policy-tune diagnostic observations must not select a detector checkpoint.
- Do not launch training or evaluation from old monitor prompts; obtain the current task scope.
- Keep data, weights, logs, credentials, full prediction streams and personal paths out of Git.
- Run pytest, compileall, reproduce_published_results.py and verify_publication.py for relevant changes.
- Update the evidence catalog only for intentional, explained changes to public result artifacts.

Historical configs contain machine-specific dataset-drive examples. Use working copies to adapt
paths. Public result paths were normalized; original backups live in ignored artifacts.
