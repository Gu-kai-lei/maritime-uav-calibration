# Contributing

Use a new branch and run `python -m pytest`, `python scripts/reproduce_published_results.py`
and `python scripts/verify_publication.py` before submitting a change.

Bug fixes should include a small synthetic regression case. New experiments require a distinct
protocol, source-role definition, seed, environment record and output directory.
Preserve historical results and distinguish reruns from new findings.
Never select methods on the closed MOBDrone holdout or reinterpret oracle locations as deployable inputs.

Keep detector-dev diagnostics separate from independent confirmation. Report all registered arms,
negative results, class mapping, low-confidence floor and FP cost. Do not add images, weights,
credentials, raw detection streams or personal machine paths to Git.

When intentionally changing published evidence, document why and regenerate the integrity catalog
using `python scripts/verify_publication.py --write-catalog` after reviewing the changed files.
Generating a fresh catalog does not validate scientific claims. Ordinary code changes should
not rewrite it.

Please describe the problem, the change, its test evidence and any effect on the recorded conclusions.
