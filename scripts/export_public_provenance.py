"""Prepare public provenance without publishing weights or changing numerical results.

Explicit --apply backs up and normalizes personal paths in existing result text.
This is a publication utility, not part of the training pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_text(text: str) -> str:
    for source, replacement in ((ROOT, "."), (Path.home(), "<USER_HOME>")):
        variants = {str(source), source.as_posix(), str(source).replace("\\", "\\\\")}
        for variant in sorted(variants, key=len, reverse=True):
            text = text.replace(variant, replacement)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("pass --apply to export provenance and normalize backed-up result files")
    backup = ROOT / "artifacts/publication_originals"
    changed = []
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".csv"}:
            continue
        original = path.read_text(encoding="utf-8")
        normalized = portable_text(original)
        if normalized != original:
            saved = backup / path.relative_to(ROOT)
            saved.parent.mkdir(parents=True, exist_ok=True)
            if not saved.exists():
                shutil.copy2(path, saved)
            path.write_text(normalized, encoding="utf-8")
            changed.append(path.relative_to(ROOT).as_posix())
    records = []
    for manifest in sorted((ROOT / "runs/detect").glob("*/run_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        run = manifest.parent
        best = run / "weights/best.pt"
        actual = sha256(best) if best.exists() else None
        expected = data.get("best_weights_sha256")
        if actual and expected and actual != expected:
            raise ValueError(f"best.pt differs from recorded manifest: {run.name}")
        output = ROOT / "results/training_runs" / run.name
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_manifest.json").write_text(
            portable_text(json.dumps(data, indent=2)) + "\n", encoding="utf-8"
        )
        if (run / "results.csv").exists():
            shutil.copy2(run / "results.csv", output / "results.csv")
        records.append({
            "run": run.name,
            "kind": "preflight" if "preflight" in run.name else "experiment",
            "manifest": (output / "run_manifest.json").relative_to(ROOT).as_posix(),
            "best_weights_local_path": best.relative_to(ROOT).as_posix(),
            "best_weights_sha256_recorded": expected,
            "best_weights_sha256_observed": actual,
            "best_weights_matches_manifest": actual == expected if actual and expected else None,
            "weights_distributed": False,
            "epochs_completed": data.get("epochs_completed"),
            "imgsz": data.get("imgsz"),
            "batch_effective": data.get("batch_effective"),
            "amp": data.get("amp"),
            "resumed": data.get("resumed", False),
        })
    result = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local best.pt hashes compared with historical manifests; not a new inference run",
        "runs": records,
    }
    (ROOT / "results/checkpoint_catalog.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(records)} run manifests; normalized {len(changed)} files with backups.")


if __name__ == "__main__":
    main()
