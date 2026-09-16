"""Check public result integrity and tracked-file publication boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "results/artifact_catalog.json"
TEXT = {".json", ".csv", ".yaml", ".yml", ".md", ".py", ".ps1", ".toml", ".cff", ""}
PRIVATE_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.I)
SECRET = re.compile(r"(?:gh[pousr]_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}")
EXPOSED_SHARE = re.compile(
    r"(?im)^\s*(?:official share token|share_token)\s*[:=]\s*[`'\"]?[A-Za-z0-9]{12,}"
)


def candidates() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted(set(name.decode("utf-8") for name in result.stdout.split(b"\0") if name))


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT else data


def public_violations(name: str, data: bytes) -> list[str]:
    errors = []
    if name.startswith(("artifacts/", "runs/", "data/", ".venv/")) or Path(name).suffix in {".pt", ".log", ".parquet", ".joblib"}:
        errors.append(f"local-only artifact tracked: {name}")
    if len(data) > 5 * 1024 * 1024:
        errors.append(f"large file requires explicit artifact policy: {name}")
    if Path(name).suffix in TEXT:
        text = data.decode("utf-8")
        if PRIVATE_PATH.search(text):
            errors.append(f"personal absolute path in {name}")
        if SECRET.search(text):
            errors.append(f"credential-like value in {name}")
        if EXPOSED_SHARE.search(text):
            errors.append(f"embedded dataset share token in {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-catalog", action="store_true", help="Intentional publication refresh after reviewing results")
    args = parser.parse_args()
    files = candidates()
    errors = []
    records = []
    for name in files:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"missing tracked file: {name}")
            continue
        data = canonical_bytes(path)
        errors.extend(public_violations(name, data))
        if name.startswith("results/") and name != CATALOG:
            records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if path.suffix == ".json":
            json.loads(data)
    if errors:
        raise SystemExit("\n".join(errors))
    catalog = {
        "schema_version": 1,
        "hash_scope": "published files; CRLF normalized to LF for text, original bytes for images",
        "artifacts": records,
    }
    target = ROOT / CATALOG
    if args.write_catalog:
        target.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    else:
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved != catalog:
            expected = {r["path"]: r for r in saved["artifacts"]}
            current = {r["path"]: r for r in records}
            changed = sorted(k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k))
            raise SystemExit("Published artifact catalog differs: " + ", ".join(changed))
    print(f"Verified {len(files)} public files and {len(records)} result artifacts.")


if __name__ == "__main__":
    main()
