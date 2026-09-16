from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DAV_BASE = "https://cloud.cs.uni-tuebingen.de/public.php/dav/files"
DAV = "{DAV:}"
USER_AGENT = "maritime-uav-calibration/0.1 (academic reproducibility download)"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    bytes: int
    etag: str


def request_bytes(request: urllib.request.Request, timeout: int) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_remote_split(split: str, timeout: int, dav_root: str) -> list[RemoteFile]:
    url = f"{dav_root}/{urllib.parse.quote(split)}/"
    request = urllib.request.Request(
        url,
        method="PROPFIND",
        headers={"Depth": "1", "User-Agent": USER_AGENT},
    )
    root = ET.fromstring(request_bytes(request, timeout))
    files: list[RemoteFile] = []
    for response in root.findall(f"{DAV}response"):
        href = response.findtext(f"{DAV}href")
        length = response.findtext(f".//{DAV}getcontentlength")
        resource_type = response.find(f".//{DAV}resourcetype")
        if not href or length is None:
            continue
        if resource_type is not None and resource_type.find(f"{DAV}collection") is not None:
            continue
        name = urllib.parse.unquote(urllib.parse.urlsplit(href).path.rsplit("/", 1)[-1])
        etag = (response.findtext(f".//{DAV}getetag") or "").strip('"')
        files.append(RemoteFile(name=name, bytes=int(length), etag=etag))
    if not files:
        raise RuntimeError(f"official WebDAV returned no files for split={split}")
    if len({item.name for item in files}) != len(files):
        raise RuntimeError(f"official WebDAV returned duplicate names for split={split}")
    return sorted(files, key=lambda item: item.name)


def annotation_file_names(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    return {str(image["file_name"]) for image in dataset["images"]}


def verify_jpeg(path: Path, expected_bytes: int) -> None:
    if path.stat().st_size != expected_bytes:
        raise IOError(
            f"size mismatch for {path.name}: {path.stat().st_size} != {expected_bytes}"
        )
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, os.SEEK_END)
        end = handle.read(2)
    if start != b"\xff\xd8" or end != b"\xff\xd9":
        raise IOError(f"invalid JPEG boundary markers for {path.name}")


def download_one(
    split: str,
    remote: RemoteFile,
    target_dir: Path,
    timeout: int,
    retries: int,
    dav_root: str,
) -> str:
    final_path = target_dir / remote.name
    partial_path = target_dir / f"{remote.name}.partial"
    if final_path.exists():
        verify_jpeg(final_path, remote.bytes)
        return "skipped"
    if partial_path.exists():
        partial_path.unlink()

    url = f"{dav_root}/{urllib.parse.quote(split)}/{urllib.parse.quote(remote.name)}"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            verify_jpeg(partial_path, remote.bytes)
            os.replace(partial_path, final_path)
            return "downloaded"
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt >= retries:
                raise
            if partial_path.exists():
                partial_path.unlink()
            time.sleep(min(2**attempt, 30))
    raise AssertionError("retry loop terminated unexpectedly")


def download_split(
    split: str,
    target_root: Path,
    workers: int,
    timeout: int,
    retries: int,
    dav_root: str,
) -> dict[str, int | str]:
    annotations = target_root / "compressed" / "annotations" / f"instances_{split}.json"
    if not annotations.is_file():
        raise FileNotFoundError(f"required official annotations are missing: {annotations}")
    remote_files = list_remote_split(split, timeout, dav_root)
    remote_names = {item.name for item in remote_files}
    annotated_names = annotation_file_names(annotations)
    if remote_names != annotated_names:
        missing_remote = sorted(annotated_names - remote_names)[:5]
        extra_remote = sorted(remote_names - annotated_names)[:5]
        raise RuntimeError(
            "official image listing and annotations differ: "
            f"missing_remote={missing_remote}, extra_remote={extra_remote}"
        )

    manifest_dir = target_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split}_webdav_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": "official SeaDronesSee WebDAV image listing",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": [asdict(item) for item in remote_files],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    target_dir = target_root / "compressed" / "images" / split
    target_dir.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0}
    failures: list[str] = []
    lock = threading.Lock()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, split, item, target_dir, timeout, retries, dav_root): item
            for item in remote_files
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                outcome = future.result()
                with lock:
                    counts[outcome] += 1
            except Exception as error:  # keep other independent downloads running
                failures.append(f"{item.name}: {error}")
            if completed % 100 == 0 or completed == len(futures):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{split}] {completed}/{len(futures)} files; "
                    f"downloaded={counts['downloaded']} skipped={counts['skipped']} "
                    f"failed={len(failures)} rate={completed / elapsed:.2f} files/s",
                    flush=True,
                )
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"{len(failures)} downloads failed; rerun to resume completed files:\n{preview}"
        )
    return {
        "split": split,
        "files": len(remote_files),
        "bytes": sum(item.bytes for item in remote_files),
        **counts,
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reliably download official SeaDronesSee ODv2 JPEGs over public WebDAV"
    )
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    share_token = os.environ.get("SEADRONESSEE_SHARE_TOKEN", "").strip()
    if not share_token:
        parser.error("set SEADRONESSEE_SHARE_TOKEN from the official dataset host")
    if "/" in share_token or "\\" in share_token or "?" in share_token:
        parser.error("SEADRONESSEE_SHARE_TOKEN must be a single path token")
    dav_root = f"{DAV_BASE}/{urllib.parse.quote(share_token)}/Compressed%20Version/images"

    args.target_root.mkdir(parents=True, exist_ok=True)
    reports = [
        download_split(split, args.target_root, args.workers, args.timeout, args.retries, dav_root)
        for split in args.splits
    ]
    output = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "splits": reports}
    summary_path = args.target_root / "manifests" / "download_summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
