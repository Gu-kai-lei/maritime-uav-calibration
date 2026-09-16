from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def multi_digest(path: Path) -> dict[str, str]:
    hashers = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def video_entry_map(names: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v"}:
            continue
        key = PurePosixPath(name).stem.lower()
        if key in output:
            raise ValueError(f"duplicate video stem in archive: {key}")
        output[key] = name
    return output


def required_sources(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in manifest["frames"]:
        grouped[str(frame["source_group"])].append(frame)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return dict(grouped)


def extract_required_videos(
    archive: Path, grouped: dict[str, list[dict[str, Any]]], video_dir: Path
) -> dict[str, Path]:
    video_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as bundle:
        entries = video_entry_map(bundle.namelist())
        missing = sorted(source for source in grouped if source.lower() not in entries)
        if missing:
            raise ValueError(f"required videos missing from archive: {missing}")
        for source in sorted(grouped):
            entry = entries[source.lower()]
            suffix = PurePosixPath(entry).suffix.lower()
            destination = video_dir / f"{source}{suffix}"
            if not destination.exists():
                with bundle.open(entry) as source_handle, destination.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=4 * 1024 * 1024)
            output[source] = destination
    return output


def videos_from_acquisition(
    acquisition_path: Path, grouped: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Path], dict[str, Any]]:
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("status") != "selected_official_zip_entries_acquired_and_verified":
        raise ValueError("selected-video acquisition is not verified")
    entries = {str(row["stem"]): row for row in acquisition["entries"]}
    missing = sorted(source for source in grouped if source not in entries)
    if missing:
        raise ValueError(f"required videos missing from acquisition: {missing}")
    videos: dict[str, Path] = {}
    for source in sorted(grouped):
        row = entries[source]
        path = Path(str(row["video_path"]))
        if not path.is_file():
            raise ValueError(f"acquired video is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != str(row["video_sha256"]):
            raise ValueError(f"acquired video SHA-256 mismatch: {path}")
        videos[source] = path
    provenance = {
        "selected_acquisition": str(acquisition_path.resolve()),
        "selected_acquisition_sha256": sha256_file(acquisition_path),
        "official_source": acquisition["source"],
    }
    return videos, provenance


def extract_frames(
    grouped: dict[str, list[dict[str, Any]]],
    videos: dict[str, Path],
    frame_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    video_stats: dict[str, Any] = {}
    for source in sorted(grouped):
        requests = {int(row["frame_index"]): row for row in grouped[source]}
        if len(requests) != len(grouped[source]):
            raise ValueError(f"duplicate requested frame index in {source}")
        maximum = max(requests)
        capture = cv2.VideoCapture(str(videos[source]))
        if not capture.isOpened():
            raise ValueError(f"cannot open video {videos[source]}")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        found: set[int] = set()
        index = 0
        while index <= maximum:
            ok, pixels = capture.read()
            if not ok:
                break
            if index in requests:
                row = dict(requests[index])
                height, width = pixels.shape[:2]
                expected = (int(row["width"]), int(row["height"]))
                actual = (width, height)
                if actual != expected:
                    capture.release()
                    raise ValueError(
                        f"frame geometry mismatch for {row['file_name']}: expected {expected}, got {actual}"
                    )
                destination = frame_dir / str(row["file_name"])
                if not destination.exists():
                    if not cv2.imwrite(str(destination), pixels):
                        capture.release()
                        raise ValueError(f"failed to write {destination}")
                row["extracted_path"] = str(destination.resolve())
                row["extracted_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
                row["decoded_from_video"] = str(videos[source].resolve())
                extracted.append(row)
                found.add(index)
            index += 1
        capture.release()
        missing = sorted(set(requests) - found)
        if missing:
            raise ValueError(f"video {source} ended before requested frames {missing[:10]}")
        video_stats[source] = {
            "path": str(videos[source].resolve()),
            "bytes": videos[source].stat().st_size,
            "sha256": sha256_file(videos[source]),
            "reported_frames": reported_frames,
            "reported_width": reported_width,
            "reported_height": reported_height,
            "requested_frames": len(requests),
            "maximum_requested_frame_index": maximum,
        }
    extracted.sort(key=lambda row: str(row["file_name"]))
    return extracted, video_stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "pre_extraction_frozen":
        raise ValueError("input frame manifest is not pre-extraction frozen")
    grouped = required_sources(manifest)
    if args.acquisition:
        videos, source_provenance = videos_from_acquisition(args.acquisition, grouped)
    else:
        if not args.archive or not args.expected_md5 or not args.video_dir:
            raise ValueError(
                "full-archive mode requires --archive, --expected-md5, and --video-dir"
            )
        archive = args.archive.resolve()
        archive_hashes = multi_digest(archive)
        if archive_hashes["md5"].lower() != args.expected_md5.lower():
            raise ValueError(
                f"video archive MD5 mismatch: expected {args.expected_md5}, got {archive_hashes['md5']}"
            )
        videos = extract_required_videos(archive, grouped, args.video_dir.resolve())
        source_provenance = {
            "full_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                **archive_hashes,
            }
        }
    frames, video_stats = extract_frames(grouped, videos, args.frame_dir.resolve())
    output = {
        "schema_version": 1,
        "status": "extracted_and_geometry_verified",
        "source_provenance": source_provenance,
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "derived_video_directory": str(
            args.video_dir.resolve() if args.video_dir else Path(next(iter(videos.values()))).parent
        ),
        "derived_frame_directory": str(args.frame_dir.resolve()),
        "source_videos": video_stats,
        "frames": frames,
        "summary": {
            "source_videos": len(video_stats),
            "frames": len(frames),
            "unique_frame_sha256": len({row["extracted_sha256"] for row in frames}),
            "geometry_mismatches": 0,
            "missing_frames": 0,
        },
        "training_or_inference_performed": False,
        "model_or_prediction_files_accessed": False,
    }
    if output["summary"]["unique_frame_sha256"] != len(frames):
        raise ValueError("duplicate decoded frame content detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-md5")
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2))
