from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CENTRAL_SIGNATURE = b"PK\x01\x02"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
LOCAL_SIGNATURE = b"PK\x03\x04"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_zip64_extra(
    extra: bytes, uncompressed: int, compressed: int, local_offset: int
) -> tuple[int, int, int]:
    position = 0
    values: list[int] = []
    while position + 4 <= len(extra):
        identifier, size = struct.unpack_from("<HH", extra, position)
        data = extra[position + 4 : position + 4 + size]
        position += 4 + size
        if identifier == 1:
            if len(data) % 8:
                raise ValueError("invalid ZIP64 extra field")
            values = [item[0] for item in struct.iter_unpack("<Q", data)]
            break
    iterator = iter(values)
    if uncompressed == 0xFFFFFFFF:
        uncompressed = next(iterator)
    if compressed == 0xFFFFFFFF:
        compressed = next(iterator)
    if local_offset == 0xFFFFFFFF:
        local_offset = next(iterator)
    return uncompressed, compressed, local_offset


def parse_central_directory(tail: bytes, archive_bytes: int) -> list[dict[str, Any]]:
    base = archive_bytes - len(tail)
    eocd_position = tail.rfind(ZIP64_EOCD_SIGNATURE)
    if eocd_position < 0:
        raise ValueError("ZIP64 end-of-central-directory record not found in tail")
    fields = struct.unpack_from("<4sQ2H2L4Q", tail, eocd_position)
    entry_count = int(fields[7])
    central_size = int(fields[8])
    central_offset = int(fields[9])
    if central_offset < base or central_offset + central_size > archive_bytes:
        raise ValueError("central directory is not fully contained in the supplied tail")
    position = central_offset - base
    output: list[dict[str, Any]] = []
    for _ in range(entry_count):
        if tail[position : position + 4] != CENTRAL_SIGNATURE:
            raise ValueError("invalid central-directory entry signature")
        fields = struct.unpack_from("<4s6H3L5H2L", tail, position)
        method = int(fields[4])
        crc32 = int(fields[7])
        compressed = int(fields[8])
        uncompressed = int(fields[9])
        name_length = int(fields[10])
        extra_length = int(fields[11])
        comment_length = int(fields[12])
        local_offset = int(fields[16])
        name_start = position + 46
        name = tail[name_start : name_start + name_length].decode("utf-8")
        extra = tail[
            name_start + name_length : name_start + name_length + extra_length
        ]
        uncompressed, compressed, local_offset = parse_zip64_extra(
            extra, uncompressed, compressed, local_offset
        )
        output.append(
            {
                "name": name,
                "stem": PurePosixPath(name).stem,
                "compression_method": method,
                "crc32": crc32,
                "compressed_bytes": compressed,
                "uncompressed_bytes": uncompressed,
                "local_header_offset": local_offset,
            }
        )
        position += 46 + name_length + extra_length + comment_length
    ordered = sorted(output, key=lambda row: int(row["local_header_offset"]))
    for index, row in enumerate(ordered):
        next_offset = (
            int(ordered[index + 1]["local_header_offset"])
            if index + 1 < len(ordered)
            else central_offset
        )
        row["range_start"] = int(row["local_header_offset"])
        row["range_end"] = next_offset - 1
        row["range_bytes"] = next_offset - int(row["local_header_offset"])
    return ordered


def required_stems(manifest: dict[str, Any]) -> list[str]:
    return sorted({str(row["source_group"]) for row in manifest["frames"]})


def download_range(
    url: str, entry: dict[str, Any], range_dir: Path, retries: int
) -> Path:
    destination = range_dir / f"{entry['stem']}.zip-entry.part"
    expected = int(entry["range_bytes"])
    range_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        existing = destination.stat().st_size if destination.exists() else 0
        if existing == expected:
            return destination
        if existing > expected:
            raise ValueError(f"range file exceeds expected size: {destination}")
        start = int(entry["range_start"]) + existing
        end = int(entry["range_end"])
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 600),
            ) as response:
                if response.status_code != 206:
                    raise ValueError(
                        f"expected HTTP 206 for {entry['name']}, got {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ValueError(
                        f"unexpected Content-Range for {entry['name']}: {content_range!r}"
                    )
                with destination.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if destination.stat().st_size == expected:
                print(f"downloaded {entry['stem']} ({expected / 1e6:.1f} MB)", flush=True)
                return destination
        except (requests.RequestException, OSError) as error:
            if attempt == retries:
                raise
            print(
                f"retry {attempt}/{retries} for {entry['stem']}: {error}", flush=True
            )
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"failed to download {entry['name']}")


def seed_ranges_from_prefix(
    partial_prefix: Path | None, entries: list[dict[str, Any]], range_dir: Path
) -> dict[str, Any] | None:
    if partial_prefix is None:
        return None
    partial_prefix = partial_prefix.resolve()
    available_bytes = partial_prefix.stat().st_size
    seeded = 0
    range_dir.mkdir(parents=True, exist_ok=True)
    with partial_prefix.open("rb") as source:
        for entry in entries:
            start = int(entry["range_start"])
            if start >= available_bytes:
                continue
            count = min(int(entry["range_bytes"]), available_bytes - start)
            destination = range_dir / f"{entry['stem']}.zip-entry.part"
            if destination.exists():
                continue
            source.seek(start)
            remaining = count
            with destination.open("wb") as target:
                while remaining:
                    chunk = source.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("partial archive ended before its reported size")
                    target.write(chunk)
                    remaining -= len(chunk)
            seeded += count
    return {
        "path": str(partial_prefix),
        "bytes": available_bytes,
        "sha256": sha256_file(partial_prefix),
        "range_bytes_seeded": seeded,
    }


def verify_existing_video(path: Path, entry: dict[str, Any]) -> bool:
    if not path.exists() or path.stat().st_size != int(entry["uncompressed_bytes"]):
        return False
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            crc = binascii.crc32(chunk, crc)
    return crc & 0xFFFFFFFF == int(entry["crc32"])


def decompress_entry(segment: Path, entry: dict[str, Any], video_dir: Path) -> dict[str, Any]:
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = PurePosixPath(str(entry["name"])).suffix.lower()
    destination = video_dir / f"{entry['stem']}{suffix}"
    if not verify_existing_video(destination, entry):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with segment.open("rb") as source:
            header = source.read(30)
            if len(header) != 30 or header[:4] != LOCAL_SIGNATURE:
                raise ValueError(f"invalid local header for {entry['name']}")
            local = struct.unpack("<4s5H3L2H", header)
            method = int(local[3])
            name_length = int(local[9])
            extra_length = int(local[10])
            if method != int(entry["compression_method"]):
                raise ValueError(f"compression method mismatch for {entry['name']}")
            source.seek(name_length + extra_length, os.SEEK_CUR)
            remaining = int(entry["compressed_bytes"])
            decompressor = zlib.decompressobj(-15) if method == 8 else None
            crc = 0
            written = 0
            with temporary.open("wb") as target:
                while remaining:
                    chunk = source.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated compressed payload for {entry['name']}")
                    remaining -= len(chunk)
                    decoded = decompressor.decompress(chunk) if decompressor else chunk
                    if decoded:
                        target.write(decoded)
                        written += len(decoded)
                        crc = binascii.crc32(decoded, crc)
                if decompressor:
                    decoded = decompressor.flush()
                    target.write(decoded)
                    written += len(decoded)
                    crc = binascii.crc32(decoded, crc)
        if written != int(entry["uncompressed_bytes"]):
            raise ValueError(f"uncompressed byte count mismatch for {entry['name']}")
        if crc & 0xFFFFFFFF != int(entry["crc32"]):
            raise ValueError(f"CRC32 mismatch for {entry['name']}")
        temporary.replace(destination)
    return {
        **entry,
        "range_file": str(segment.resolve()),
        "video_path": str(destination.resolve()),
        "video_sha256": sha256_file(destination),
        "crc32_verified": True,
        "uncompressed_bytes_verified": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tail = args.tail.read_bytes()
    entries = parse_central_directory(tail, args.archive_bytes)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = required_stems(manifest)
    by_stem = {str(row["stem"]).lower(): row for row in entries}
    missing = [stem for stem in required if stem.lower() not in by_stem]
    if missing:
        raise ValueError(f"required video entries absent from ZIP central directory: {missing}")
    selected = [by_stem[stem.lower()] for stem in required]
    args.range_dir.mkdir(parents=True, exist_ok=True)
    partial_prefix = seed_ranges_from_prefix(args.partial_prefix, selected, args.range_dir)
    acquired: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_range, args.url, entry, args.range_dir, args.retries): entry
            for entry in selected
        }
        for future in as_completed(futures):
            entry = futures[future]
            segment = future.result()
            acquired.append(decompress_entry(segment, entry, args.video_dir))
            print(f"verified {entry['stem']}", flush=True)
    acquired.sort(key=lambda row: str(row["stem"]))
    output = {
        "schema_version": 1,
        "status": "selected_official_zip_entries_acquired_and_verified",
        "source": {
            "url": args.url,
            "zenodo_record": "10.5281/zenodo.5996890",
            "full_archive_bytes": args.archive_bytes,
            "full_archive_expected_md5_not_locally_verified": "250c19fdd6fb0b5aa5fa43e5ad140306",
            "zip_tail": str(args.tail.resolve()),
            "zip_tail_sha256": hashlib.sha256(tail).hexdigest(),
            "partial_archive_prefix": partial_prefix,
        },
        "selection_manifest": str(args.manifest.resolve()),
        "selection_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "entries": acquired,
        "summary": {
            "central_directory_entries": len(entries),
            "selected_entries": len(acquired),
            "selected_compressed_bytes": sum(int(row["compressed_bytes"]) for row in acquired),
            "selected_uncompressed_bytes": sum(
                int(row["uncompressed_bytes"]) for row in acquired
            ),
            "all_crc32_verified": all(bool(row["crc32_verified"]) for row in acquired),
            "all_uncompressed_bytes_verified": all(
                bool(row["uncompressed_bytes_verified"]) for row in acquired
            ),
        },
        "training_or_inference_performed": False,
        "model_or_prediction_files_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive-bytes", type=int, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partial-prefix", type=Path)
    parser.add_argument("--range-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
