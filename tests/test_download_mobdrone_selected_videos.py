from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_mobdrone_selected_videos.py"
)
SPEC = importlib.util.spec_from_file_location("download_mobdrone_selected_videos", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_required_stems_are_unique_and_sorted() -> None:
    manifest = {
        "frames": [
            {"source_group": "B"},
            {"source_group": "A"},
            {"source_group": "B"},
        ]
    }
    assert MODULE.required_stems(manifest) == ["A", "B"]


def test_parse_zip64_extra_replaces_sentinel_values() -> None:
    values = struct.pack("<QQQ", 11, 12, 13)
    extra = struct.pack("<HH", 1, len(values)) + values
    assert MODULE.parse_zip64_extra(extra, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == (
        11,
        12,
        13,
    )


def test_zip64_extra_requires_complete_uint64_values() -> None:
    extra = struct.pack("<HH", 1, 3) + b"bad"
    with pytest.raises(ValueError, match="invalid ZIP64"):
        MODULE.parse_zip64_extra(extra, 0xFFFFFFFF, 1, 1)


def test_seed_ranges_uses_only_available_prefix_bytes(tmp_path: Path) -> None:
    prefix = tmp_path / "archive.partial"
    prefix.write_bytes(b"0123456789")
    entries = [
        {"stem": "a", "range_start": 2, "range_bytes": 4},
        {"stem": "b", "range_start": 8, "range_bytes": 5},
        {"stem": "c", "range_start": 10, "range_bytes": 1},
    ]
    result = MODULE.seed_ranges_from_prefix(prefix, entries, tmp_path / "ranges")
    assert result and result["range_bytes_seeded"] == 6
    assert (tmp_path / "ranges" / "a.zip-entry.part").read_bytes() == b"2345"
    assert (tmp_path / "ranges" / "b.zip-entry.part").read_bytes() == b"89"
    assert not (tmp_path / "ranges" / "c.zip-entry.part").exists()
