from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_mobdrone_holdout_frames.py"
SPEC = importlib.util.spec_from_file_location("extract_mobdrone_holdout_frames", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_video_entry_map_is_case_insensitive() -> None:
    assert MODULE.video_entry_map(["folder/A.MP4", "notes/readme.txt"]) == {
        "a": "folder/A.MP4"
    }


def test_video_entry_map_rejects_duplicate_stems() -> None:
    with pytest.raises(ValueError, match="duplicate video stem"):
        MODULE.video_entry_map(["one/A.MP4", "two/a.mov"])


def test_required_sources_sorts_frame_indices() -> None:
    manifest = {
        "frames": [
            {"source_group": "A", "frame_index": 30},
            {"source_group": "A", "frame_index": 0},
            {"source_group": "B", "frame_index": 1},
        ]
    }
    grouped = MODULE.required_sources(manifest)
    assert [row["frame_index"] for row in grouped["A"]] == [0, 30]
