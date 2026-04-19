import json
from pathlib import Path

from madlibs.scripts.rerun_gpu_mismatches import (
    find_mismatches,
    load_records,
    resolve_path,
)


def test_find_mismatches_includes_missing_and_changed_records():
    left = {
        "a.wav": {"metadata": {"audio_filename": "a.wav"}, "tempo": 120},
        "b.wav": {"metadata": {"audio_filename": "b.wav"}, "tempo": 100},
    }
    right = {
        "a.wav": {"metadata": {"audio_filename": "a.wav"}, "tempo": 121},
        "c.wav": {"metadata": {"audio_filename": "c.wav"}, "tempo": 90},
    }

    assert find_mismatches(left, right) == ["a.wav", "b.wav", "c.wav"]


def test_load_records_supports_top_level_audio_filename_in_json(tmp_path: Path):
    path = tmp_path / "records.json"
    payload = [
        {"audio_filename": "one.wav", "tempo": 1},
        {"audio_filename": "two.wav", "tempo": 2},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_records(path)

    assert sorted(records) == ["one.wav", "two.wav"]


def test_resolve_path_prefers_base_dir_for_missing_relative_path(tmp_path: Path):
    nested = tmp_path / "audio" / "song.wav"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")

    resolved = resolve_path("audio/song.wav", tmp_path)

    assert resolved == nested.resolve()
