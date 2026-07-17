import json
from pathlib import Path

from madlibs.scripts.rerun_gpu_mismatches import (
    find_mismatches,
    find_missing_from_right,
    load_records,
    resolve_path,
    rewrite_output_audio_filenames,
    stage_subset,
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


def test_find_missing_from_right_only_returns_missing_keys():
    left = {
        "a.wav": {"metadata": {"audio_filename": "a.wav"}, "tempo": 120},
        "b.wav": {"metadata": {"audio_filename": "b.wav"}, "tempo": 100},
    }
    right = {
        "a.wav": {"metadata": {"audio_filename": "a.wav"}, "tempo": 999},
        "c.wav": {"metadata": {"audio_filename": "c.wav"}, "tempo": 90},
    }

    assert find_missing_from_right(left, right) == ["b.wav"]


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


def test_stage_subset_and_rewrite_output_restore_original_audio_filename(
    tmp_path: Path,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    audio_path = source_dir / "song.wav"
    prompt_path = source_dir / "prompt.txt"
    audio_path.write_text("audio", encoding="utf-8")
    prompt_path.write_text("prompt", encoding="utf-8")

    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    staged_to_original = stage_subset(
        audio_filenames=["source/song.wav"],
        base_dir=tmp_path,
        prompt_filename="prompt.txt",
        subset_dir=subset_dir,
    )

    staged_audio = next(iter(staged_to_original))
    output_path = tmp_path / "gpu_analysis_rerun.jsonl"
    output_record = {
        "metadata": {"audio_filename": staged_audio},
        "tempo": 120,
    }
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    rewrite_output_audio_filenames(output_path, staged_to_original)

    rewritten = load_records(output_path)
    assert list(rewritten) == ["source/song.wav"]
