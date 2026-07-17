import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        raw_records = _load_jsonl(path)
    elif suffix == ".json":
        raw_records = _load_json(path)
    else:
        raise ValueError(f"Unsupported file type for {path}; expected .json or .jsonl")

    for record in raw_records:
        key = get_audio_filename(record)
        if key in records:
            raise ValueError(f"Duplicate audio_filename in {path}: {key}")
        records[key] = record

    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_num} of {path}")
            records.append(record)
    return records


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Expected a list of objects in {path}")
        return payload

    if isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
        raise ValueError(
            f"Expected a list of objects or mapping of objects in {path}, got dict"
        )

    raise ValueError(f"Unsupported JSON payload in {path}")


def get_audio_filename(record: dict[str, Any]) -> str:
    audio_filename = record.get("audio_filename")
    if isinstance(audio_filename, str):
        return audio_filename

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        metadata_audio_filename = metadata.get("audio_filename")
        if isinstance(metadata_audio_filename, str):
            return metadata_audio_filename

    raise ValueError("Record is missing 'audio_filename' or 'metadata.audio_filename'")


def find_mismatches(
    left_records: dict[str, dict[str, Any]],
    right_records: dict[str, dict[str, Any]],
) -> list[str]:
    mismatch_keys: set[str] = set()
    shared_keys = set(left_records) & set(right_records)

    mismatch_keys.update(set(left_records) ^ set(right_records))

    for key in shared_keys:
        if normalize_record(left_records[key]) != normalize_record(right_records[key]):
            mismatch_keys.add(key)

    return sorted(mismatch_keys)


def find_missing_from_right(
    left_records: dict[str, dict[str, Any]],
    right_records: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(set(left_records) - set(right_records))


def normalize_record(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    base_candidate = (base_dir / candidate).resolve()
    return base_candidate


def stage_subset(
    audio_filenames: list[str],
    base_dir: Path,
    prompt_filename: str,
    subset_dir: Path,
) -> dict[str, str]:
    staged_to_original: dict[str, str] = {}
    for index, audio_filename in enumerate(audio_filenames):
        audio_path = resolve_path(audio_filename, base_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        prompt_path = audio_path.parent / prompt_filename
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file does not exist for {audio_path}: {prompt_path}"
            )

        target_dir = subset_dir / f"{index:05d}"
        target_dir.mkdir(parents=True, exist_ok=False)

        staged_audio = target_dir / f"audio{audio_path.suffix}"
        staged_prompt = target_dir / prompt_filename

        os.symlink(audio_path, staged_audio)
        os.symlink(prompt_path, staged_prompt)
        staged_to_original[str(staged_audio)] = audio_filename

    return staged_to_original


def rewrite_output_audio_filenames(
    output_path: Path,
    staged_to_original: dict[str, str],
) -> None:
    rewritten_lines: list[str] = []
    with output_path.open("r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Record on line {line_num} of {output_path} is missing metadata"
                )

            staged_audio_filename = metadata.get("audio_filename")
            if not isinstance(staged_audio_filename, str):
                raise ValueError(
                    f"Record on line {line_num} of {output_path} is missing "
                    "metadata.audio_filename"
                )

            original_audio_filename = staged_to_original.get(staged_audio_filename)
            if original_audio_filename is None:
                raise ValueError(
                    f"Unexpected staged audio path on line {line_num} of "
                    f"{output_path}: {staged_audio_filename}"
                )

            metadata["audio_filename"] = original_audio_filename
            rewritten_lines.append(json.dumps(record, ensure_ascii=False))

    output_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def build_command(
    repo_root: Path,
    subset_dir: Path,
    output_path: Path,
    hydra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "run_mir.py"),
        "analysis_type=gpu",
        f"run.audio_glob={subset_dir}/*/audio*",
        f"analysis_type.output_filename={output_path}",
    ]
    command.extend(hydra_args)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two JSON/JSONL files by audio_filename and rerun GPU analysis "
            "only on mismatching records."
        )
    )
    parser.add_argument(
        "directory",
        help="Base directory used to resolve relative JSON and audio paths",
    )
    parser.add_argument(
        "--file1",
        required=True,
        help="First JSON/JSONL file, absolute or relative to directory",
    )
    parser.add_argument(
        "--file2",
        required=True,
        help="Second JSON/JSONL file, absolute or relative to directory",
    )
    parser.add_argument(
        "--output",
        default="gpu_analysis_rerun.jsonl",
        help="Output JSONL path, absolute or relative to directory",
    )
    parser.add_argument(
        "--prompt-filename",
        default="prompt.txt",
        help="Sibling prompt filename expected next to each audio file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mismatches and planned command without running analysis",
    )
    parser.add_argument(
        "--mode",
        choices=["missing_from_file2", "mismatched", "different_or_missing"],
        default="missing_from_file2",
        help=(
            "How to select rerun files. Default reruns files present in file1 but "
            "missing from file2."
        ),
    )
    args, hydra_args = parser.parse_known_args()
    if hydra_args and hydra_args[0] == "--":
        hydra_args = hydra_args[1:]
    args.hydra_args = hydra_args
    return args


def main() -> None:
    args = parse_args()
    base_dir = Path(args.directory).resolve()
    file1 = resolve_path(args.file1, base_dir)
    file2 = resolve_path(args.file2, base_dir)
    output_path = resolve_path(args.output, base_dir)

    hydra_args = list(args.hydra_args)

    left_records = load_records(file1)
    right_records = load_records(file2)
    if args.mode == "missing_from_file2":
        mismatch_keys = find_missing_from_right(left_records, right_records)
        summary_label = "missing from file2"
    elif args.mode == "mismatched":
        mismatch_keys = find_mismatches(left_records, right_records)
        summary_label = "mismatching"
    else:
        mismatch_keys = find_mismatches(left_records, right_records)
        summary_label = "different or missing"

    print(f"Found {len(mismatch_keys)} audio files {summary_label}")
    if mismatch_keys:
        preview = "\n".join(f"  {key}" for key in mismatch_keys[:20])
        print(preview)
        if len(mismatch_keys) > 20:
            print(f"  ... and {len(mismatch_keys) - 20} more")

    if not mismatch_keys:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        print(f"No mismatches found; wrote empty file to {output_path}")
        return

    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="gpu_rerun_subset_") as temp_dir:
        subset_dir = Path(temp_dir)
        staged_to_original = stage_subset(
            audio_filenames=mismatch_keys,
            base_dir=base_dir,
            prompt_filename=args.prompt_filename,
            subset_dir=subset_dir,
        )

        command = build_command(
            repo_root=repo_root,
            subset_dir=subset_dir,
            output_path=output_path,
            hydra_args=hydra_args,
        )
        print("Running command:")
        print(" ".join(shlex_quote(arg) for arg in command))

        if args.dry_run:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, cwd=repo_root)
        rewrite_output_audio_filenames(output_path, staged_to_original)

        print(f"Wrote rerun GPU analysis to {output_path}")


def shlex_quote(value: str) -> str:
    if value == "":
        return "''"
    if all(ch.isalnum() or ch in "@%_-+=:,./*" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
