import sys
import os
import json
import csv
import argparse
import glob
from collections import Counter
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="MADLibs_output")
    parser.add_argument("--output_dir", default="analysis_data")
    parser.add_argument("--include_strategies", nargs="*", default=None)
    parser.add_argument(
        "--exclude_strategies",
        nargs="*",
        default=["killed_run_partial_results", "test"]
    )
    parser.add_argument(
        "--export",
        choices=["mir", "gpu", "both"],
        default="both"
    )
    args = parser.parse_args()

    include_strategies = normalize_name_list(args.include_strategies)
    exclude_strategies = normalize_name_list(args.exclude_strategies)

    mir_rows, gpu_rows, coverage_rows = collect_rows(
        input_root=args.input_root,
        include_strategies=include_strategies,
        exclude_strategies=exclude_strategies
    )

    if not mir_rows:
        print("No MIR rows found under %s" % args.input_root)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.export in ["mir", "both"]:
        mir_output = os.path.join(args.output_dir, "mir_features.csv")
        write_csv(
            rows=mir_rows,
            output_path=mir_output,
            fieldnames=get_mir_fieldnames()
        )
        print("Wrote %d MIR rows -> %s" % (len(mir_rows), mir_output))

    if args.export in ["gpu", "both"]:
        gpu_output = os.path.join(args.output_dir, "gpu_features.csv")
        write_csv(
            rows=gpu_rows,
            output_path=gpu_output,
            fieldnames=get_gpu_fieldnames()
        )
        print("Wrote %d GPU rows -> %s" % (len(gpu_rows), gpu_output))

    coverage_output = os.path.join(args.output_dir, "coverage_summary.csv")
    write_csv(
        rows=coverage_rows,
        output_path=coverage_output,
        fieldnames=get_coverage_fieldnames()
    )
    print("Wrote %d coverage rows -> %s" % (len(coverage_rows), coverage_output))


def normalize_name_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None

    normalized = []
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        normalized.append(stripped)

    return normalized


def collect_rows(
    input_root: str,
    include_strategies: list[str] | None,
    exclude_strategies: list[str] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mir_paths = discover_result_paths(
        input_root=input_root,
        filename="mir_results.jsonl",
        include_strategies=include_strategies,
        exclude_strategies=exclude_strategies
    )
    gpu_paths = discover_result_paths(
        input_root=input_root,
        filename="gpu_results.jsonl",
        include_strategies=include_strategies,
        exclude_strategies=exclude_strategies
    )

    mir_rows = []
    gpu_rows = []
    mir_counts = {}
    gpu_counts = {}

    for path in mir_paths:
        strategy, model = extract_strategy_model(path)
        rows = load_rows(
            jsonl_path=path,
            row_builder=build_mir_row,
            strategy=strategy,
            model=model
        )
        mir_rows.extend(rows)
        mir_counts[(strategy, model)] = len(rows)

    for path in gpu_paths:
        strategy, model = extract_strategy_model(path)
        rows = load_rows(
            jsonl_path=path,
            row_builder=build_gpu_row,
            strategy=strategy,
            model=model
        )
        gpu_rows.extend(rows)
        gpu_counts[(strategy, model)] = len(rows)

    coverage_rows = build_coverage_rows(
        mir_counts=mir_counts,
        gpu_counts=gpu_counts
    )

    return mir_rows, gpu_rows, coverage_rows


def discover_result_paths(
    input_root: str,
    filename: str,
    include_strategies: list[str] | None,
    exclude_strategies: list[str] | None
) -> list[str]:
    pattern = os.path.join(input_root, "*", "*", "mir", filename)
    paths = sorted(glob.glob(pattern))
    filtered_paths = []

    for path in paths:
        strategy, _model = extract_strategy_model(path)

        if include_strategies is not None and strategy not in include_strategies:
            continue

        if exclude_strategies is not None and strategy in exclude_strategies:
            continue

        filtered_paths.append(path)

    return filtered_paths


def extract_strategy_model(path: str) -> tuple[str, str]:
    parts = path.split(os.sep)
    assert len(parts) >= 4, "Unexpected path: %s" % path
    return parts[1], parts[2]


def load_rows(
    jsonl_path: str,
    row_builder: Any,
    strategy: str,
    model: str
) -> list[dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                row_builder(
                    record=record,
                    strategy=strategy,
                    model=model,
                    line_number=line_number
                )
            )
    return rows


def build_mir_row(
    record: dict[str, Any],
    strategy: str,
    model: str,
    line_number: int
) -> dict[str, Any]:
    identifiers = extract_identifiers(
        audio_filename=record["metadata"]["audio_filename"],
        strategy=strategy,
        model=model
    )
    chromasum = record["chromasum"]

    row = {
        "strategy": strategy,
        "model": model,
        "line_number": line_number,
        "join_key": identifiers["join_key"],
        "audio_filename": identifiers["audio_filename"],
        "audio_basename": identifiers["audio_basename"],
        "audio_stem": identifiers["audio_stem"],
        "audio_id": identifiers["audio_id"],
        "prompt_id": identifiers["prompt_id"],
        "cluster_id": identifiers["cluster_id"],
        "run_id": identifiers["run_id"],
        "rmsstd": record["rmsstd"],
        "longest_silence_ms": record["longest_silence_ms"],
        "nonzero_silence": int(record["longest_silence_ms"] > 0)
    }

    for index, value in enumerate(chromasum):
        row["chromasum_%d" % index] = value

    return row


def build_gpu_row(
    record: dict[str, Any],
    strategy: str,
    model: str,
    line_number: int
) -> dict[str, Any]:
    identifiers = extract_identifiers(
        audio_filename=record["metadata"]["audio_filename"],
        strategy=strategy,
        model=model
    )
    instruments = normalize_instrument_list(record.get("instruments", []))
    key_names = normalize_string_list(record.get("key_info", {}).get("keys", []))
    key_positions = normalize_number_list(record.get("key_info", {}).get("positions", []))
    similarity_scores = normalize_number_list(record.get("similarity_scores", []))
    canonical_song_key = select_song_key(key_names)

    return {
        "strategy": strategy,
        "model": model,
        "line_number": line_number,
        "join_key": identifiers["join_key"],
        "audio_filename": identifiers["audio_filename"],
        "audio_basename": identifiers["audio_basename"],
        "audio_stem": identifiers["audio_stem"],
        "audio_id": identifiers["audio_id"],
        "prompt_id": identifiers["prompt_id"],
        "cluster_id": identifiers["cluster_id"],
        "run_id": identifiers["run_id"],
        "tempo": record.get("tempo"),
        "time_signature": record.get("time_signature"),
        "instruments": "|".join(instruments),
        "n_instruments": len(instruments),
        "raw_keys": "|".join(key_names),
        "key_positions": ",".join([str(value) for value in key_positions]),
        "canonical_song_key": canonical_song_key,
        "similarity_scores": ",".join([str(value) for value in similarity_scores]),
        "mean_similarity": mean_or_blank(similarity_scores)
    }


def extract_identifiers(
    audio_filename: str,
    strategy: str,
    model: str
) -> dict[str, str]:
    normalized_path = audio_filename.replace("\\", "/")
    parts = [part for part in normalized_path.split("/") if part]
    audio_basename = os.path.basename(normalized_path)
    audio_stem = os.path.splitext(audio_basename)[0]
    prompt_id = ""
    run_id = ""

    for index, part in enumerate(parts):
        if part == "generated" and index + 1 < len(parts):
            run_id = parts[index + 1]
        if part.startswith("prompt_"):
            prompt_id = part

    if strategy == "sdd" and model == "real":
        prompt_id = audio_stem

    if not prompt_id:
        prompt_id = audio_stem

    if strategy == "sdd" and model == "real":
        cluster_id = "song_%s" % prompt_id
        audio_id = audio_stem
    else:
        cluster_id = "%s_%s" % (strategy, prompt_id)
        audio_id = "%s__%s" % (prompt_id, audio_stem)

    return {
        "join_key": normalized_path,
        "audio_filename": normalized_path,
        "audio_basename": audio_basename,
        "audio_stem": audio_stem,
        "audio_id": audio_id,
        "prompt_id": prompt_id,
        "cluster_id": cluster_id,
        "run_id": run_id
    }


def normalize_instrument_list(values: list[Any]) -> list[str]:
    normalized = []
    seen = set()

    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)

    return normalized


def normalize_string_list(values: list[Any]) -> list[str]:
    normalized = []
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def normalize_number_list(values: list[Any]) -> list[float]:
    normalized = []
    for value in values:
        if isinstance(value, int) or isinstance(value, float):
            normalized.append(value)
    return normalized


def mean_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def normalize_note_name(note: str) -> str:
    return note.strip().upper().replace("♯", "#").replace("♭", "B")


def canonicalize_key_name(key_name: str) -> str | None:
    note_to_pitch_class = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "FB": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
        "CB": 11
    }
    relative_key_labels = {
        0: "C",
        1: "Db",
        2: "D",
        3: "Eb",
        4: "E",
        5: "F",
        6: "Gb",
        7: "G",
        8: "Ab",
        9: "A",
        10: "Bb",
        11: "B"
    }

    cleaned = key_name.strip()
    if not cleaned:
        return None

    parts = cleaned.split()
    if len(parts) < 2:
        return None

    tonic = normalize_note_name(parts[0])
    mode = parts[-1].lower()
    pitch_class = note_to_pitch_class.get(tonic)

    if pitch_class is None:
        return None

    if mode == "major":
        return "%s Major" % relative_key_labels[pitch_class]
    if mode == "minor":
        return "%s minor" % relative_key_labels[pitch_class]

    return None


def select_song_key(key_names: list[str]) -> str:
    canonical_keys = []
    for key_name in key_names:
        canonical_key = canonicalize_key_name(key_name)
        if canonical_key is not None:
            canonical_keys.append(canonical_key)

    if not canonical_keys:
        return ""

    counts = Counter(canonical_keys)
    max_count = max(counts.values())

    for canonical_key in canonical_keys:
        if counts[canonical_key] == max_count:
            return canonical_key

    return ""


def build_coverage_rows(
    mir_counts: dict[tuple[str, str], int],
    gpu_counts: dict[tuple[str, str], int]
) -> list[dict[str, Any]]:
    keys = sorted(set(mir_counts.keys()) | set(gpu_counts.keys()))
    rows = []

    for strategy, model in keys:
        mir_count = mir_counts.get((strategy, model), 0)
        gpu_count = gpu_counts.get((strategy, model), 0)
        coverage = ""
        if mir_count > 0:
            coverage = gpu_count / mir_count

        rows.append(
            {
                "strategy": strategy,
                "model": model,
                "mir_rows": mir_count,
                "gpu_rows": gpu_count,
                "gpu_coverage": coverage
            }
        )

    return rows


def write_csv(
    rows: list[dict[str, Any]],
    output_path: str,
    fieldnames: list[str]
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_mir_fieldnames() -> list[str]:
    base_fields = [
        "strategy",
        "model",
        "line_number",
        "join_key",
        "audio_filename",
        "audio_basename",
        "audio_stem",
        "audio_id",
        "prompt_id",
        "cluster_id",
        "run_id",
        "rmsstd",
        "longest_silence_ms",
        "nonzero_silence"
    ]
    chroma_fields = ["chromasum_%d" % index for index in range(12)]
    return base_fields + chroma_fields


def get_gpu_fieldnames() -> list[str]:
    return [
        "strategy",
        "model",
        "line_number",
        "join_key",
        "audio_filename",
        "audio_basename",
        "audio_stem",
        "audio_id",
        "prompt_id",
        "cluster_id",
        "run_id",
        "tempo",
        "time_signature",
        "instruments",
        "n_instruments",
        "raw_keys",
        "key_positions",
        "canonical_song_key",
        "similarity_scores",
        "mean_similarity"
    ]


def get_coverage_fieldnames() -> list[str]:
    return ["strategy", "model", "mir_rows", "gpu_rows", "gpu_coverage"]


if __name__ == "__main__":
    main()
