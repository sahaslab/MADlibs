import argparse
import json
from typing import Any, Dict, Iterable, Tuple


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e


def get_audio_filename(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Record is missing a valid 'metadata' object")

    audio_filename = metadata.get("audio_filename")
    if not isinstance(audio_filename, str):
        raise ValueError("Record is missing 'metadata.audio_filename'")

    return audio_filename


def split_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata = record.get("metadata", {})
    other_fields = {k: v for k, v in record.items() if k != "metadata"}
    return metadata, other_fields


def merge_records(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefer: str = "left",
) -> Dict[str, Any]:
    left_meta, left_fields = split_record(left)
    right_meta, right_fields = split_record(right)

    # Merge metadata, preferring the chosen side on overlapping metadata keys.
    merged_meta = dict(right_meta if prefer == "right" else left_meta)
    secondary_meta = left_meta if prefer == "right" else right_meta
    for k, v in secondary_meta.items():
        if k not in merged_meta:
            merged_meta[k] = v

    # Merge top-level non-metadata fields.
    # Overlapping field names are pruned from the secondary record.
    if prefer == "left":
        merged_fields = dict(left_fields)
        for k, v in right_fields.items():
            if k not in merged_fields:
                merged_fields[k] = v
    else:
        merged_fields = dict(right_fields)
        for k, v in left_fields.items():
            if k not in merged_fields:
                merged_fields[k] = v

    return {"metadata": merged_meta, **merged_fields}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two JSONL files by metadata.audio_filename."
    )
    parser.add_argument("--file1", help="First input JSONL file")
    parser.add_argument("--file2", help="Second input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")
    parser.add_argument(
        "--prefer",
        choices=["left", "right"],
        default="left",
        help="Which file wins when overlapping fields exist (default: left)",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Also include records that appear in only one file",
    )
    args = parser.parse_args()

    file1_records: Dict[str, Dict[str, Any]] = {}
    file2_records: Dict[str, Dict[str, Any]] = {}

    for record in load_jsonl(args.file1):
        key = get_audio_filename(record)
        if key in file1_records:
            raise ValueError(f"Duplicate audio_filename in {args.file1}: {key}")
        file1_records[key] = record

    for record in load_jsonl(args.file2):
        key = get_audio_filename(record)
        if key in file2_records:
            raise ValueError(f"Duplicate audio_filename in {args.file2}: {key}")
        file2_records[key] = record

    keys1 = set(file1_records)
    keys2 = set(file2_records)
    shared_keys = keys1 & keys2

    output_records = []

    for key in sorted(shared_keys):
        merged = merge_records(
            file1_records[key], file2_records[key], prefer=args.prefer
        )
        output_records.append(merged)

    if args.keep_unmatched:
        only1 = keys1 - keys2
        only2 = keys2 - keys1

        for key in sorted(only1):
            output_records.append(file1_records[key])
        for key in sorted(only2):
            output_records.append(file2_records[key])

    with open(args.output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(output_records)} records to {args.output} "
        f"({len(shared_keys)} merged matches)"
    )


if __name__ == "__main__":
    main()
