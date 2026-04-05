import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt


NOTE_TO_PITCH_CLASS = {
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
    "CB": 11,
}

RELATIVE_KEY_LABELS = {
    0: "C Major / A minor",
    1: "Db Major / Bb minor",
    2: "D Major / B minor",
    3: "Eb Major / C minor",
    4: "E Major / C# minor",
    5: "F Major / D minor",
    6: "Gb Major / Eb minor",
    7: "G Major / E minor",
    8: "Ab Major / F minor",
    9: "A Major / F# minor",
    10: "Bb Major / G minor",
    11: "B Major / G# minor",
}


def normalize_note_name(note: str) -> str:
    return (
        note.strip()
        .upper()
        .replace("♯", "#")
        .replace("♭", "B")
    )


def canonicalize_key_name(key_name: str) -> int | None:
    cleaned = key_name.strip()
    if not cleaned:
        return None

    parts = cleaned.split()
    if len(parts) < 2:
        return None

    tonic = normalize_note_name(parts[0])
    mode = parts[-1].lower()
    pitch_class = NOTE_TO_PITCH_CLASS.get(tonic)
    if pitch_class is None:
        return None

    if mode == "major":
        return pitch_class
    if mode == "minor":
        return (pitch_class + 3) % 12
    return None


def select_song_key(keys: list[str]) -> int | None:
    canonical_keys = [
        canonical_key
        for canonical_key in (canonicalize_key_name(key_name) for key_name in keys)
        if canonical_key is not None
    ]
    if not canonical_keys:
        return None

    counts = collections.Counter(canonical_keys)
    max_count = max(counts.values())
    for canonical_key in canonical_keys:
        if counts[canonical_key] == max_count:
            return canonical_key
    return None


def load_song_key_counts(jsonl_path: Path) -> collections.Counter:
    song_key_counts = collections.Counter()

    with jsonl_path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {jsonl_path}: {error}"
                ) from error

            key_info = record.get("key_info", {})
            if not isinstance(key_info, dict):
                continue

            keys = key_info.get("keys", [])
            if not isinstance(keys, list):
                continue

            song_key = select_song_key([key for key in keys if isinstance(key, str)])
            if song_key is None:
                continue

            song_key_counts[RELATIVE_KEY_LABELS[song_key]] += 1

    return song_key_counts


def plot_song_key_counts(
    song_key_counts: collections.Counter,
    output_path: Path,
    figure_title: str | None,
) -> None:
    sorted_keys = sorted(song_key_counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in sorted_keys]
    values = [value for _, value in sorted_keys]

    figure_height = max(7, 0.6 * len(labels))
    figure, axis = plt.subplots(1, 1, figsize=(14, figure_height), constrained_layout=True)
    if figure_title:
        figure.suptitle(figure_title, fontsize=16)

    axis.barh(labels, values)
    axis.set_title("Most Common Key Per Song")
    axis.set_xlabel("Number of Songs")
    axis.set_ylabel("Canonical Key Group")
    axis.set_xscale("log")
    axis.invert_yaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the most common song key from a GPU MIR results JSONL file, "
            "collapsing enharmonic equivalents and relative major/minor pairs."
        )
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("song_keys.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional overall title for the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    song_key_counts = load_song_key_counts(args.jsonl_path)
    plot_song_key_counts(
        song_key_counts=song_key_counts,
        output_path=args.output,
        figure_title=args.title,
    )


if __name__ == "__main__":
    main()
