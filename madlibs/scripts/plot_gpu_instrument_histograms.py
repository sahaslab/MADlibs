import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_instrument_data(jsonl_path: Path) -> tuple[collections.Counter, collections.Counter]:
    instrument_song_counts: collections.Counter[str] = collections.Counter()
    instruments_per_song_counts: collections.Counter[int] = collections.Counter()

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

            instruments = record.get("instruments", [])
            if not isinstance(instruments, list):
                raise ValueError(
                    f"Expected 'instruments' to be a list on line {line_number}."
                )

            unique_instruments = []
            seen = set()
            for instrument in instruments:
                if not isinstance(instrument, str):
                    continue
                normalized = instrument.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                unique_instruments.append(normalized)

            instrument_song_counts.update(unique_instruments)
            instruments_per_song_counts[len(unique_instruments)] += 1

    return instrument_song_counts, instruments_per_song_counts


def plot_histograms(
    instrument_song_counts: collections.Counter,
    instruments_per_song_counts: collections.Counter,
    output_path: Path,
    figure_title: str | None,
) -> None:
    sorted_instruments = sorted(
        instrument_song_counts.items(), key=lambda item: (-item[1], item[0])
    )
    instrument_labels = [label for label, _ in sorted_instruments]
    instrument_values = [value for _, value in sorted_instruments]

    sorted_song_counts = sorted(instruments_per_song_counts.items())
    num_instruments_labels = [count for count, _ in sorted_song_counts]
    num_instruments_values = [value for _, value in sorted_song_counts]

    figure, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)
    if figure_title:
        figure.suptitle(figure_title, fontsize=16)

    axes[0].bar(instrument_labels, instrument_values)
    axes[0].set_title("Songs Containing Each Instrument")
    axes[0].set_xlabel("Instrument")
    axes[0].set_ylabel("Number of Songs")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(num_instruments_labels, num_instruments_values, width=0.8)
    axes[1].set_title("Number of Instruments Per Song")
    axes[1].set_xlabel("Number of Instruments")
    axes[1].set_ylabel("Number of Songs")
    axes[1].set_xticks(num_instruments_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot instrument-count histograms from a GPU MIR results JSONL file."
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
        default=Path("instrument_histograms.png"),
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
    instrument_song_counts, instruments_per_song_counts = load_instrument_data(
        args.jsonl_path
    )
    plot_histograms(
        instrument_song_counts=instrument_song_counts,
        instruments_per_song_counts=instruments_per_song_counts,
        output_path=args.output,
        figure_title=args.title,
    )


if __name__ == "__main__":
    main()
