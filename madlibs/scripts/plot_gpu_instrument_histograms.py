import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_instrument_data(
    jsonl_path: Path,
) -> tuple[collections.Counter, collections.Counter, list[dict[str, object]]]:
    instrument_song_counts = collections.Counter()
    instruments_per_song_counts = collections.Counter()
    outlier_rows: list[dict[str, object]] = []

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
            num_instruments = len(unique_instruments)
            instruments_per_song_counts[num_instruments] += 1

            metadata = record.get("metadata", {})
            audio_filename = None
            if isinstance(metadata, dict):
                candidate = metadata.get("audio_filename")
                if isinstance(candidate, str):
                    audio_filename = candidate

            outlier_rows.append(
                {
                    "line_number": line_number,
                    "audio_filename": audio_filename,
                    "num_instruments_identified": num_instruments,
                }
            )

    return instrument_song_counts, instruments_per_song_counts, outlier_rows


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

    instrument_plot_height = max(8, 0.32 * len(instrument_labels))
    figure_height = instrument_plot_height + 6
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(18, figure_height),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [max(2, len(instrument_labels) / 12.0), 1.5]},
    )
    if figure_title:
        figure.suptitle(figure_title, fontsize=16)

    axes[0].barh(instrument_labels, instrument_values)
    axes[0].set_title("Songs Containing Each Instrument")
    axes[0].set_xlabel("Number of Songs")
    axes[0].set_ylabel("Instrument")
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=9)

    axes[1].bar(num_instruments_labels, num_instruments_values, width=0.8)
    axes[1].set_title("Number of Instruments Per Song")
    axes[1].set_xlabel("Number of Instruments")
    axes[1].set_ylabel("Number of Songs")
    axes[1].set_yscale("log")
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
    parser.add_argument(
        "--report-outliers-above",
        type=int,
        default=None,
        help="If set, write songs with more than this many instruments to JSONL.",
    )
    parser.add_argument(
        "--outliers-output",
        type=Path,
        default=None,
        help="Path to the output JSONL file for reported outliers.",
    )
    return parser.parse_args()


def write_outliers(
    outlier_rows: list[dict[str, object]],
    threshold: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in outlier_rows:
            num_instruments = row["num_instruments_identified"]
            if not isinstance(num_instruments, int) or num_instruments <= threshold:
                continue
            audio_filename = row["audio_filename"]
            prompt = None
            if isinstance(audio_filename, str) and audio_filename:
                prompt_path = Path(audio_filename).parent / "prompt.txt"
                if prompt_path.exists():
                    prompt = prompt_path.read_text(encoding="utf-8").strip()
            file_obj.write(
                json.dumps(
                    {
                        "audio_filename": audio_filename,
                        "prompt": prompt,
                        "num_instruments_identified": num_instruments,
                    }
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    instrument_song_counts, instruments_per_song_counts, outlier_rows = (
        load_instrument_data(args.jsonl_path)
    )
    plot_histograms(
        instrument_song_counts=instrument_song_counts,
        instruments_per_song_counts=instruments_per_song_counts,
        output_path=args.output,
        figure_title=args.title,
    )
    if args.report_outliers_above is not None:
        if args.outliers_output is None:
            raise ValueError(
                "--outliers-output must be provided when --report-outliers-above is set."
            )
        write_outliers(
            outlier_rows=outlier_rows,
            threshold=args.report_outliers_above,
            output_path=args.outliers_output,
        )


if __name__ == "__main__":
    main()
