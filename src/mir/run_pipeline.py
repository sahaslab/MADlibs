import argparse
import os
import json

from src.mir.Pipeline import process

def run_pipeline(input_dir: str, n: int, p: float):
    total = []

    for i in range(n):
        prompt_dir = os.path.join(input_dir, f"prompt_{i}")

        if not os.path.isdir(prompt_dir):
            continue

        files = os.listdir(prompt_dir)

        wav_files = [f for f in files if f.lower().endswith(".wav")]

        if len(wav_files) != 1:
            print(f"Unexpected contents in {prompt_dir}: {files}")
            continue

        wav_path = os.path.join(prompt_dir, wav_files[0])
        out = process(audio_filename=wav_path)

        total.append(out)

    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to write outputs",
        default="processing_outputs",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Directory to find audio inputs",
        default="outputs/1",
    )
    parser.add_argument("-n", "--n_prompts", help="Number of Prompts", default=100)
    parser.add_argument("-p", "--percentage", help="percentage of outputs to process", default=1.0)
    args = parser.parse_args()
    output_dir = args.output_dir
    input_dir = args.input_dir
    n = args.n_prompts
    p = args.percentage
    processed = run_pipeline(input_dir, n, p)
    # write processed to file
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "results.json")

    with open(output_path, "w") as f:
        json.dump(processed, f)

if __name__ == "__main__":
    main()
