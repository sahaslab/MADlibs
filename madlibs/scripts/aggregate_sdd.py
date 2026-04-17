import csv
import shutil
import argparse
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

def aggregate_audio_data(csv_path, synthetic_dir, real_dir, write_dir, caption_col='caption', audio_col='path'):
    """
    Aggregates synthetic and real audio data based on a CSV mapping.
    """
    # 1. Setup paths
    synth_path = Path(synthetic_dir)
    real_path = Path(real_dir)
    out_path = Path(write_dir)
    
    # Create the base output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Parse the CSV to create a mapping of Caption -> Real MP3 Filename
    caption_to_real_file = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle potential missing/empty fields
            if caption_col not in row or audio_col not in row:
                continue
                
            caption = row[caption_col].strip()
            orig_audio_path = row[audio_col].strip()
            
            # Transform "34/1004034.mp3" into "1004034.2min.mp3"
            base_filename = Path(orig_audio_path).stem 
            target_real_filename = f"{base_filename}.2min.mp3"
            
            caption_to_real_file[caption] = target_real_filename

    # 3. Iterate through the synthetic directory
    for prompt_dir in tqdm(synth_path.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue

        prompt_txt_path = prompt_dir / "prompt.txt"
        if not prompt_txt_path.exists():
            continue

        # Extract caption from prompt.txt
        with open(prompt_txt_path, mode='r', encoding='utf-8') as f:
            # Replaces quotes if they were accidentally double-escaped, and strips whitespace
            caption = f.read().strip().strip('"')

        # 4. Check if the caption exists in our CSV mapping
        if caption not in caption_to_real_file:
            print(f"Notice: Caption for {prompt_dir.name} not found in CSV. Skipping.")
            continue  # Skip if there's no match

        real_filename = caption_to_real_file[caption]
        real_file_path = real_path / real_filename

        # Verify the real audio file actually exists
        if not real_file_path.exists():
            print(f"Warning: Real file {real_filename} not found in {real_dir}. Skipping {prompt_dir.name}.")
            continue

        # Locate the synthetic .wav file
        synth_wavs = list(prompt_dir.glob("*.wav"))
        if not synth_wavs:
            print(f"Warning: No synthetic .wav file found in {prompt_dir}. Skipping.")
            continue
        synth_wav_path = synth_wavs[0]

        # 5. Write to the new directory structure
        out_prompt_dir = out_path / prompt_dir.name
        out_prompt_dir.mkdir(exist_ok=True)

        # Copy prompt.txt
        shutil.copy2(prompt_txt_path, out_prompt_dir / "prompt.txt")

        # Copy synthetic.wav
        shutil.copy2(synth_wav_path, out_prompt_dir / "synthetic.wav")

        # Convert and save real.wav
        out_real_wav_path = out_prompt_dir / "real.wav"
        print(f"Converting and writing data for {prompt_dir.name}...")
        
        # Load the mp3 and export it as a wav file
        audio = AudioSegment.from_mp3(real_file_path)
        audio.export(out_real_wav_path, format="wav")

    print("Aggregation complete.")

# ==========================================
# Command Line Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and match synthetic/real audio data based on a CSV mapping.")
    
    # Required Arguments
    parser.add_argument("--csv", required=True, help="Path to the metadata CSV file")
    parser.add_argument("--synth_dir", required=True, help="Path to the synthetic directory")
    parser.add_argument("--real_dir", required=True, help="Path to the real directory containing .mp3s")
    parser.add_argument("--write_dir", required=True, help="Path to the output write directory")
    
    # Optional Arguments (with defaults matching your CSV)
    parser.add_argument("--caption_col", default="caption", help="Name of the caption column in the CSV (default: 'caption')")
    parser.add_argument("--audio_col", default="path", help="Name of the audio path column in the CSV (default: 'path')")
    
    args = parser.parse_args()

    aggregate_audio_data(
        csv_path=args.csv,
        synthetic_dir=args.synth_dir,
        real_dir=args.real_dir,
        write_dir=args.write_dir,
        caption_col=args.caption_col,
        audio_col=args.audio_col
    )