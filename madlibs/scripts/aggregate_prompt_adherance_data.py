import json
import random
import shutil
import argparse
from pathlib import Path

def sample_dataset(input_base_dir: str, output_dir: str, samples_per_model: int = 8, target_strategies: list[str] = None):
    """
    Iterates through prompt strategies and models, randomly samples a set number 
    of audio/text pairs, copies/renames the audio, and creates a mapping JSONL.
    """
    base_path = Path(input_base_dir)
    out_path = Path(output_dir)
    
    # Valid audio extensions
    valid_audio_extensions = {'.wav', '.mp3'}
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The input directory '{input_base_dir}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_output_path = out_path / "sampled_dataset.jsonl"
    
    total_sampled = 0

    with open(jsonl_output_path, 'w', encoding='utf-8') as jsonl_out:
        
        # Iterate over prompt strategy directories
        for strategy_dir in base_path.iterdir():
            if not strategy_dir.is_dir():
                continue
                
            strategy_name = strategy_dir.name
            
            # If a list of target strategies was provided, skip any that aren't in the list
            if target_strategies and strategy_name not in target_strategies:
                continue
                
            # Iterate over model directories within the strategy
            for model_dir in strategy_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                
                # Collect all available valid samples for this model
                available_samples = []
                
                for prompt_path in model_dir.rglob("prompt.txt"):
                    subdir = prompt_path.parent
                    
                    # Look for the audio file
                    audio_path = None
                    for file_path in subdir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in valid_audio_extensions:
                            audio_path = file_path
                            break
                    
                    if audio_path:
                        # Extract the prompt number (e.g., 'prompt_0' -> '0')
                        folder_parts = subdir.name.split('_')
                        prompt_number = folder_parts[-1] if len(folder_parts) > 1 else subdir.name
                        
                        # Read the text
                        with open(prompt_path, 'r', encoding='utf-8') as text_file:
                            text_content = text_file.read().strip()
                            
                        available_samples.append({
                            "original_audio": audio_path,
                            "text": text_content,
                            "prompt_number": prompt_number
                        })
                
                # Check if we have enough samples
                if not available_samples:
                    print(f"Skipping {strategy_name}/{model_name}: No valid data found.")
                    continue
                
                if len(available_samples) < samples_per_model:
                    print(f"Warning: {strategy_name}/{model_name} only has {len(available_samples)} samples. Taking all of them.")
                    sampled_data = available_samples
                else:
                    # Randomly sample the desired amount
                    sampled_data = random.sample(available_samples, samples_per_model)
                
                # Copy files, rename them, and write to JSONL
                for item in sampled_data:
                    orig_audio = item["original_audio"]
                    ext = orig_audio.suffix
                    
                    # Construct new filename: <model_name>_<prompting_strategy>_<prompt_number>.wav
                    new_filename = f"{model_name}_{strategy_name}_{item['prompt_number']}{ext}"
                    new_filepath = out_path / new_filename
                    
                    # Copy the file to the new directory
                    shutil.copy2(orig_audio, new_filepath)
                    
                    # Write to JSONL (saving the path relative to the output directory)
                    data_entry = {
                        "audio_path": new_filename,
                        "text": item["text"]
                    }
                    jsonl_out.write(json.dumps(data_entry) + '\n')
                    total_sampled += 1
                    
                print(f"[{strategy_name} | {model_name}] Sampled {len(sampled_data)} files.")

    print(f"\nSuccess! Copied {total_sampled} total audio files to '{output_dir}'.")
    print(f"JSONL mapping written to '{jsonl_output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample audio generations across specific prompt strategies and models."
    )
    
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Path to the top-level directory containing prompt strategies."
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Path to the new directory where sampled audio and the JSONL will be saved."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=8,
        help="Number of samples to draw per model per strategy (default: 8)."
    )
    parser.add_argument(
        "--strategies", 
        nargs="+", 
        type=str,
        help="A space-separated list of prompt strategies to sample from. If omitted, all are sampled."
    )
    
    args = parser.parse_args()
    sample_dataset(args.input_dir, args.output_dir, args.samples, args.strategies)