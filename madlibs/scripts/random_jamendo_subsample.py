import random
import argparse 
from pathlib import Path
import shutil 
import tqdm
random.seed(33)

from concurrent.futures import ThreadPoolExecutor

def copy_file(file_info):
    """Helper function to copy a single file to the destination."""
    src_path, dst_dir = file_info
    # .name gets just the filename, ignoring the subfolders
    dst_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(description="Process Jamendo data and save results.")

    # Input path argument
    parser.add_argument(
        '--jamendo_path', 
        type=str, 
        required=True, 
        help='The file path to the Jamendo dataset'
    )

    # Output path argument
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./output', 
        help='The directory where results will be saved (default: ./output)'
    )

        # Output path argument
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=10000,
        help='Number of random samples to take from jamendo'
    )

    args = parser.parse_args()

    # Initialize the path object
    root_dir = Path(args.jamendo_path)



    # Get all .wav files in subdirectories
    # The '**/*.wav' pattern means "search all subfolders for files ending in .mp3"
    wav_files = list(root_dir.glob('**/*.mp3'))

    random_subset = random.sample(wav_files, args.num_samples)

    output_base = Path(args.output_path)

    tasks = [(f, output_base) for f in random_subset]

    with ThreadPoolExecutor(max_workers=8) as executor:
            # Use tqdm to track progress
        list(tqdm.tqdm(executor.map(copy_file, tasks), total=len(tasks)))
    
    print(f'{args.num_samples} samples written to {args.output_path}')

if __name__ == '__main__':
    main()