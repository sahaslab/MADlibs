#!/bin/bash

# Name of the job
#SBATCH --job-name=generate_blank_diffrhythm
#SBATCH --output=logs/generate_blank_diffrhythm_%j.out
#SBATCH --error=logs/generate_blank_diffrhythm_%j.err

# Account info
#SBATCH --account=sahaslab-gpu

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition=h200

# Request the GPU partition
#SBATCH --gpus=1

# Walltime (job duration)
#SBATCH --time=7:00:00

# Request CPU memory
#SBATCH --mem-per-cpu 300GB

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs

conda activate environment_diffrhythm
python run_generation.py generator=diffrhythm prompt_strategy=blank run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared

conda deactivate
conda activate madlibs
python run_mir.py analysis_type=mir generator=diffrhythm prompt_strategy=blank run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared
