#!/bin/bash

# Name of the job
#SBATCH --job-name=analysis_keyword_mix_musicgen
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
#SBATCH --time=5:00:00

# Request CPU memory 
#SBATCH --mem-per-cpu 300GB

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs

conda activate madlibs
python run_mir.py analysis_type=gpu generator=musicgen prompt_strategy=keyword_mix run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared
