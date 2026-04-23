#!/bin/bash

# Name of the job
#SBATCH --job-name=top500_mir
#SBATCH --output=logs/top500_mir_%j.out
#SBATCH --error=logs/top500_mir_%j.err

# Account info
#SBATCH --account=sahaslab-gpu

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition=standard

# Walltime (job duration)
#SBATCH --time=6:00:00

# Request CPU memory
#SBATCH --mem-per-cpu 300GB

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs

conda activate madlibs
python run_mir.py analysis_type=mir generator=real prompt_strategy=top500 run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared