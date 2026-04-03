#!/bin/bash

# Name of the job
#SBATCH --job-name=generate_madlibs_templates
#SBATCH --output=logs/generate_madlibs_templates_%j.out
#SBATCH --error=logs/generate_madlibs_templates_%j.err

# Account info
#SBATCH --account=sahaslab-gpu

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition=l40s_nova

# Request the GPU partition
#SBATCH --gpus=1

# Walltime (job duration)
#SBATCH --time=1:15:00

# Request CPU memory
#SBATCH --mem-per-cpu 300GB

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

conda activate madlibs

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs

python3 run_prompts.py prompt_strategy=madlibs prompt_strategy.induction_method=templates run.prompt_strategy_name=madlibs_templates run.n_prompts=10000 run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared
