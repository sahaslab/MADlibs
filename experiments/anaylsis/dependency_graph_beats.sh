#!/bin/bash

# Name of the job
#SBATCH --job-name=beats_dependency_graph
#SBATCH --output=logs/beats_dependency_graph_%j.out
#SBATCH --error=logs/beats_dependency_graph_%j.err

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
#SBATCH --time=6:00:00

# Request CPU memory
#SBATCH --mem-per-cpu 300GB


source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python run_mir.py analysis_type=beats generator=acestep prompt_strategy=dependency_graph run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared
python run_mir.py analysis_type=beats generator=musicgen prompt_strategy=dependency_graph run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared
python run_mir.py analysis_type=beats generator=diffrhythm prompt_strategy=dependency_graph run.output_dir=MADLibs_output hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared