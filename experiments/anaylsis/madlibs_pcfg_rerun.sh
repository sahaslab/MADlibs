#!/bin/bash

# Name of the job
#SBATCH --job-name=rerun_madlibs_pcfg
#SBATCH --output=logs/rerun_madlibs_pcfg_%j.out
#SBATCH --error=logs/rerun_madlibs_pcfg_%j.err

# Account info
#SBATCH --account=free

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition=gpu_preempt

#SBATCH --nodelist=amp04

# Request the GPU partition
#SBATCH --gpus=1

# Walltime (job duration)
#SBATCH --time=30:00

# Request CPU memory
#SBATCH --mem-per-cpu 300GB

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs

conda activate madlibs

python madlibs/scripts/rerun_gpu_mismatches.py \
  /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_pcfg/musicgen/mir \
  --file1 mir_results.jsonl \
  --file2 gpu_results.jsonl \
  -- \
  generator=musicgen \
  prompt_strategy=madlibs_pcfg \
  run.output_dir=MADLibs_output \
  hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared

  python madlibs/scripts/rerun_gpu_mismatches.py \
  /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_pcfg/diffrhythm/mir \
  --file1 mir_results.jsonl \
  --file2 gpu_results.jsonl \
  -- \
  generator=diffrhythm \
  prompt_strategy=madlibs_pcfg \
  run.output_dir=MADLibs_output \
  hydra.run.dir=/dartfs/rc/lab/S/SinghN/shared