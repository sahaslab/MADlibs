#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/keys/acestep_keys.png --title "ACE-Step Dependency Graph Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/keys/diffrhythm_keys.png --title "DiffRhythm Dependency Graph Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/keys/musicgen_keys.png --title "MusicGen Dependency Graph Keys"
