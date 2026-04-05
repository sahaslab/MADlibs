#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/keys/acestep_keys.png --title "ACE-Step Keyword Mix Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/keys/diffrhythm_keys.png --title "DiffRhythm Keyword Mix Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/keys/musicgen_keys.png --title "MusicGen Keyword Mix Keys"
