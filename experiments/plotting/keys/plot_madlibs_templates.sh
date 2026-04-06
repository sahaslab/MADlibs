#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/keys/acestep_keys.png --title "ACE-Step Madlibs Templates Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/keys/diffrhythm_keys.png --title "DiffRhythm Madlibs Templates Keys"
python madlibs/scripts/plot_song_keys.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/keys/musicgen_keys.png --title "MusicGen Madlibs Templates Keys"
