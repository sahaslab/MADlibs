#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/acestep_instruments.png --title "ACE-Step Madlibs Templates Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/acestep_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/diffrhythm_instruments.png --title "DiffRhythm Madlibs Templates Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/diffrhythm_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/madlibs_templates/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/musicgen_instruments.png --title "MusicGen Madlibs Templates Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/madlibs_templates/instruments/musicgen_outliers.jsonl
