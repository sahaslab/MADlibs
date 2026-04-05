#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/acestep_instruments.png --title "ACE-Step Keyword Mix Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/acestep_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/diffrhythm_instruments.png --title "DiffRhythm Keyword Mix Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/diffrhythm_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/keyword_mix/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/musicgen_instruments.png --title "MusicGen Keyword Mix Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/keyword_mix/instruments/musicgen_outliers.jsonl