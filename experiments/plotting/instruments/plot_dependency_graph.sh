#!/bin/bash

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

cd /dartfs/rc/lab/S/SinghN/noah/imPROMPTu/madlibs


conda activate madlibs
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/acestep/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/acestep_instruments.png --title "ACE-Step Dependency Graph Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/acestep_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/diffrhythm/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/diffrhythm_instruments.png --title "DiffRhythm Dependency Graph Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/diffrhythm_outliers.jsonl
python madlibs/scripts/plot_gpu_instrument_histograms.py /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/dependency_graph/musicgen/mir/gpu_results.jsonl --output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/musicgen_instruments.png --title "MusicGen Dependency Graph Instruments" --report-outliers-above 10 --outliers-output /dartfs-hpc/rc/home/t/f00814t/lab/shared/MADLibs_output/plots/dependency_graph/instruments/musicgen_outliers.jsonl