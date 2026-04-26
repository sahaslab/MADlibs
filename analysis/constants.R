library(tidyverse)

main_story_prompt_method <- "All Prompts"

gen_synthetic_strategies <- c(
  "blank",
  "dependency_graph",
  "keyword_mix",
  "madlibs_pcfg",
  "madlibs_templates"
)
gen_reference_strategies <- c("jamendo")
sdd_strategy <- "sdd"

model_labels <- c(
  "acestep" = "ACE-Step 1.5",
  "diffrhythm" = "DiffRhythm 1.2",
  "musicgen" = "MusicGen Large",
  "real" = "Real (SDD)"
)
reference_model_key <- "real"
acestep_source <- unname(model_labels["acestep"])
diffrhythm_source <- unname(model_labels["diffrhythm"])
musicgen_source <- unname(model_labels["musicgen"])
reference_source <- unname(model_labels[reference_model_key])

real_source_labels <- c(
  "sdd" = reference_source,
  "jamendo" = "Jamendo",
  "top500" = "Top 500",
  "audioset" = "AudioSet"
)
real_source_levels <- unname(real_source_labels)

generator_levels <- c(
  acestep_source,
  diffrhythm_source,
  musicgen_source
)

model_levels <- c(reference_source, generator_levels)
source_levels <- c(real_source_levels, generator_levels)
progression_source_levels <- c(
  musicgen_source,
  diffrhythm_source,
  acestep_source
)

model_year_labels <- setNames(
  c(
    "MusicGen (2023)",
    "DiffRhythm (2025)",
    "ACE-Step 1.5 (2026)"
  ),
  progression_source_levels
)

prompt_method_labels <- c(
  "blank" = "Blank",
  "dependency_graph" = "Dependency Graph",
  "keyword_mix" = "Keyword Mix",
  "madlibs_templates" = "Templates",
  "madlibs_pcfg" = "PCFG",
  "sdd" = "SDD",
  "jamendo" = "Reference"
)
reference_prompt_method <- unname(prompt_method_labels[gen_reference_strategies[1]])
sdd_prompt_method <- unname(prompt_method_labels[sdd_strategy])
synthetic_prompt_method_order <- c(
  "blank",
  "keyword_mix",
  "madlibs_templates",
  "dependency_graph",
  "madlibs_pcfg"
)

synthetic_prompt_method_levels <- unname(prompt_method_labels[synthetic_prompt_method_order])

strategy_labels <- prompt_method_labels
strategy_levels <- c(synthetic_prompt_method_levels, sdd_prompt_method)

prompt_method_palette <- c(
  "#4C566A",
  "#1F78B4",
  "#7A8DA1",
  "#D95F02",
  "#E7298A",
  "#1B9E77",
  "#7570B3",
  "#1F78B4"
)
names(prompt_method_palette) <- c(
  reference_prompt_method,
  main_story_prompt_method,
  synthetic_prompt_method_levels,
  sdd_prompt_method
)

source_palette <- c(
  "#4C566A",
  "#7570B3",
  "#CC79A7",
  "#7A8DA1",
  "#009E73",
  "#0072B2",
  "#D55E00"
)
names(source_palette) <- c(real_source_levels, generator_levels)

audio_embedding_source_levels <- c(generator_levels, reference_source)
audio_embedding_reference_label <- reference_source

audio_embedding_diversity_metric_labels <- c(
  "within_mean_cosine_distance" = "Mean cosine distance",
  "effective_rank_centered" = "Embedding effective rank"
)
audio_embedding_diversity_metric_levels <- unname(audio_embedding_diversity_metric_labels)

chroma_labels <- c(
  "C", "C#/Db", "D", "D#/Eb", "E", "F",
  "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
)

outcome_specs <- tibble::tribble(
  ~name,                  ~family,      ~transform, ~label,                  ~short,           ~source, ~plot_kind,   ~mega_include,
  "rmsstd",               "continuous", "identity", "RMS Std.",              "RMS",            "mir",   "density",    TRUE,
  "nonzero_silence",      "binary",     "identity", "Any Silence",           "Any Sil.",       "mir",   "bar",        FALSE,
  "chroma_concentration", "continuous", "log10",    "Chroma Concentration",  "Chroma Conc.",   "mir",   "density",    TRUE,
  "n_instruments",        "continuous", "identity", "Instrument Count",      "# Inst.",        "gpu",   "histogram",  TRUE,
  "tempo",                "continuous", "log10",    "Tempo",                 "Tempo",          "gpu",   "density",    TRUE,
  "tempo_fold_60",        "continuous", "identity", "Folded Tempo (60-BPM)", "Tempo60",        "gpu",   "density",    TRUE,
  "mean_similarity",      "continuous", "identity", "Mean CLAP Similarity",  "CLAP Sim.",      "gpu",   "density",    FALSE
)

outcome_levels <- outcome_specs$label

tail_probabilities <- c(0.05)
neighbor_k <- 10
distance_row_cap <- 2500

# Toggle for continuous outcome distributions. "ridgeline" uses
# ggridges::geom_ridgeline; "density" uses overlaid geom_density.
outcome_distribution_density_geom <- "ridgeline"

# CLAP similarity is GPU-only for these data, so reference-relative reach and
# support plots exclude it rather than plotting incomplete real comparisons.
support_plot_outcome_levels <- c(
  "RMS Std.",
  "Chroma Concentration",
  "Instrument Count",
  "Tempo",
  "Folded Tempo (60-BPM)"
)

feature_similarity_outcome_levels <- c(
  "RMS Std.",
  "Any Silence",
  "Chroma Concentration",
  "Instrument Count",
  "Tempo",
  "Folded Tempo (60-BPM)",
  "Mean CLAP Similarity"
)

# Support-interval plots use folded tempo only; raw tempo is redundant there.
reference_interval_outcome_levels <- c(
  "Folded Tempo (60-BPM)",
  "RMS Std.",
  "Chroma Concentration"
)
reference_interval_outcome_names <- c(
  "tempo_fold_60",
  "rmsstd",
  "chroma_concentration"
)

feature_space_feature_names <- c(
  "rmsstd",
  "nonzero_silence",
  "chroma_concentration",
  "tempo",
  "tempo_fold_60",
  "n_instruments",
  "mean_similarity"
)

distance_feature_names <- c(
  "rmsstd",
  "nonzero_silence",
  "chroma_concentration",
  "tempo",
  "tempo_fold_60",
  "n_instruments"
)

feature_similarity_feature_names <- feature_space_feature_names

model_pair_specs <- tibble(
  source_1 = c(
    musicgen_source,
    diffrhythm_source,
    musicgen_source
  ),
  source_2 = c(
    diffrhythm_source,
    acestep_source,
    acestep_source
  ),
  pair_rank = c(1, 2, 3),
  later_year = c(2025, 2026, 2026)
) %>%
  mutate(pair_label = sprintf("%s vs %s", model_year_labels[source_1], model_year_labels[source_2])) %>%
  select(source_1, source_2, pair_label, pair_rank, later_year)

human_region_specs <- tibble::tribble(
  ~region_name, ~lower_prob, ~upper_prob,
  "q05_q95",    0.05,        0.95,
  "q10_q90",    0.10,        0.90,
  "q25_q75",    0.25,        0.75
)
