source("constants.R")
source("io.R")
source("data_prep.R")
source("stats.R")
source("plots.R")
source("prompt_analysis.R")
source("mir_analysis.R")


# Setup ------------------------------------------------------------------

data_dir <- "analysis_data"
figures_dir <- "figures"
tables_dir <- "tables"
ensure_output_dir(figures_dir)
ensure_output_dir(tables_dir)

# Prompt diversity and support -------------------------------------------

prompt_diversity_summary <- read.csv(
  file.path(data_dir, "prompt_strategy_diversity_summary.csv"),
  stringsAsFactors = FALSE
)

if (!("within_mean_cosine_distance_lci" %in% names(prompt_diversity_summary))) {
  prompt_diversity_summary$within_mean_cosine_distance_lci <- NA_real_
}
if (!("within_mean_cosine_distance_uci" %in% names(prompt_diversity_summary))) {
  prompt_diversity_summary$within_mean_cosine_distance_uci <- NA_real_
}

prompt_diversity_summary <- prompt_diversity_summary %>%
  mutate(
    prompt_method = recode(strategy, !!!prompt_method_labels),
    is_real_prompt_reference = strategy == sdd_strategy
  ) %>%
  arrange(is_real_prompt_reference, effective_rank_centered, distinct_prompt_text_share, prompt_method) %>%
  mutate(prompt_method = factor(prompt_method, levels = unique(prompt_method))) %>%
  select(-is_real_prompt_reference)
prompt_method_levels <- levels(prompt_diversity_summary$prompt_method)

prompt_support_rank_plot <- build_prompt_support_rank_plot(prompt_diversity_summary)
save_plot_pdf(
  prompt_support_rank_plot,
  file.path(figures_dir, "prompts_prompt_support_rank.pdf"),
  8.4,
  2.4
)
save_table_outputs(
  prompt_diversity_summary,
  file.path(tables_dir, "prompts_prompt_diversity_summary")
)

# Prompt strategy similarity ---------------------------------------------

prompt_strategy_similarity_summary <- read.csv(
  file.path(data_dir, "prompt_strategy_pairwise_similarity.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prompt_method_1 = recode(strategy_1, !!!prompt_method_labels),
    prompt_method_2 = recode(strategy_2, !!!prompt_method_labels),
    prompt_method_1 = factor(prompt_method_1, levels = prompt_method_levels),
    prompt_method_2 = factor(prompt_method_2, levels = prompt_method_levels)
  ) %>%
  arrange(prompt_method_1, prompt_method_2)

prompt_strategy_similarity_plot <- build_prompt_strategy_similarity_plot(
  prompt_diversity_summary,
  prompt_strategy_similarity_summary
)

save_plot_pdf(
  prompt_strategy_similarity_plot,
  file.path(figures_dir, "prompts_prompt_strategy_similarity.pdf"),
  4,
  2.8
)
save_table_outputs(
  prompt_strategy_similarity_summary,
  file.path(tables_dir, "prompts_prompt_strategy_similarity_summary")
)

# Prompt embedding neighborhoods -----------------------------------------

prompt_neighbor_mixing_summary <- read.csv(
  file.path(data_dir, "prompt_neighbor_mixing_summary.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prompt_method = recode(strategy, !!!prompt_method_labels),
    neighbor_prompt_method = recode(neighbor_strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels),
    neighbor_prompt_method = factor(neighbor_prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(prompt_method, neighbor_prompt_method)

prompt_embedding_neighbor_summary <- read.csv(
  file.path(data_dir, "prompt_embedding_neighbor_summary.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(prompt_method)

prompt_embedding_cluster_summary <- read.csv(
  file.path(data_dir, "prompt_embedding_cluster_summary_median.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(prompt_method)

prompt_neighbor_mixing_plot <- build_prompt_neighbor_mixing_plot(prompt_neighbor_mixing_summary)

save_plot_pdf(
  prompt_neighbor_mixing_plot,
  file.path(figures_dir, "prompts_prompt_neighbor_mixing.pdf"),
  4,
  3.2
)
save_table_outputs(
  prompt_neighbor_mixing_summary,
  file.path(tables_dir, "prompts_prompt_neighbor_mixing_summary")
)
save_table_outputs(
  prompt_embedding_neighbor_summary,
  file.path(tables_dir, "prompts_prompt_embedding_neighbor_summary")
)
save_table_outputs(
  prompt_embedding_cluster_summary,
  file.path(tables_dir, "prompts_prompt_embedding_cluster_summary")
)

# Prompt panel assembly --------------------------------------------------

prompt_panel <- prompt_support_rank_plot / (prompt_strategy_similarity_plot | prompt_neighbor_mixing_plot)
save_plot_pdf(
  prompt_panel,
  file.path(figures_dir, "panel_prompts.pdf"),
  10,
  6
)
save_table_outputs(
  build_prompt_aggregate_table(
    prompt_diversity_summary,
    prompt_strategy_similarity_summary,
    prompt_neighbor_mixing_summary,
    prompt_embedding_cluster_summary
  ),
  file.path(tables_dir, "panel_prompts")
)

# Audio embedding diversity and rank -------------------------------------

audio_embedding_diversity_summary <- read.csv(
  file.path(data_dir, "audio_embedding_diversity_summary.csv"),
  stringsAsFactors = FALSE
)

if (!("within_mean_cosine_distance_lci" %in% names(audio_embedding_diversity_summary))) {
  audio_embedding_diversity_summary$within_mean_cosine_distance_lci <- NA_real_
}
if (!("within_mean_cosine_distance_uci" %in% names(audio_embedding_diversity_summary))) {
  audio_embedding_diversity_summary$within_mean_cosine_distance_uci <- NA_real_
}

audio_embedding_diversity_summary <- audio_embedding_diversity_summary %>%
  mutate(
    source = recode(model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(source, prompt_method)

audio_embedding_diversity_plot <- build_audio_embedding_diversity_plot(audio_embedding_diversity_summary)
save_plot_pdf(
  audio_embedding_diversity_plot,
  file.path(figures_dir, "prompts_audio_embedding_diversity.pdf"),
  8,
  4
)
save_table_outputs(
  audio_embedding_diversity_summary,
  file.path(tables_dir, "prompts_audio_embedding_diversity_summary")
)

# Audio embedding similarity ---------------------------------------------

audio_embedding_strategy_pairwise_similarity <- read.csv(
  file.path(data_dir, "audio_embedding_strategy_pairwise_similarity.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    source = recode(model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    prompt_method_1 = recode(strategy_1, !!!prompt_method_labels),
    prompt_method_2 = recode(strategy_2, !!!prompt_method_labels),
    prompt_method_1 = factor(prompt_method_1, levels = prompt_method_levels),
    prompt_method_2 = factor(prompt_method_2, levels = prompt_method_levels)
  ) %>%
  arrange(source, prompt_method_1, prompt_method_2)

audio_embedding_model_pairwise_similarity <- read.csv(
  file.path(data_dir, "audio_embedding_model_pairwise_similarity.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels),
    source_1 = recode(model_1, !!!model_labels),
    source_2 = recode(model_2, !!!model_labels),
    source_1 = factor(source_1, levels = audio_embedding_source_levels),
    source_2 = factor(source_2, levels = audio_embedding_source_levels)
  ) %>%
  arrange(prompt_method, source_1, source_2)

audio_strategy_similarity_plot <- build_audio_strategy_similarity_plot(
  audio_embedding_diversity_summary,
  audio_embedding_strategy_pairwise_similarity
)
save_plot_pdf(
  audio_strategy_similarity_plot,
  file.path(figures_dir, "prompts_audio_embedding_strategy_similarity.pdf"),
  10,
  3.2
)

audio_model_similarity_plot <- build_audio_model_similarity_plot(
  audio_embedding_diversity_summary,
  audio_embedding_model_pairwise_similarity
)
save_plot_pdf(
  audio_model_similarity_plot,
  file.path(figures_dir, "prompts_audio_embedding_model_similarity.pdf"),
  8,
  4
)
save_table_outputs(
  audio_embedding_strategy_pairwise_similarity,
  file.path(tables_dir, "prompts_audio_embedding_strategy_similarity_summary")
)
save_table_outputs(
  audio_embedding_model_pairwise_similarity,
  file.path(tables_dir, "prompts_audio_embedding_model_similarity_summary")
)

# Audio embedding neighborhoods ------------------------------------------

audio_embedding_neighbor_strategy_mixing_summary <- read.csv(
  file.path(data_dir, "audio_embedding_neighbor_strategy_mixing_summary.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    source = recode(model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    neighbor_prompt_method = recode(neighbor_strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels),
    neighbor_prompt_method = factor(neighbor_prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(source, prompt_method, neighbor_prompt_method)

audio_embedding_neighbor_model_mixing_summary <- read.csv(
  file.path(data_dir, "audio_embedding_neighbor_model_mixing_summary.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    source = recode(model, !!!model_labels),
    neighbor_source = recode(neighbor_model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    neighbor_source = factor(neighbor_source, levels = audio_embedding_source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(prompt_method, source, neighbor_source)

audio_embedding_neighbor_summary <- read.csv(
  file.path(data_dir, "audio_embedding_neighbor_summary.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    source = recode(model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(source, prompt_method)

audio_embedding_cluster_summary <- read.csv(
  file.path(data_dir, "audio_embedding_cluster_summary_median.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    source = recode(model, !!!model_labels),
    source = factor(source, levels = audio_embedding_source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_method = factor(prompt_method, levels = prompt_method_levels)
  ) %>%
  arrange(source, prompt_method)

audio_neighbor_strategy_mixing_plot <- build_audio_neighbor_strategy_mixing_plot(audio_embedding_neighbor_strategy_mixing_summary)
save_plot_pdf(
  audio_neighbor_strategy_mixing_plot,
  file.path(figures_dir, "prompts_audio_neighbor_strategy_mixing.pdf"),
  10,
  3.2
)

audio_neighbor_model_mixing_plot <- build_audio_neighbor_model_mixing_plot(audio_embedding_neighbor_model_mixing_summary)
save_plot_pdf(
  audio_neighbor_model_mixing_plot,
  file.path(figures_dir, "prompts_audio_neighbor_model_mixing.pdf"),
  8,
  4
)
save_table_outputs(
  audio_embedding_neighbor_strategy_mixing_summary,
  file.path(tables_dir, "prompts_audio_embedding_neighbor_strategy_mixing_summary")
)
save_table_outputs(
  audio_embedding_neighbor_model_mixing_summary,
  file.path(tables_dir, "prompts_audio_embedding_neighbor_model_mixing_summary")
)
save_table_outputs(
  audio_embedding_neighbor_summary,
  file.path(tables_dir, "prompts_audio_embedding_neighbor_summary")
)
save_table_outputs(
  audio_embedding_cluster_summary,
  file.path(tables_dir, "prompts_audio_embedding_cluster_summary")
)

# Audio panel assembly ---------------------------------------------------

audio_panel <- audio_embedding_diversity_plot / audio_strategy_similarity_plot / audio_neighbor_strategy_mixing_plot /
  (audio_model_similarity_plot | audio_neighbor_model_mixing_plot) +
  plot_layout(axes = "collect", heights = c(1.2, 1, 1, 1.8))
save_plot_pdf(
  audio_panel,
  file.path(figures_dir, "panel_audio.pdf"),
  10,
  12
)
save_table_outputs(
  build_audio_aggregate_table(
    audio_embedding_diversity_summary,
    audio_embedding_strategy_pairwise_similarity,
    audio_embedding_model_pairwise_similarity,
    audio_embedding_neighbor_summary,
    audio_embedding_cluster_summary
  ),
  file.path(tables_dir, "panel_audio")
)


# MIR Data Setup ---------------------------------------------------------

mir_raw <- read.csv(file.path(data_dir, "mir_features.csv"), stringsAsFactors = FALSE)
gpu_raw <- read.csv(file.path(data_dir, "gpu_features.csv"), stringsAsFactors = FALSE)

chroma_columns <- names(mir_raw)[str_detect(names(mir_raw), "^chromasum_")]
chroma_matrix <- as.matrix(mir_raw[, chroma_columns])
chroma_probabilities <- chroma_matrix / rowSums(chroma_matrix)
chroma_entropy <- -rowSums(chroma_probabilities * log(chroma_probabilities)) / log(ncol(chroma_matrix))

mir_data <- mir_raw %>%
  mutate(
    source = if_else(
      model == reference_model_key,
      recode(strategy, !!!real_source_labels),
      recode(model, !!!model_labels)
    ),
    source = factor(source, levels = source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_id = as.character(prompt_id),
    cluster_id = as.character(cluster_id),
    chroma_entropy = chroma_entropy,
    chroma_concentration = 1 - chroma_entropy
  )

gpu_data <- gpu_raw %>%
  mutate(
    source = if_else(
      model == reference_model_key,
      recode(strategy, !!!real_source_labels),
      recode(model, !!!model_labels)
    ),
    source = factor(source, levels = source_levels),
    prompt_method = recode(strategy, !!!prompt_method_labels),
    prompt_id = as.character(prompt_id),
    cluster_id = as.character(cluster_id),
    tempo = as.numeric(tempo),
    tempo_fold_60 = ((tempo %% 60) + 60) %% 60,
    tempo_fold_60 = if_else(tempo_fold_60 == 0 & !is.na(tempo), 60, tempo_fold_60),
    mean_similarity = as.numeric(mean_similarity),
    canonical_song_key = na_if(canonical_song_key, ""),
    instruments = na_if(instruments, ""),
    raw_keys = na_if(raw_keys, ""),
    time_signature = if_else(is.na(time_signature), NA_real_, as.numeric(time_signature))
  )

analysis_data <- mir_data %>%
  left_join(
    gpu_data %>%
      select(
        -line_number,
        -audio_filename,
        -audio_basename,
        -audio_stem,
        -audio_id,
        -prompt_id,
        -cluster_id,
        -run_id,
        -source,
        -prompt_method
      ),
    by = c("strategy", "model", "join_key")
  )

# Full categorical plots can get unreadable if instrument/key extraction
# produces a long tail. Adjust these in place when changing plot scope.
instrument_distribution_top_n <- 12
key_distribution_top_n <- 24


# SDD MIR panel assembly -------------------------------------------------

sdd_design_data <- analysis_data %>%
  filter(strategy == sdd_strategy) %>%
  mutate(
    prompt_method = if_else(source == reference_source, reference_prompt_method, sdd_prompt_method),
    prompt_method = factor(prompt_method, levels = c(reference_prompt_method, sdd_prompt_method)),
    source = factor(source, levels = model_levels)
  )

sdd_summary <- summarize_mir_bundle(sdd_design_data)

sdd_distribution_plot <- build_outcome_distribution_plot(
  sdd_design_data,
  outcome_specs,
  facet_prompt_method = FALSE
)
save_plot_pdf(
  sdd_distribution_plot,
  file.path(figures_dir, "panel_sdd_outcome_distributions.pdf"),
  12,
  2.8
)

sdd_tail_plot <- build_tail_support_plot(sdd_summary$tail_summary, facet_prompt_method = FALSE)
save_plot_pdf(
  sdd_tail_plot,
  file.path(figures_dir, "panel_sdd_tail_reach.pdf"),
  8,
  2.8
)

sdd_coverage_plot <- build_human_region_coverage_plot(sdd_summary$region_summary, facet_prompt_method = FALSE)
save_plot_pdf(
  sdd_coverage_plot,
  file.path(figures_dir, "panel_sdd_human_region_coverage.pdf"),
  8,
  2.8
)

sdd_neighbor_plot <- build_neighbor_overlap_plot(sdd_summary$neighbor_summary, facet_prompt_method = FALSE)
save_plot_pdf(
  sdd_neighbor_plot,
  file.path(figures_dir, "panel_sdd_neighbor_overlap.pdf"),
  4,
  2
)

sdd_modes_plot <- build_dominant_modes_plot(sdd_summary$dominant_modes_summary, facet_prompt_method = FALSE)
save_plot_pdf(
  sdd_modes_plot,
  file.path(figures_dir, "panel_sdd_dominant_modes.pdf"),
  6,
  2.8
)

sdd_instrument_distribution_plot <- build_categorical_distribution_plot(
  sdd_design_data,
  "instruments",
  "Instrument",
  separator = "\\|",
  top_n = instrument_distribution_top_n,
  facet_prompt_method = FALSE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  sdd_instrument_distribution_plot,
  file.path(figures_dir, "sdd_instrument_distribution.pdf"),
  8,
  3.8
)

sdd_time_signature_distribution_plot <- build_categorical_distribution_plot(
  sdd_design_data,
  "time_signature",
  "Time Signature",
  facet_prompt_method = FALSE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  sdd_time_signature_distribution_plot,
  file.path(figures_dir, "sdd_time_signature_distribution.pdf"),
  8,
  3
)

sdd_key_distribution_plot <- build_categorical_distribution_plot(
  sdd_design_data,
  "canonical_song_key",
  "Key",
  top_n = key_distribution_top_n,
  facet_prompt_method = FALSE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  sdd_key_distribution_plot,
  file.path(figures_dir, "sdd_key_distribution.pdf"),
  8,
  4.5
)

sdd_support_width_plot <- build_support_width_plot(sdd_summary$support_summary, facet_prompt_method = FALSE)
save_plot_pdf(
  sdd_support_width_plot,
  file.path(figures_dir, "sdd_support_width.pdf"),
  6,
  2.8
)

sdd_support_interval_plot <- build_reference_interval_suite(
  sdd_summary$support_summary,
  reference_interval_outcome_names,
  facet_prompt_method = FALSE
)
save_plot_pdf(
  sdd_support_interval_plot,
  file.path(figures_dir, "sdd_support_intervals.pdf"),
  10,
  1.8
)

sdd_within_feature_model_heatmap <- build_within_feature_model_heatmap(sdd_summary$within_feature_model_similarity_summary)
save_plot_pdf(
  sdd_within_feature_model_heatmap,
  file.path(figures_dir, "sdd_within_feature_model_heatmap.pdf"),
  14,
  2.8
)

sdd_within_model_feature_heatmap <- build_within_model_feature_heatmap(sdd_summary$within_model_feature_similarity_summary)
save_plot_pdf(
  sdd_within_model_feature_heatmap,
  file.path(figures_dir, "sdd_within_model_feature_heatmap.pdf"),
  12,
  4
)

sdd_panel <- (sdd_distribution_plot /
  ((sdd_tail_plot | sdd_neighbor_plot) + plot_layout(widths = c(2, 1))) /
  ((sdd_coverage_plot | sdd_modes_plot) + plot_layout(widths = c(2, 1)))) +
  plot_layout(heights = c(1, 1, 1.4))

save_plot_pdf(
  sdd_panel,
  file.path(figures_dir, "panel_sdd.pdf"),
  14,
  10
)

save_table_outputs(
  build_mir_aggregate_table(
    sdd_summary$support_summary,
    sdd_summary$tail_summary,
    sdd_summary$region_summary,
    sdd_summary$neighbor_summary,
    sdd_summary$dominant_modes_summary
  ),
  file.path(tables_dir, "panel_sdd")
)
save_mir_supplemental_tables(
  sdd_summary,
  "sdd",
  tables_dir
)


# Generator MIR panel assembly ------------------------------------------

# Gen uses Jamendo as the human reference distribution
gen_reference_sources <- unname(real_source_labels[gen_reference_strategies])
gen_reference_source <- gen_reference_sources[1]
gen_source_levels <- c(gen_reference_sources, generator_levels)

gen_design_data <- analysis_data %>%
  filter(strategy %in% c(gen_synthetic_strategies, gen_reference_strategies)) %>%
  mutate(
    prompt_method = if_else(
      strategy %in% gen_reference_strategies,
      reference_prompt_method,
      as.character(prompt_method)
    ),
    prompt_method = factor(
      prompt_method,
      levels = c(reference_prompt_method, synthetic_prompt_method_levels)
    ),
    source = factor(source, levels = gen_source_levels)
  ) %>%
  filter(!is.na(prompt_method))

# Main generator panel aggregates all non-reference prompt methods
gen_main_design_data <- gen_design_data %>%
  mutate(
    prompt_method = if_else(
      as.character(source) %in% gen_reference_sources,
      reference_prompt_method,
      main_story_prompt_method
    ),
    prompt_method = factor(
      prompt_method,
      levels = c(reference_prompt_method, main_story_prompt_method)
    ),
    prompt_id = if_else(
      as.character(source) %in% gen_reference_sources,
      prompt_id,
      paste(strategy, prompt_id, sep = "::")
    )
  )

gen_summary <- summarize_mir_bundle(
  gen_main_design_data,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
gen_supplemental_summary <- summarize_mir_bundle(
  gen_design_data,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)

gen_distribution_plot <- build_outcome_distribution_plot(
  gen_main_design_data %>% subset(n_instruments <= 10), # Subset for summary plot
  outcome_specs,
  facet_prompt_method = FALSE
)
save_plot_pdf(
  gen_distribution_plot,
  file.path(figures_dir, "panel_gen_outcome_distributions.pdf"),
  12,
  2.8
)

gen_tail_plot <- build_tail_support_plot(
  gen_summary$tail_summary,
  facet_prompt_method = FALSE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_tail_plot,
  file.path(figures_dir, "panel_gen_tail_reach.pdf"),
  8,
  2.4
)

gen_coverage_plot <- build_human_region_coverage_plot(
  gen_summary$region_summary,
  facet_prompt_method = FALSE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_coverage_plot,
  file.path(figures_dir, "panel_gen_human_region_coverage.pdf"),
  8,
  2.4
)

gen_neighbor_plot <- build_neighbor_overlap_plot(
  gen_summary$neighbor_summary,
  facet_prompt_method = FALSE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_neighbor_plot,
  file.path(figures_dir, "panel_gen_neighbor_overlap.pdf"),
  4,
  2
)

gen_modes_plot <- build_dominant_modes_plot(
  gen_summary$dominant_modes_summary,
  facet_prompt_method = FALSE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_modes_plot,
  file.path(figures_dir, "panel_gen_dominant_modes.pdf"),
  6,
  2.8
)

gen_distribution_split_plot <- build_outcome_distribution_plot(
  gen_design_data %>% subset(n_instruments <= 10), # Subset for summary plot
  outcome_specs,
  facet_prompt_method = TRUE
)
save_plot_pdf(
  gen_distribution_split_plot,
  file.path(figures_dir, "gen_outcome_distributions.pdf"),
  14,
  8
)

gen_support_width_plot <- build_support_width_plot(
  gen_supplemental_summary$support_summary,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_support_width_plot,
  file.path(figures_dir, "gen_support_width.pdf"),
  8,
  3.2
)

gen_tail_split_plot <- build_tail_support_plot(
  gen_supplemental_summary$tail_summary,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_tail_split_plot,
  file.path(figures_dir, "gen_tail_reach_curves.pdf"),
  10,
  8
)

gen_coverage_split_plot <- build_human_region_coverage_plot(
  gen_supplemental_summary$region_summary,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_coverage_split_plot,
  file.path(figures_dir, "gen_human_region_coverage.pdf"),
  8,
  6
)

gen_support_interval_plot <- build_reference_interval_suite(
  gen_supplemental_summary$support_summary,
  reference_interval_outcome_names,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_support_interval_plot,
  file.path(figures_dir, "gen_support_intervals.pdf"),
  14,
  4
)

gen_neighbor_split_plot <- build_neighbor_overlap_plot(
  gen_supplemental_summary$neighbor_summary,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_neighbor_split_plot,
  file.path(figures_dir, "gen_neighbor_overlap.pdf"),
  4,
  6
)

gen_modes_split_plot <- build_dominant_modes_plot(
  gen_supplemental_summary$dominant_modes_summary,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_modes_split_plot,
  file.path(figures_dir, "gen_dominant_modes.pdf"),
  10,
  6
)

gen_instrument_distribution_plot <- build_categorical_distribution_plot(
  gen_design_data,
  "instruments",
  "Instrument",
  separator = "\\|",
  top_n = instrument_distribution_top_n,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_instrument_distribution_plot,
  file.path(figures_dir, "gen_instrument_distribution.pdf"),
  10,
  8
)

gen_time_signature_distribution_plot <- build_categorical_distribution_plot(
  gen_design_data,
  "time_signature",
  "Time Signature",
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_time_signature_distribution_plot,
  file.path(figures_dir, "gen_time_signature_distribution.pdf"),
  10,
  7
)

gen_key_distribution_plot <- build_categorical_distribution_plot(
  gen_design_data,
  "canonical_song_key",
  "Key",
  top_n = key_distribution_top_n,
  facet_prompt_method = TRUE,
  reference_source_name = gen_reference_source,
  reference_prompt_method_name = reference_prompt_method
)
save_plot_pdf(
  gen_key_distribution_plot,
  file.path(figures_dir, "gen_key_distribution.pdf"),
  10,
  12
)

gen_within_feature_model_heatmap <- build_within_feature_model_heatmap(gen_supplemental_summary$within_feature_model_similarity_summary)
save_plot_pdf(
  gen_within_feature_model_heatmap,
  file.path(figures_dir, "gen_within_feature_model_heatmap.pdf"),
  14,
  2.8
)

gen_within_model_feature_heatmap <- build_within_model_feature_heatmap(gen_supplemental_summary$within_model_feature_similarity_summary)
save_plot_pdf(
  gen_within_model_feature_heatmap,
  file.path(figures_dir, "gen_within_model_feature_heatmap.pdf"),
  10,
  4
)

gen_panel <- (gen_distribution_plot /
  ((gen_tail_plot | gen_neighbor_plot) + plot_layout(widths = c(2, 1))) /
  ((gen_coverage_plot | gen_modes_plot) + plot_layout(widths = c(2, 1)))) +
  plot_layout(heights = c(1, 1, 1.4))

save_plot_pdf(
  gen_panel,
  file.path(figures_dir, "panel_gen.pdf"),
  14,
  10
)

save_table_outputs(
  build_mir_aggregate_table(
    gen_summary$support_summary,
    gen_summary$tail_summary,
    gen_summary$region_summary,
    gen_summary$neighbor_summary,
    gen_summary$dominant_modes_summary,
    reference_source_name = gen_reference_source,
    reference_prompt_method_name = reference_prompt_method
  ),
  file.path(tables_dir, "panel_gen")
)
save_mir_supplemental_tables(
  gen_supplemental_summary,
  "gen",
  tables_dir
)


# Statistical analysis ---------------------------------------------------

stats_source_levels <- c(reference_source, generator_levels, gen_reference_source)
stats_strategy_levels <- c(gen_synthetic_strategies, sdd_strategy, "real")

stats_data <- bind_rows(sdd_design_data, gen_design_data) %>%
  mutate(
    strategy = if_else(as.character(source) %in% c(reference_source, gen_reference_source), "real", strategy),
    strategy = factor(strategy, levels = stats_strategy_levels),
    source = factor(as.character(source), levels = stats_source_levels)
  )
stats_reference_source_index <- which(levels(droplevels(stats_data$source)) == reference_source)

# Mean-level source contrasts --------------------------------------------

stats_chroma_concentration_model <- stats_data %>%
  feols(
    chroma_concentration ~ source * strategy,
    vcov = ~ prompt_id
  ) %>%
  emmeans(~ source, data = stats_data)
stats_chroma_concentration_contrasts <- stats_chroma_concentration_model %>%
  contrast(method = "trt.vs.ctrl", ref = stats_reference_source_index, adjust = "BH")
stats_chroma_concentration_contrast_table <- stats_chroma_concentration_contrasts %>%
  summary(infer = TRUE) %>%
  as_tibble() %>%
  mutate(outcome = "Chroma Concentration", .before = 1)
save_table_outputs(
  stats_chroma_concentration_contrast_table,
  file.path(tables_dir, "stats_chroma_concentration_source_contrasts")
)

stats_longest_silence_model <- stats_data %>%
  feols(
    longest_silence_ms ~ source * strategy,
    vcov = ~ prompt_id
  ) %>%
  emmeans(~ source, data = stats_data)
stats_longest_silence_contrasts <- stats_longest_silence_model %>%
  contrast(method = "trt.vs.ctrl", ref = stats_reference_source_index, adjust = "BH")
stats_longest_silence_contrast_table <- stats_longest_silence_contrasts %>%
  summary(infer = TRUE) %>%
  as_tibble() %>%
  mutate(outcome = "Longest Silence", .before = 1)
save_table_outputs(
  stats_longest_silence_contrast_table,
  file.path(tables_dir, "stats_longest_silence_source_contrasts")
)

stats_nonzero_silence_model <- stats_data %>%
  feols(
    nonzero_silence ~ source * strategy,
    vcov = ~ prompt_id
  ) %>%
  emmeans(~ source, data = stats_data)
stats_nonzero_silence_contrasts <- stats_nonzero_silence_model %>%
  contrast(method = "trt.vs.ctrl", ref = stats_reference_source_index, adjust = "BH")
stats_nonzero_silence_contrast_table <- stats_nonzero_silence_contrasts %>%
  summary(infer = TRUE) %>%
  as_tibble() %>%
  mutate(outcome = "Any Silence", .before = 1)
save_table_outputs(
  stats_nonzero_silence_contrast_table,
  file.path(tables_dir, "stats_nonzero_silence_source_contrasts")
)

# Distribution-level KS-style contrasts ----------------------------------

stats_ks_test_against_reference <- function(
  data,
  outcome_column,
  outcome_label,
  reference_source_name = reference_source
) {
  working <- data %>%
    transmute(source, value = .data[[outcome_column]]) %>%
    filter(!is.na(value))

  reference_values <- working %>%
    filter(source == reference_source_name) %>%
    pull(value)

  working %>%
    filter(source != reference_source_name) %>%
    group_by(source) %>%
    group_modify(\(.x, .y) {
      test <- suppressWarnings(ks.test(.x$value, reference_values, exact = FALSE))
      tibble(
        n = length(.x$value),
        reference_n = length(reference_values),
        statistic = unname(test$statistic),
        p.value = test$p.value
      )
    }) %>%
    ungroup() %>%
    mutate(
      outcome = outcome_label,
      p_adj = p.adjust(p.value, method = "BH"),
      stars = significance_stars(p_adj),
      .before = 1
    )
}

stats_distribution_ks_tests <- bind_rows(
  stats_ks_test_against_reference(stats_data, "rmsstd", "RMS Std."),
  stats_ks_test_against_reference(stats_data, "tempo_fold_60", "Folded Tempo (60-BPM)")
)
save_table_outputs(
  stats_distribution_ks_tests,
  file.path(tables_dir, "stats_distribution_ks_tests")
)

# Variation-level source contrasts ---------------------------------------

# Brown-Forsythe-style analogue to the FEOLS workflow
stats_variability_data <- stats_data %>%
  group_by(source, strategy) %>%
  mutate(
    rmsstd_abs_deviation = abs(rmsstd - median(rmsstd, na.rm = TRUE)),
    tempo_fold_60_abs_deviation = abs(tempo_fold_60 - median(tempo_fold_60, na.rm = TRUE))
  ) %>%
  ungroup()

stats_rmsstd_variability_model <- stats_variability_data %>%
  feols(
    rmsstd_abs_deviation ~ source * strategy,
    vcov = ~ prompt_id
  ) %>%
  emmeans(~ source, data = stats_variability_data)
stats_rmsstd_variability_contrasts <- stats_rmsstd_variability_model %>%
  contrast(method = "trt.vs.ctrl", ref = stats_reference_source_index, adjust = "BH")
stats_rmsstd_variability_contrast_table <- stats_rmsstd_variability_contrasts %>%
  summary(infer = TRUE) %>%
  as_tibble() %>%
  mutate(outcome = "RMS Std. Absolute Deviation", .before = 1)

stats_folded_tempo_variability_model <- stats_variability_data %>%
  feols(
    tempo_fold_60_abs_deviation ~ source * strategy,
    vcov = ~ prompt_id
  ) %>%
  emmeans(~ source, data = stats_variability_data)
stats_folded_tempo_variability_contrasts <- stats_folded_tempo_variability_model %>%
  contrast(method = "trt.vs.ctrl", ref = stats_reference_source_index, adjust = "BH")
stats_folded_tempo_variability_contrast_table <- stats_folded_tempo_variability_contrasts %>%
  summary(infer = TRUE) %>%
  as_tibble() %>%
  mutate(outcome = "Folded Tempo Absolute Deviation", .before = 1)

stats_variability_contrast_table <- bind_rows(
  stats_rmsstd_variability_contrast_table,
  stats_folded_tempo_variability_contrast_table
)
save_table_outputs(
  stats_variability_contrast_table,
  file.path(tables_dir, "stats_variability_source_contrasts")
)
