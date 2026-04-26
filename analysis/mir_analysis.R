library(tidyverse)

summarize_mir_bundle <- function(
  data,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  reference_outcome_specs <- outcome_specs %>%
    filter(label %in% support_plot_outcome_levels)
  feature_space_data <- build_feature_space_table(data)
  feature_space_analysis_data <- cap_feature_space_rows(
    feature_space_data,
    max_rows_per_cell = distance_row_cap
  )
  within_feature_by_prompt <- summarize_model_feature_similarity_by_prompt_method(data)
  within_model_by_prompt <- summarize_within_model_feature_correlation_by_prompt_method(data)

  list(
    descriptive_summary = summarize_outcomes(data, outcome_specs),
    support_summary = summarize_support_against_reference(
      data,
      reference_outcome_specs,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    tail_summary = summarize_tail_support_against_reference(
      data,
      reference_outcome_specs,
      tail_probabilities = tail_probabilities,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    region_summary = summarize_human_region_coverage(
      data,
      reference_outcome_specs,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    neighbor_summary = summarize_neighbor_overlap(
      feature_space_analysis_data,
      neighbor_k = neighbor_k,
      reference_source_name = reference_source_name
    ),
    dominant_modes_summary = summarize_dominant_modes(data),
    within_feature_model_similarity_by_prompt_method = within_feature_by_prompt,
    within_feature_model_similarity_summary = summarize_model_feature_similarity(within_feature_by_prompt),
    within_model_feature_similarity_by_prompt_method = within_model_by_prompt,
    within_model_feature_similarity_summary = summarize_within_model_feature_correlation(within_model_by_prompt),
    chroma_profile = summarize_chroma_profile(data),
    chroma_contrasts = compute_chroma_contrasts(
      data,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    instrument_contrasts = compute_categorical_contrasts(
      data,
      "instruments",
      separator = "\\|",
      top_n = 8,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    key_contrasts = compute_categorical_contrasts(
      data,
      "canonical_song_key",
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    time_signature_contrasts = compute_categorical_contrasts(
      data,
      "time_signature",
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    ),
    fits = fit_outcomes(
      data,
      outcome_specs,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    )
  )
}

save_mir_supplemental_tables <- function(
  summary,
  prefix,
  tables_dir
) {
  support_width_table <- summary$support_summary %>%
    filter(family == "continuous") %>%
    select(prompt_method, source, outcome, outcome_name, iqr_ratio, p80_ratio, overlap_ratio)
  folded_tempo_support_table <- summary$support_summary %>%
    filter(outcome_name == "tempo_fold_60")
  support_intervals_table <- summary$support_summary %>%
    filter(outcome_name %in% reference_interval_outcome_names)

  save_table_outputs(summary$descriptive_summary, file.path(tables_dir, sprintf("%s_descriptive_summary", prefix)))
  save_table_outputs(summary$support_summary, file.path(tables_dir, sprintf("%s_support_summary", prefix)))
  save_table_outputs(summary$tail_summary, file.path(tables_dir, sprintf("%s_tail_support_summary", prefix)))
  save_table_outputs(summary$region_summary, file.path(tables_dir, sprintf("%s_human_region_coverage_summary", prefix)))
  save_table_outputs(summary$neighbor_summary, file.path(tables_dir, sprintf("%s_neighbor_overlap_summary", prefix)))
  save_table_outputs(summary$dominant_modes_summary, file.path(tables_dir, sprintf("%s_dominant_modes_summary", prefix)))
  save_table_outputs(summary$within_feature_model_similarity_by_prompt_method, file.path(tables_dir, sprintf("%s_within_feature_model_similarity_by_prompt_method", prefix)))
  save_table_outputs(summary$within_feature_model_similarity_summary, file.path(tables_dir, sprintf("%s_within_feature_model_similarity_summary", prefix)))
  save_table_outputs(summary$within_model_feature_similarity_by_prompt_method, file.path(tables_dir, sprintf("%s_within_model_feature_similarity_by_prompt_method", prefix)))
  save_table_outputs(summary$within_model_feature_similarity_summary, file.path(tables_dir, sprintf("%s_within_model_feature_similarity_summary", prefix)))
  save_table_outputs(support_width_table, file.path(tables_dir, sprintf("%s_support_width", prefix)))
  save_table_outputs(folded_tempo_support_table, file.path(tables_dir, sprintf("%s_folded_tempo_support", prefix)))
  save_table_outputs(support_intervals_table, file.path(tables_dir, sprintf("%s_support_intervals", prefix)))
  save_table_outputs(summary$chroma_contrasts, file.path(tables_dir, sprintf("%s_chroma_vs_real", prefix)))
  save_table_outputs(summary$chroma_profile, file.path(tables_dir, sprintf("%s_chroma_profile", prefix)))
  save_table_outputs(summary$instrument_contrasts, file.path(tables_dir, sprintf("%s_instrument_vs_real", prefix)))
  save_table_outputs(summary$key_contrasts, file.path(tables_dir, sprintf("%s_key_vs_real", prefix)))
  save_table_outputs(summary$time_signature_contrasts, file.path(tables_dir, sprintf("%s_time_signature_vs_real", prefix)))
  save_table_outputs(summary$fits$summary, file.path(tables_dir, sprintf("%s_fit_cell_summary", prefix)))
  save_table_outputs(summary$fits$pairwise, file.path(tables_dir, sprintf("%s_fit_pairwise", prefix)))
  save_table_outputs(summary$fits$control, file.path(tables_dir, sprintf("%s_fit_vs_real_reference", prefix)))
}
