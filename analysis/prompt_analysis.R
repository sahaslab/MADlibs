library(tidyverse)
library(patchwork)

build_prompt_support_rank_plot <- function(prompt_diversity_summary) {
  unique_share_plot <- ggplot(
    prompt_diversity_summary %>% transmute(prompt_method, value = distinct_prompt_text_share),
    aes(value, prompt_method, fill = prompt_method)
  ) +
    geom_col(width = 0.72) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = c(0.01, 0.1, 1),
      labels = c("1%", "10%", "100%"),
      expand = expansion(mult = c(0, 0.1)),
      transform = scales::pseudo_log_trans(sigma = 0.01)
    ) +
    scale_fill_manual(values = prompt_method_palette) +
    labs(x = NULL, y = NULL, title = "Unique prompt share") +
    theme_analysis() +
    theme(
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      plot.title = element_text(size = 9, face = "bold")
    )

  effective_count_plot <- ggplot(
    prompt_diversity_summary %>% transmute(prompt_method, value = effective_prompt_text_count),
    aes(value, prompt_method, fill = prompt_method)
  ) +
    geom_col(width = 0.72) +
    scale_x_log10(
      labels = scales::label_number(accuracy = 1),
      expand = expansion(mult = c(0, 0.1))
    ) +
    scale_fill_manual(values = prompt_method_palette) +
    labs(x = NULL, y = NULL, title = "Effective prompt count") +
    theme_analysis() +
    theme(
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      plot.title = element_text(size = 9, face = "bold")
    )

  rank_plot <- ggplot(
    prompt_diversity_summary %>% transmute(prompt_method, value = effective_rank_centered),
    aes(value, prompt_method, fill = prompt_method)
  ) +
    geom_col(width = 0.72) +
    scale_fill_manual(values = prompt_method_palette) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
    labs(x = NULL, y = NULL, title = "Embedding effective rank") +
    theme_analysis() +
    theme(
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      plot.title = element_text(size = 9, face = "bold")
    )
  
  metric_plot_data <- prompt_diversity_summary %>%
    transmute(prompt_method, value = within_mean_cosine_distance, lci = within_mean_cosine_distance_lci, uci = within_mean_cosine_distance_uci)

  cosine_plot <- ggplot(metric_plot_data, aes(value, prompt_method, fill = prompt_method)) +
    geom_col(width = 0.72) +
    geom_errorbar(aes(xmin = lci, xmax = uci), width = 0.2, linewidth = 0.4, na.rm = TRUE) +
    scale_fill_manual(values = prompt_method_palette) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
    labs(x = NULL, y = NULL, title = "Mean cosine distance") +
    theme_analysis() +
    theme(
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      plot.title = element_text(size = 9, face = "bold")
    )

  (unique_share_plot | effective_count_plot | rank_plot | cosine_plot) + plot_layout(axes = "collect")
}

build_prompt_strategy_similarity_plot <- function(
  prompt_diversity_summary,
  prompt_strategy_similarity_summary
) {
  diagonal_rows <- prompt_diversity_summary %>%
    transmute(prompt_method_1 = prompt_method, prompt_method_2 = prompt_method, similarity = within_mean_cosine)

  upper_rows <- prompt_strategy_similarity_summary %>%
    transmute(prompt_method_1, prompt_method_2, similarity = between_mean_cosine)

  lower_rows <- prompt_strategy_similarity_summary %>%
    mutate(
      mirrored_prompt_method_1 = prompt_method_2,
      mirrored_prompt_method_2 = prompt_method_1
    ) %>%
    transmute(prompt_method_1 = mirrored_prompt_method_1, prompt_method_2 = mirrored_prompt_method_2, similarity = between_mean_cosine)

  plot_data <- bind_rows(diagonal_rows, upper_rows, lower_rows) %>%
    distinct(prompt_method_1, prompt_method_2, .keep_all = TRUE) %>%
    mutate(similarity_label = sprintf("%.2f", similarity))

  ggplot(plot_data, aes(x = prompt_method_2, y = prompt_method_1, fill = similarity)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = similarity_label), size = 3.1) +
    scale_fill_gradient2(low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0.7, limits = c(0.55, 1.0)) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_prompt_neighbor_mixing_plot <- function(prompt_neighbor_mixing_summary) {
  plot_data <- prompt_neighbor_mixing_summary %>%
    group_by(prompt_method, neighbor_prompt_method) %>%
    summarize(mean_neighbor_share = mean(mean_neighbor_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(share_label = sprintf("%.2f", mean_neighbor_share))

  ggplot(plot_data, aes(x = neighbor_prompt_method, y = prompt_method, fill = mean_neighbor_share)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = share_label), size = 3.0) +
    scale_fill_gradient(low = "#F7F7F7", high = "#1F4E79", limits = c(0, 1)) +
    labs(x = "Neighbor method", y = "Origin method") +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_audio_embedding_diversity_plot <- function(audio_embedding_diversity_summary) {
  distance_metric <- unname(audio_embedding_diversity_metric_labels["within_mean_cosine_distance"])
  rank_metric <- unname(audio_embedding_diversity_metric_labels["effective_rank_centered"])

  reference_lines <- audio_embedding_diversity_summary %>%
    filter(source == reference_source) %>%
    summarize(
      mean_cosine_distance = mean(within_mean_cosine_distance, na.rm = TRUE),
      embedding_effective_rank = mean(effective_rank_centered, na.rm = TRUE)
    ) %>%
    pivot_longer(
      cols = everything(),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = recode(
        metric,
        "mean_cosine_distance" = distance_metric,
        "embedding_effective_rank" = rank_metric
      ),
      metric = factor(metric, levels = audio_embedding_diversity_metric_levels)
    )

  distance_rows <- audio_embedding_diversity_summary %>%
    filter(source != reference_source) %>%
    transmute(
      source,
      prompt_method,
      metric = distance_metric,
      value = within_mean_cosine_distance,
      lci = within_mean_cosine_distance_lci,
      uci = within_mean_cosine_distance_uci
    )

  rank_rows <- audio_embedding_diversity_summary %>%
    filter(source != reference_source) %>%
    transmute(
      source,
      prompt_method,
      metric = rank_metric,
      value = effective_rank_centered,
      lci = NA_real_,
      uci = NA_real_
    )

  metric_plot_data <- bind_rows(distance_rows, rank_rows) %>%
    mutate(metric = factor(metric, levels = audio_embedding_diversity_metric_levels))

  ggplot(metric_plot_data, aes(x = prompt_method, y = value, fill = prompt_method)) +
    geom_col(alpha = 0.9, width = 0.72) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = 0.12, linewidth = 0.4, na.rm = TRUE) +
    geom_hline(
      data = reference_lines,
      aes(yintercept = value, linetype = audio_embedding_reference_label),
      color = source_palette[[reference_source]],
      linewidth = 0.4
    ) +
    geom_text(
      data = reference_lines %>% mutate(source = factor(generator_levels[1], levels = audio_embedding_source_levels)),
      inherit.aes = FALSE,
      aes(x = 1.5, y = value * 1.12, label = audio_embedding_reference_label),
      label = "Real (SDD)",
      size = 3,
      color = source_palette[[reference_source]]
    ) +
    facet_grid(
      metric ~ source,
      scales = "free_y",
      labeller = labeller(metric = label_wrap_gen(width = 18))
    ) +
    scale_fill_manual(values = prompt_method_palette, guide = "none") +
    scale_linetype_manual(values = setNames("dashed", audio_embedding_reference_label)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
    labs(x = "Prompt method", y = NULL) +
    theme_analysis() +
    theme(
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      panel.spacing = unit(0.35, "lines")
    )
}

build_audio_strategy_similarity_plot <- function(
  audio_embedding_diversity_summary,
  audio_embedding_strategy_pairwise_similarity
) {
  diagonal_rows <- audio_embedding_diversity_summary %>%
    transmute(source, prompt_method_1 = prompt_method, prompt_method_2 = prompt_method, similarity = within_mean_cosine)

  upper_rows <- audio_embedding_strategy_pairwise_similarity %>%
    transmute(source, prompt_method_1, prompt_method_2, similarity = between_mean_cosine)

  lower_rows <- audio_embedding_strategy_pairwise_similarity %>%
    mutate(
      mirrored_prompt_method_1 = prompt_method_2,
      mirrored_prompt_method_2 = prompt_method_1
    ) %>%
    transmute(source, prompt_method_1 = mirrored_prompt_method_1, prompt_method_2 = mirrored_prompt_method_2, similarity = between_mean_cosine)

  plot_data <- bind_rows(diagonal_rows, upper_rows, lower_rows) %>%
    distinct(source, prompt_method_1, prompt_method_2, .keep_all = TRUE) %>%
    mutate(similarity_label = sprintf("%.2f", similarity))

  ggplot(plot_data, aes(x = prompt_method_2, y = prompt_method_1, fill = similarity)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = similarity_label), size = 2.5) +
    facet_wrap(vars(source), nrow = 1) +
    scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", midpoint = 0.5) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_audio_model_similarity_plot <- function(
  audio_embedding_diversity_summary,
  audio_embedding_model_pairwise_similarity
) {
  diagonal_rows <- audio_embedding_diversity_summary %>%
    transmute(prompt_method, source_1 = source, source_2 = source, similarity = within_mean_cosine)

  upper_rows <- audio_embedding_model_pairwise_similarity %>%
    transmute(prompt_method, source_1, source_2, similarity = between_mean_cosine)

  lower_rows <- audio_embedding_model_pairwise_similarity %>%
    mutate(
      mirrored_source_1 = source_2,
      mirrored_source_2 = source_1
    ) %>%
    transmute(prompt_method, source_1 = mirrored_source_1, source_2 = mirrored_source_2, similarity = between_mean_cosine)

  plot_data <- bind_rows(diagonal_rows, upper_rows, lower_rows) %>%
    distinct(prompt_method, source_1, source_2, .keep_all = TRUE) %>%
    mutate(similarity_label = sprintf("%.2f", similarity))

  ggplot(plot_data, aes(x = source_2, y = source_1, fill = similarity)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = similarity_label), size = 2.7) +
    facet_wrap(vars(prompt_method), ncol = 3) +
    scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", midpoint = 0.5) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_audio_neighbor_strategy_mixing_plot <- function(audio_embedding_neighbor_strategy_mixing_summary) {
  plot_data <- audio_embedding_neighbor_strategy_mixing_summary %>%
    group_by(source, prompt_method, neighbor_prompt_method) %>%
    summarize(mean_neighbor_share = mean(mean_neighbor_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(share_label = sprintf("%.2f", mean_neighbor_share))

  ggplot(plot_data, aes(x = neighbor_prompt_method, y = prompt_method, fill = mean_neighbor_share)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = share_label), size = 2.6) +
    facet_wrap(vars(source), nrow = 1) +
    scale_fill_gradient(low = "#F7F7F7", high = "#1F4E79", limits = c(0, 1)) +
    labs(x = "Neighbor method", y = "Origin method") +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_audio_neighbor_model_mixing_plot <- function(audio_embedding_neighbor_model_mixing_summary) {
  plot_data <- audio_embedding_neighbor_model_mixing_summary %>%
    group_by(prompt_method, source, neighbor_source) %>%
    summarize(mean_neighbor_share = mean(mean_neighbor_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(share_label = sprintf("%.2f", mean_neighbor_share))

  ggplot(plot_data, aes(x = neighbor_source, y = source, fill = mean_neighbor_share)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = share_label), size = 2.6) +
    facet_wrap(vars(prompt_method), ncol = 3) +
    scale_fill_gradient(low = "#F7F7F7", high = "#1F4E79", limits = c(0, 1)) +
    labs(x = "Neighbor source", y = "Origin source") +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_prompt_aggregate_table <- function(
  prompt_diversity_summary,
  prompt_strategy_similarity_summary,
  prompt_neighbor_mixing_summary,
  prompt_embedding_cluster_summary = NULL
) {
  similarity_summary <- prompt_strategy_similarity_summary %>%
    group_by(prompt_method_1) %>%
    summarize(
      mean_between_prompt_cosine = mean(between_mean_cosine, na.rm = TRUE),
      closest_prompt_method = as.character(prompt_method_2[which.max(between_mean_cosine)]),
      .groups = "drop"
    ) %>%
    rename(prompt_method = prompt_method_1)

  neighbor_summary <- prompt_neighbor_mixing_summary %>%
    filter(prompt_method == neighbor_prompt_method) %>%
    select(prompt_method, self_prompt_neighbor_share = mean_neighbor_share)

  output <- prompt_diversity_summary %>%
    select(
      prompt_method,
      prompt_count,
      distinct_prompt_text_share,
      effective_prompt_text_count,
      effective_rank_centered,
      within_mean_cosine_distance,
      same_strategy_neighbor_share,
      effective_neighbor_strategy_count
    ) %>%
    left_join(similarity_summary, by = "prompt_method") %>%
    left_join(neighbor_summary, by = "prompt_method")

  if (!is.null(prompt_embedding_cluster_summary)) {
    output <- output %>%
      left_join(
        prompt_embedding_cluster_summary %>%
          select(prompt_method, prompt_cluster_effective_count = effective_cluster_count, prompt_cluster_top_1_mass = top_1_mass, prompt_cluster_top_3_mass = top_3_mass),
        by = "prompt_method"
      )
  }

  output
}

build_audio_aggregate_table <- function(
  audio_embedding_diversity_summary,
  audio_embedding_strategy_pairwise_similarity = NULL,
  audio_embedding_model_pairwise_similarity = NULL,
  audio_embedding_neighbor_summary = NULL,
  audio_embedding_cluster_summary = NULL
) {
  audio_summary <- audio_embedding_diversity_summary %>%
    select(
      source,
      prompt_method,
      audio_count,
      effective_rank_centered,
      within_mean_cosine_distance,
      same_strategy_audio_neighbor_share,
      same_model_audio_neighbor_share,
      real_audio_neighbor_share,
      effective_audio_neighbor_strategy_count,
      effective_audio_neighbor_model_count
    )

  if (!is.null(audio_embedding_strategy_pairwise_similarity)) {
    strategy_summary <- audio_embedding_strategy_pairwise_similarity %>%
      group_by(source, prompt_method_1) %>%
      summarize(
        mean_between_prompt_audio_cosine = mean(between_mean_cosine, na.rm = TRUE),
        closest_audio_prompt_method = as.character(prompt_method_2[which.max(between_mean_cosine)]),
        .groups = "drop"
      ) %>%
      rename(prompt_method = prompt_method_1)
    audio_summary <- audio_summary %>% left_join(strategy_summary, by = c("source", "prompt_method"))
  }

  if (!is.null(audio_embedding_model_pairwise_similarity)) {
    model_summary <- audio_embedding_model_pairwise_similarity %>%
      group_by(prompt_method, source_1) %>%
      summarize(
        mean_between_model_audio_cosine = mean(between_mean_cosine, na.rm = TRUE),
        closest_audio_model = as.character(source_2[which.max(between_mean_cosine)]),
        .groups = "drop"
      ) %>%
      rename(source = source_1)
    audio_summary <- audio_summary %>% left_join(model_summary, by = c("source", "prompt_method"))
  }

  if (!is.null(audio_embedding_neighbor_summary)) {
    audio_summary <- audio_summary %>%
      left_join(
        audio_embedding_neighbor_summary %>%
          select(source, prompt_method, embedding_same_source_share = same_source_share, embedding_reference_share = reference_share, embedding_effective_neighbor_source_count = effective_neighbor_source_count),
        by = c("source", "prompt_method")
      )
  }

  if (!is.null(audio_embedding_cluster_summary)) {
    audio_summary <- audio_summary %>%
      left_join(
        audio_embedding_cluster_summary %>%
          select(source, prompt_method, audio_cluster_effective_count = effective_cluster_count, audio_human_cluster_coverage = human_cluster_coverage, audio_cluster_top_1_mass = top_1_mass, audio_cluster_top_3_mass = top_3_mass),
        by = c("source", "prompt_method")
      )
  }

  audio_summary
}
