library(tidyverse)
library(patchwork)
library(ggridges)

theme_analysis <- function() {
  theme_minimal(base_size = 12) +
    theme(
      axis.title = element_text(face = "bold"),
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.line.x = element_line(color = "black", linewidth = 0.4),
      axis.line.y = element_line(color = "black", linewidth = 0.4),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(4, 6, 4, 4),
      strip.background = element_rect(fill = "#F1F3F5", color = NA),
      strip.text = element_text(face = "bold", color = "black"),
      plot.title = element_text(face = "bold", size = 10)
    )
}

scale_fill_analysis <- function(levels = names(source_palette), guide = waiver(), ...) {
  scale_fill_manual(
    values = source_palette[names(source_palette) %in% levels],
    limits = levels,
    breaks = levels,
    drop = FALSE,
    guide = guide,
    ...
  )
}

scale_color_analysis <- function(levels = names(source_palette), guide = "none", ...) {
  scale_color_manual(
    values = source_palette[names(source_palette) %in% levels],
    limits = levels,
    breaks = levels,
    drop = FALSE,
    guide = guide,
    ...
  )
}

source_legend_guide <- function() {
  guide_legend(nrow = 1, byrow = TRUE, override.aes = list(alpha = 0.85, linewidth = 0.8))
}

integer_when_possible_labels <- function(values) {
  if (all(is.na(values) | abs(values - round(values)) < 1e-8)) {
    return(as.character(as.integer(round(values))))
  }
  scales::label_number(accuracy = 0.01)(values)
}

bootstrap_mean_interval <- function(values, bootstrap_reps = 1000) {
  values <- as.numeric(values)
  estimate <- mean(values)
  if (length(values) == 1) return(tibble(lci = estimate, uci = estimate))
  bootstrap_estimates <- replicate(bootstrap_reps, mean(sample(values, length(values), replace = TRUE)))
  tibble(
    lci = quantile(bootstrap_estimates, 0.025, names = FALSE),
    uci = quantile(bootstrap_estimates, 0.975, names = FALSE)
  )
}

summarize_bootstrap_mean_by <- function(
  data,
  group_columns,
  value_column,
  bootstrap_reps = 1000
) {
  set.seed(20260424)
  data %>%
    group_by(across(all_of(group_columns))) %>%
    group_modify(\(.x, .y) {
      values <- as.numeric(.x[[value_column]])
      interval <- bootstrap_mean_interval(values, bootstrap_reps = bootstrap_reps)
      tibble(n = length(values), estimate = mean(values), lci = interval$lci, uci = interval$uci)
    }) %>%
    ungroup()
}

add_bootstrap_share_intervals <- function(
  count_data,
  group_columns,
  count_column = "n",
  bootstrap_reps = 1000
) {
  set.seed(20260424)
  count_data %>%
    group_by(across(all_of(group_columns))) %>%
    group_modify(\(.x, .y) {
      counts <- as.integer(.x[[count_column]])
      total_n <- sum(counts)
      if (total_n == 0) return(.x %>% mutate(lci = 0, uci = 0))
      probabilities <- counts / total_n
      bootstrap_counts <- rmultinom(bootstrap_reps, total_n, probabilities)
      bootstrap_props <- bootstrap_counts / total_n
      .x %>%
        mutate(
          lci = apply(bootstrap_props, 1, quantile, probs = 0.025, names = FALSE),
          uci = apply(bootstrap_props, 1, quantile, probs = 0.975, names = FALSE)
        )
    }) %>%
    ungroup()
}

inverse_label_fn <- function(transform_name) {
  if (transform_name == "log10") {
    return(function(x) format(signif(10^x, 3), big.mark = "", trim = TRUE, scientific = FALSE))
  }
  if (transform_name == "log1p") {
    return(function(x) format(round(expm1(x)), big.mark = "", trim = TRUE, scientific = FALSE))
  }
  integer_when_possible_labels
}

inverse_break_fn <- function(transform_name) {
  if (transform_name == "log10") {
    return(function(lim) {
      raw <- 10^lim
      brks <- scales::breaks_log(n = 5)(raw)
      log10(brks[brks > 0])
    })
  }
  if (transform_name == "log1p") {
    return(function(lim) {
      raw <- expm1(lim)
      brks <- scales::breaks_extended(n = 5)(raw)
      log1p(brks[brks >= 0])
    })
  }
  waiver()
}

apply_axis_labels <- function(plot_object, transform_name) {
  if (transform_name == "identity") return(plot_object)
  plot_object +
    scale_x_continuous(
      breaks = inverse_break_fn(transform_name),
      labels = inverse_label_fn(transform_name),
      expand = expansion(mult = c(0.02, 0.05))
    )
}

build_outcome_cell <- function(data, spec, source_present, prompt_methods_present) {
  if (!(spec$name %in% names(data))) return(NULL)

  cell <- data %>%
    select(prompt_method, source, all_of(spec$name)) %>%
    rename(value = !!spec$name) %>%
    filter(!is.na(value)) %>%
    mutate(
      prompt_method = factor(as.character(prompt_method), levels = prompt_methods_present),
      source = factor(as.character(source), levels = source_present)
    )

  if (nrow(cell) == 0) return(NULL)
  cell
}

apply_panel_layout <- function(
  plot_object,
  facet_prompt_method,
  show_strip,
  show_y_axis,
  show_legend,
  source_present
) {
  plot_object <- plot_object +
    theme_analysis() +
    theme(
      legend.position = if (show_legend) "bottom" else "none",
      panel.spacing.x = unit(0.05, "lines"),
      panel.spacing.y = unit(0.12, "lines"),
      strip.placement = "outside",
      plot.margin = margin(0.5, 0.25, 0.5, 0.25),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.margin = margin(0, 0, 0, 0),
      axis.text.x = element_text(angle = 28, hjust = 1),
      axis.text.y = if (show_y_axis) element_text(size = 8) else element_blank(),
      axis.title.y = if (show_y_axis) element_text() else element_blank(),
      axis.ticks.y = if (show_y_axis) element_line() else element_blank(),
      axis.line.y.left = element_line(color = "black", linewidth = 0.4)
    )

  if (!facet_prompt_method) return(plot_object)

  # Split outcome panels have reference-only and generator-only facet rows.
  # Free y-space avoids reserving empty source slots inside each row.
  plot_object <- plot_object +
    facet_grid(
      rows = vars(prompt_method),
      switch = "y",
      drop = TRUE,
      scales = "free_y",
      space = "free_y"
    )

  if (show_strip) {
    plot_object + theme(strip.text.y.left = element_text(angle = 0, face = "bold", hjust = 1, size = 9))
  } else {
    plot_object + theme(
      strip.text.y.left = element_blank(),
      strip.background.y = element_blank()
    )
  }
}

build_binary_outcome_panel <- function(
  cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend
) {
  cell_summary <- summarize_bootstrap_mean_by(
    cell,
    group_columns = c("prompt_method", "source"),
    value_column = "value"
  )

  plot_object <- ggplot(cell_summary) +
    geom_col(aes(x = source, y = estimate, fill = source), width = 0.7, alpha = 0.85) +
    geom_errorbar(aes(x = source, ymin = lci, ymax = uci), width = 0.2, linewidth = 0.4) +
    scale_y_continuous(
      labels = scales::label_percent(accuracy = 1),
      limits = c(0, NA),
      expand = expansion(mult = c(0, 0.1))
    ) +
    labs(x = NULL, y = if (show_y_axis) spec$label else NULL) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    scale_fill_analysis(levels = source_present, guide = "none")

  apply_panel_layout(plot_object, facet_prompt_method, show_strip, show_y_axis, show_legend, source_present)
}

build_histogram_outcome_panel <- function(
  cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend
) {
  binned_data <- cell %>%
    mutate(plot_value = as.integer(round(value)))

  int_breaks <- seq(min(binned_data$plot_value), max(binned_data$plot_value))
  ridge_height <- 0.82

  # Instrument count is integer-valued; precompute bin shares so ridgelines
  # handle vertical spacing without smoothing the distribution.
  plot_data <- binned_data %>%
    count(prompt_method, source, plot_value) %>%
    complete(prompt_method, source, plot_value = int_breaks, fill = list(n = 0L)) %>%
    group_by(prompt_method, source) %>%
    mutate(total_n = sum(n), share = n / total_n) %>%
    filter(total_n > 0) %>%
    ungroup() %>%
    arrange(prompt_method, source, plot_value) %>%
    mutate(source_index = as.numeric(source))

  max_share <- max(plot_data$share, na.rm = TRUE)
  plot_data <- plot_data %>%
    mutate(height = ridge_height * share / max_share)

  # Keep the y-range identical to the continuous ridgelines even when the
  # instrument distribution does not contain the global tallest bin.
  ridge_extent_data <- plot_data %>%
    distinct(prompt_method, source, source_index) %>%
    mutate(
      plot_value = min(int_breaks),
      y_extent = source_index + ridge_height
    )

  plot_object <- ggplot(
    plot_data,
    aes(
      x = plot_value,
      y = source_index,
      height = height,
      group = interaction(prompt_method, source),
      fill = source,
      color = source
    )
  ) +
    geom_blank(
      data = ridge_extent_data,
      inherit.aes = FALSE,
      aes(x = plot_value, y = y_extent)
    ) +
    geom_ridgeline(stat = "identity", alpha = 0.4, linewidth = 0.45, min_height = 0, scale = 1) +
    scale_y_continuous(
      breaks = seq_along(source_present),
      labels = source_present,
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    scale_x_continuous(
      breaks = int_breaks,
      labels = as.character(int_breaks),
      expand = expansion(mult = c(0.02, 0.05))
    ) +
    labs(x = spec$label, y = if (show_y_axis) "Source" else NULL) +
    scale_fill_analysis(
      levels = source_present,
      guide = if (show_legend) source_legend_guide() else "none"
    ) +
    scale_color_analysis(levels = source_present, guide = "none")

  apply_panel_layout(plot_object, facet_prompt_method, show_strip, show_y_axis, show_legend, source_present)
}

build_density_outcome_panel <- function(
  cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend
) {
  transformed_cell <- cell %>%
    mutate(plot_value = apply_transform(value, spec$transform)) %>%
    filter(!is.na(plot_value))

  if (nrow(transformed_cell) == 0) return(NULL)

  x_range <- quantile(transformed_cell$plot_value, probs = c(0.005, 0.995), na.rm = TRUE)

  if (outcome_distribution_density_geom == "density") {
    plot_object <- ggplot(transformed_cell, aes(x = plot_value, fill = source, color = source)) +
      geom_density(alpha = 0.22, linewidth = 0.6, adjust = 1) +
      coord_cartesian(xlim = x_range) +
      labs(x = spec$label, y = if (show_y_axis) "Density" else NULL) +
      scale_fill_analysis(
        levels = source_present,
        guide = if (show_legend) source_legend_guide() else "none"
      ) +
      scale_color_analysis(levels = source_present, guide = "none")
  } else {
    density_data <- transformed_cell %>%
      group_by(prompt_method, source) %>%
      group_modify(\(.x, .y) {
        density_values <- density(
          .x$plot_value,
          from = x_range[1],
          to = x_range[2],
          n = 256
        )
        tibble(plot_value = density_values$x, density = density_values$y)
      }) %>%
      ungroup() %>%
      group_by(prompt_method, source) %>%
      mutate(
        height = 0.82 * density / max(density, na.rm = TRUE),
        source_index = as.numeric(source)
      ) %>%
      ungroup()

    plot_object <- ggplot(
      density_data,
      aes(x = plot_value, y = source_index, height = height, fill = source, color = source)
    ) +
      geom_ridgeline(alpha = 0.4, linewidth = 0.45, min_height = 0, scale = 1) +
      coord_cartesian(xlim = x_range) +
      scale_y_continuous(
        breaks = seq_along(source_present),
        labels = source_present,
        expand = expansion(mult = c(0.02, 0.12))
      ) +
      labs(x = spec$label, y = if (show_y_axis) "Source" else NULL) +
      scale_fill_analysis(
        levels = source_present,
        guide = if (show_legend) source_legend_guide() else "none"
      ) +
      scale_color_analysis(levels = source_present, guide = "none")
  }

  plot_object <- apply_axis_labels(plot_object, spec$transform)
  apply_panel_layout(plot_object, facet_prompt_method, show_strip, show_y_axis, show_legend, source_present)
}

build_outcome_panel <- function(
  data,
  spec,
  source_present,
  prompt_methods_present,
  show_strip = FALSE,
  facet_prompt_method = TRUE,
  show_y_axis = TRUE,
  show_legend = FALSE
) {
  cell <- build_outcome_cell(data, spec, source_present, prompt_methods_present)
  if (is.null(cell)) return(NULL)

  if (spec$family == "binary") {
    return(build_binary_outcome_panel(cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend))
  }
  if (spec$plot_kind == "histogram") {
    return(build_histogram_outcome_panel(cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend))
  }
  build_density_outcome_panel(cell, spec, source_present, facet_prompt_method, show_strip, show_y_axis, show_legend)
}

build_outcome_distribution_plot <- function(
  data,
  outcomes = outcome_specs,
  facet_prompt_method = TRUE
) {
  outcomes_joint <- outcomes %>%
    filter(mega_include)
  source_present <- levels(droplevels(data$source))
  prompt_methods_present <- levels(droplevels(data$prompt_method))

  panels <- list()
  for (i in seq_len(nrow(outcomes_joint))) {
    spec <- outcomes_joint[i, ]
    panel <- build_outcome_panel(
      data = data,
      spec = spec,
      source_present = source_present,
      prompt_methods_present = prompt_methods_present,
      show_strip = (length(panels) == 0),
      facet_prompt_method = facet_prompt_method,
      show_y_axis = (length(panels) == 0),
      show_legend = (length(panels) == 0)
    )
    if (is.null(panel)) next
    panels[[length(panels) + 1]] <- panel
  }

  if (length(panels) == 0) return(ggplot() + theme_analysis())

  # These panels mix continuous and pre-binned ridgelines. Keeping local axes
  # avoids patchwork partially collecting y spines across plot types.
  patchwork::wrap_plots(panels, nrow = 1, guides = "collect", widths = rep(1, length(panels))) &
    theme(
      legend.position = "none",
      plot.margin = margin(0.5, 0.5, 0.5, 0.5),
      legend.box = "horizontal",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.margin = margin(0, 0, 0, 0)
    )
}

build_support_width_plot <- function(
  support_summary,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- support_summary %>%
    filter(family == "continuous") %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    filter(outcome %in% support_plot_outcome_levels) %>%
    filter(!is.na(iqr_ratio)) %>%
    mutate(outcome = factor(outcome, levels = support_plot_outcome_levels), width_label = sprintf("%.2fx", iqr_ratio))

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = outcome, y = prompt_method, fill = iqr_ratio)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = width_label), size = 3) +
      facet_grid(cols = vars(source))
  } else {
    p <- ggplot(plot_data, aes(x = outcome, y = source, fill = iqr_ratio)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = width_label), size = 3)
  }

  p +
    scale_fill_gradient2(
      low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
      midpoint = 1, limits = c(0, 1.6), oob = scales::squish
    ) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.25, "lines"))
}

build_reference_interval_plot <- function(
  support_summary,
  outcome_name,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- support_summary %>%
    filter(outcome_name == !!outcome_name) %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name))

  reference_row <- support_summary %>%
    filter(
      outcome_name == !!outcome_name,
      source == reference_source_name,
      prompt_method == reference_prompt_method_name
    )

  if (nrow(plot_data) == 0 || nrow(reference_row) == 0) return(NULL)

  source_levels_present <- levels(droplevels(support_summary$source))
  prompt_method_levels_present <- levels(droplevels(support_summary$prompt_method))

  plot_data <- plot_data %>%
    mutate(
      prompt_method = factor(as.character(prompt_method), levels = prompt_method_levels_present),
      source = factor(as.character(source), levels = source_levels_present)
    )

  x_min <- min(c(plot_data$q10, reference_row$ref_q10), na.rm = TRUE)
  x_max <- max(c(plot_data$q90, reference_row$ref_q90), na.rm = TRUE)
  x_span <- x_max - x_min
  x_min <- x_min - 0.05 * x_span
  x_max <- x_max + 0.1 * x_span

  ref_annotations <- list(
    annotate("rect", xmin = reference_row$ref_q10[1], xmax = reference_row$ref_q90[1], ymin = -Inf, ymax = Inf, fill = "#DDE3EA", alpha = 0.9),
    annotate("rect", xmin = reference_row$ref_q25[1], xmax = reference_row$ref_q75[1], ymin = -Inf, ymax = Inf, fill = "#AEBBCC", alpha = 0.95),
    geom_vline(xintercept = c(reference_row$ref_q10[1], reference_row$ref_q90[1]), color = "#6B7280", linewidth = 0.5, linetype = "22"),
    geom_vline(xintercept = c(reference_row$ref_q25[1], reference_row$ref_q75[1]), color = "#4B5563", linewidth = 0.6)
  )

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(y = prompt_method, color = source)) +
      ref_annotations +
      geom_segment(aes(x = q10, xend = q90, yend = prompt_method), linewidth = 0.9) +
      geom_segment(aes(x = q25, xend = q75, yend = prompt_method), linewidth = 3) +
      geom_point(aes(x = median), size = 2.2) +
      facet_grid(cols = vars(source))
  } else {
    p <- ggplot(plot_data, aes(y = source, color = source)) +
      ref_annotations +
      geom_segment(aes(x = q10, xend = q90, yend = source), linewidth = 0.9) +
      geom_segment(aes(x = q25, xend = q75, yend = source), linewidth = 3) +
      geom_point(aes(x = median), size = 2.2)
  }

  p +
    labs(x = plot_data$outcome[1], y = NULL) +
    coord_cartesian(xlim = c(x_min, x_max), clip = "off") +
    scale_x_continuous(expand = expansion(mult = c(0.05, 0.1))) +
    scale_color_analysis(levels = levels(droplevels(plot_data$source)), guide = "none") +
    theme_analysis() +
    theme(panel.spacing = unit(0.2, "lines"), axis.text.x = element_text(angle = 30, hjust = 1))
}

build_reference_interval_suite <- function(
  support_summary,
  outcome_names,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  panels <- map(outcome_names, function(outcome_name) {
    build_reference_interval_plot(
      support_summary,
      outcome_name,
      facet_prompt_method = facet_prompt_method,
      reference_source_name = reference_source_name,
      reference_prompt_method_name = reference_prompt_method_name
    )
  })
  panels <- panels[!map_lgl(panels, is.null)]
  if (length(panels) == 0) return(NULL)
  patchwork::wrap_plots(panels, nrow = 1) + plot_layout(axes = "collect")
}

build_tail_support_plot <- function(
  tail_summary,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- tail_summary %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    filter(outcome %in% support_plot_outcome_levels) %>%
    transmute(
      prompt_method, source, outcome, tail_probability,
      lower_ratio = lower_tail_ratio, upper_ratio = upper_tail_ratio,
      lower_share = lower_tail_share, upper_share = upper_tail_share
    ) %>%
    pivot_longer(
      cols = c(lower_ratio, upper_ratio, lower_share, upper_share),
      names_to = c("tail_side", ".value"),
      names_pattern = "(lower|upper)_(ratio|share)"
    ) %>%
    mutate(
      tail_side = factor(tail_side, levels = c("lower", "upper"), labels = c("Lower tail", "Upper tail")),
      outcome = factor(outcome, levels = support_plot_outcome_levels),
      threshold_label = sprintf("q%02d", round(tail_probability * 100)),
      ratio_label = sprintf("%.2fx", ratio)
    )

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = outcome, y = tail_side, fill = ratio)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = ratio_label), size = 3) +
      facet_grid(prompt_method + threshold_label ~ source)
  } else {
    p <- ggplot(plot_data, aes(x = outcome, y = tail_side, fill = ratio)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = ratio_label), size = 3) +
      facet_grid(threshold_label ~ source)
  }

  p +
    scale_fill_gradient2(low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 1, limits = c(0, 2), oob = scales::squish) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.25, "lines"))
}

build_human_region_coverage_plot <- function(
  region_summary,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- region_summary %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    filter(outcome %in% support_plot_outcome_levels) %>%
    mutate(
      outcome = factor(outcome, levels = support_plot_outcome_levels),
      region_label = recode(
        region_name,
        "q05_q95" = "Inside q05-q95",
        "q10_q90" = "Inside q10-q90",
        "q25_q75" = "Inside IQR"
      ),
      share_label = sprintf("%.2f", inside_share)
    )

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = outcome, y = region_label, fill = inside_share)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = share_label), size = 3) +
      facet_grid(prompt_method ~ source)
  } else {
    p <- ggplot(plot_data, aes(x = outcome, y = region_label, fill = inside_share)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = share_label), size = 3) +
      facet_grid(. ~ source)
  }

  p +
    scale_fill_gradient(low = "#F7FBFF", high = "#08519C", limits = c(0, 1)) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.25, "lines"))
}

build_within_feature_model_heatmap <- function(similarity_summary) {
  diagonal_rows <- tibble(
    feature = rep(feature_similarity_outcome_levels, each = length(progression_source_levels)),
    source_1 = rep(progression_source_levels, times = length(feature_similarity_outcome_levels)),
    source_2 = rep(progression_source_levels, times = length(feature_similarity_outcome_levels)),
    rho = 1
  )

  mirrored_rows <- similarity_summary %>%
    mutate(
      mirrored_source_1 = as.character(source_2),
      mirrored_source_2 = as.character(source_1)
    ) %>%
    transmute(feature = as.character(feature), source_1 = mirrored_source_1, source_2 = mirrored_source_2, rho = rho)

  plot_data <- bind_rows(
    similarity_summary %>% transmute(feature = as.character(feature), source_1 = as.character(source_1), source_2 = as.character(source_2), rho = rho),
    mirrored_rows,
    diagonal_rows
  ) %>%
    distinct(feature, source_1, source_2, .keep_all = TRUE) %>%
    mutate(
      feature = factor(feature, levels = feature_similarity_outcome_levels),
      source_1 = factor(source_1, levels = progression_source_levels, labels = unname(model_year_labels[progression_source_levels])),
      source_2 = factor(source_2, levels = progression_source_levels, labels = unname(model_year_labels[progression_source_levels])),
      rho_label = sprintf("%.2f", rho)
    )

  ggplot(plot_data, aes(x = source_2, y = source_1, fill = rho)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = rho_label), size = 3.2) +
    scale_fill_gradient2(low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0, limits = c(-1, 1)) +
    labs(x = NULL, y = NULL) +
    facet_wrap(vars(feature), nrow = 1) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.25, "lines"))
}

build_within_model_feature_heatmap <- function(feature_correlation_summary) {
  diagonal_rows <- tibble(
    source = rep(progression_source_levels, each = length(feature_similarity_outcome_levels)),
    feature_label_1 = rep(feature_similarity_outcome_levels, times = length(progression_source_levels)),
    feature_label_2 = rep(feature_similarity_outcome_levels, times = length(progression_source_levels)),
    rho = 1
  )

  mirrored_rows <- feature_correlation_summary %>%
    mutate(
      mirrored_feature_label_1 = as.character(feature_label_2),
      mirrored_feature_label_2 = as.character(feature_label_1)
    ) %>%
    transmute(source = as.character(source), feature_label_1 = mirrored_feature_label_1, feature_label_2 = mirrored_feature_label_2, rho = rho)

  plot_data <- bind_rows(
    feature_correlation_summary %>% transmute(source = as.character(source), feature_label_1 = as.character(feature_label_1), feature_label_2 = as.character(feature_label_2), rho = rho),
    mirrored_rows,
    diagonal_rows
  ) %>%
    distinct(source, feature_label_1, feature_label_2, .keep_all = TRUE) %>%
    mutate(
      source = factor(source, levels = progression_source_levels, labels = unname(model_year_labels[progression_source_levels])),
      feature_label_1 = factor(feature_label_1, levels = rev(feature_similarity_outcome_levels)),
      feature_label_2 = factor(feature_label_2, levels = feature_similarity_outcome_levels),
      rho_label = sprintf("%.2f", rho)
    )

  ggplot(plot_data, aes(x = feature_label_2, y = feature_label_1, fill = rho)) +
    geom_tile(color = "white", linewidth = 0.35) +
    geom_text(aes(label = rho_label), size = 2.6) +
    scale_fill_gradient2(low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0, limits = c(-1, 1)) +
    facet_wrap(vars(source), nrow = 1) +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.35, "lines"))
}

build_neighbor_overlap_plot <- function(
  neighbor_summary,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- neighbor_summary %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    transmute(
      prompt_method, source,
      `Same-source neighbors` = same_source_share,
      `Reference neighbors` = reference_share,
      `Effective source count` = effective_neighbor_source_count
    ) %>%
    pivot_longer(
      cols = c(`Same-source neighbors`, `Reference neighbors`, `Effective source count`),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(label = sprintf("%.2f", value))

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = metric, y = source, fill = value)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = label), size = 3) +
      facet_wrap(vars(prompt_method), ncol = 1)
  } else {
    p <- ggplot(plot_data, aes(x = metric, y = source, fill = value)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = label), size = 3)
  }

  p +
    scale_fill_gradient(low = "#F7FBFF", high = "#084594") +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none", panel.spacing = unit(0.25, "lines"))
}

build_dominant_modes_plot <- function(
  dominant_modes,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  plot_data <- dominant_modes %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    transmute(
      prompt_method, source, family_name,
      `Top-1` = top_1_share,
      `Top-3` = top_3_share,
      `Eff. count` = effective_category_count
    ) %>%
    pivot_longer(cols = c(`Top-1`, `Top-3`, `Eff. count`), names_to = "metric", values_to = "value") %>%
    mutate(label = sprintf("%.2f", value))

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = metric, y = family_name, fill = value)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = label), size = 3) +
      facet_grid(prompt_method ~ source)
  } else {
    p <- ggplot(plot_data, aes(x = metric, y = family_name, fill = value)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = label), size = 3) +
      facet_grid(. ~ source)
  }

  p +
    scale_fill_gradient(low = "#F7FBFF", high = "#084594") +
    labs(x = NULL, y = NULL) +
    theme_analysis() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
}

build_categorical_distribution_plot <- function(
  data,
  category_column,
  category_label,
  separator = NA_character_,
  top_n = NULL,
  facet_prompt_method = TRUE,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  category_data <- data %>%
    select(prompt_method, source, all_of(category_column)) %>%
    rename(category = !!category_column) %>%
    filter(!is.na(category)) %>%
    mutate(category = as.character(category)) %>%
    filter(category != "")

  if (!is.na(separator)) {
    category_data <- category_data %>%
      separate_rows(category, sep = separator) %>%
      filter(category != "")
  }

  if (!is.null(top_n)) {
    top_categories <- category_data %>%
      count(category, sort = TRUE) %>%
      slice_head(n = top_n) %>%
      pull(category)

    category_data <- category_data %>%
      mutate(category = if_else(category %in% top_categories, category, "Other"))
  }

  category_levels <- category_data %>%
    count(category, sort = TRUE) %>%
    pull(category)

  plot_cells <- category_data %>%
    distinct(prompt_method, source) %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name))

  plot_data <- category_data %>%
    semi_join(plot_cells, by = c("prompt_method", "source")) %>%
    count(prompt_method, source, category) %>%
    complete(prompt_method, source, category = category_levels, fill = list(n = 0L)) %>%
    semi_join(plot_cells, by = c("prompt_method", "source")) %>%
    group_by(prompt_method, source) %>%
    mutate(total_n = sum(n), share = n / total_n) %>%
    filter(total_n > 0) %>%
    ungroup() %>%
    mutate(
      category = factor(category, levels = rev(category_levels))
    )

  reference_distribution <- category_data %>%
    filter(source == reference_source_name, prompt_method == reference_prompt_method_name) %>%
    count(category) %>%
    complete(category = category_levels, fill = list(n = 0L)) %>%
    mutate(reference_total_n = sum(n), reference_share = n / reference_total_n) %>%
    filter(reference_total_n > 0) %>%
    select(category, reference_share)

  reference_plot_data <- plot_cells %>%
    crossing(reference_distribution) %>%
    mutate(category = factor(category, levels = rev(category_levels)))

  if (facet_prompt_method) {
    p <- ggplot(plot_data, aes(x = share, y = category, fill = source)) +
      geom_col(width = 0.8, alpha = 0.88) +
      geom_segment(
        data = reference_plot_data,
        aes(x = 0, xend = reference_share, y = category, yend = category),
        inherit.aes = FALSE,
        color = "#1F2937",
        linewidth = 0.35
      ) +
      geom_point(
        data = reference_plot_data,
        aes(x = reference_share, y = category),
        inherit.aes = FALSE,
        color = "#1F2937",
        fill = "white",
        shape = 21,
        stroke = 0.35,
        size = 1.8
      ) +
      facet_grid(prompt_method ~ source)
  } else {
    p <- ggplot(plot_data, aes(x = share, y = category, fill = source)) +
      geom_col(width = 0.8, alpha = 0.88) +
      geom_segment(
        data = reference_plot_data,
        aes(x = 0, xend = reference_share, y = category, yend = category),
        inherit.aes = FALSE,
        color = "#1F2937",
        linewidth = 0.35
      ) +
      geom_point(
        data = reference_plot_data,
        aes(x = reference_share, y = category),
        inherit.aes = FALSE,
        color = "#1F2937",
        fill = "white",
        shape = 21,
        stroke = 0.35,
        size = 1.8
      ) +
      facet_grid(. ~ source)
  }

  p +
    scale_x_continuous(
      labels = scales::label_percent(accuracy = 1),
      limits = c(0, NA),
      breaks = c(0.25, 0.5, 1),
      expand = expansion(mult = c(0, 0.04))
    ) +
    scale_fill_analysis(levels = levels(droplevels(plot_data$source)), guide = "none") +
    labs(x = "Share", y = category_label) +
    theme_analysis() +
    theme(
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(angle = 28, hjust = 1),
      legend.position = "none",
      panel.spacing = unit(0.25, "lines")
    )
}

build_mir_aggregate_table <- function(
  support_summary,
  tail_summary,
  region_summary,
  neighbor_summary,
  dominant_modes_summary,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  tail_agg <- tail_summary %>%
    group_by(prompt_method, source) %>%
    summarize(
      mean_lower_tail_ratio = mean(lower_tail_ratio, na.rm = TRUE),
      mean_upper_tail_ratio = mean(upper_tail_ratio, na.rm = TRUE),
      .groups = "drop"
    )

  region_agg <- region_summary %>%
    filter(region_name == "q10_q90") %>%
    group_by(prompt_method, source) %>%
    summarize(mean_inside_q10_q90 = mean(inside_share, na.rm = TRUE), .groups = "drop")

  support_agg <- support_summary %>%
    filter(family == "continuous") %>%
    group_by(prompt_method, source) %>%
    summarize(
      mean_iqr_ratio = mean(iqr_ratio, na.rm = TRUE),
      mean_overlap_ratio = mean(overlap_ratio, na.rm = TRUE),
      .groups = "drop"
    )

  neighbor_agg <- neighbor_summary %>%
    select(prompt_method, source, same_source_share, reference_share, effective_neighbor_source_count)

  modes_agg <- dominant_modes_summary %>%
    group_by(prompt_method, source) %>%
    summarize(
      mean_top1_share = mean(top_1_share, na.rm = TRUE),
      mean_effective_category_count = mean(effective_category_count, na.rm = TRUE),
      .groups = "drop"
    )

  list(support_agg, tail_agg, region_agg, neighbor_agg, modes_agg) %>%
    purrr::reduce(full_join, by = c("prompt_method", "source")) %>%
    filter(!(source == reference_source_name & prompt_method == reference_prompt_method_name)) %>%
    arrange(source, prompt_method)
}
