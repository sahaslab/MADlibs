library(tidyverse)
library(fixest)
library(emmeans)

significance_stars <- function(p.value) {
  case_when(
    is.na(p.value) ~ "",
    p.value < 0.0001 ~ "****",
    p.value < 0.001 ~ "***",
    p.value < 0.01 ~ "**",
    p.value < 0.05 ~ "*",
    TRUE ~ ""
  )
}

fit_outcome_frequentist <- function(
  data,
  outcome_col,
  family,
  transform_name = "identity",
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  inverse_fn <- inverse_transform_fn(transform_name)
  transformed_col <- ".outcome_value"

  working <- data %>%
    mutate(
      !!transformed_col := if (family == "binary") {
        as.numeric(.data[[outcome_col]])
      } else {
        apply_transform(.data[[outcome_col]], transform_name)
      }
    ) %>%
    filter(!is.na(.data[[transformed_col]])) %>%
    mutate(
      source = droplevels(source),
      prompt_method = droplevels(prompt_method),
      cell = factor(paste(as.character(source), as.character(prompt_method), sep = " | "))
    )

  if (nrow(working) == 0 || nlevels(working$cell) < 2) {
    return(empty_outcome_fit(levels(droplevels(data$source))))
  }

  metric_model <- feols(
    as.formula(paste(transformed_col, "~ cell")),
    data = working,
    vcov = ~cluster_id
  )
  metric_emmeans <- emmeans(metric_model, specs = "cell", data = working, vcov. = vcov(metric_model))

  cell_lookup <- working %>%
    distinct(cell, source, prompt_method)

  summary_table <- summary(metric_emmeans, infer = TRUE) %>%
    as_tibble() %>%
    rename(
      estimate_link = emmean,
      lci_link = lower.CL,
      uci_link = upper.CL
    ) %>%
    mutate(
      estimate = if (family == "binary") estimate_link else inverse_fn(estimate_link),
      lci = if (family == "binary") lci_link else inverse_fn(lci_link),
      uci = if (family == "binary") uci_link else inverse_fn(uci_link)
    ) %>%
    left_join(cell_lookup, by = "cell")

  pairwise_raw <- contrast(metric_emmeans, method = "pairwise") %>%
    summary(infer = TRUE) %>%
    as_tibble()

  pairwise_table <- pairwise_raw %>%
    separate_wider_delim(contrast, delim = " - ", names = c("cell_a", "cell_b")) %>%
    mutate(
      cell_a = str_remove_all(cell_a, "^\\(|\\)$"),
      cell_b = str_remove_all(cell_b, "^\\(|\\)$")
    ) %>%
    separate_wider_delim(cell_a, delim = " | ", names = c("source_1", "prompt_method_1")) %>%
    separate_wider_delim(cell_b, delim = " | ", names = c("source_2", "prompt_method_2")) %>%
    mutate(
      p.value.label = scales::label_pvalue(accuracy = 0.001)(p.value),
      stars = significance_stars(p.value)
    ) %>%
    filter(prompt_method_1 == prompt_method_2) %>%
    select(prompt_method = prompt_method_1, source_1, source_2, estimate, lci = lower.CL, uci = upper.CL, p.value, p.value.label, stars)

  control_table <- build_control_contrasts(
    pairwise_raw,
    cell_lookup,
    reference_source_name,
    reference_prompt_method_name
  )

  list(
    summary = summary_table %>%
      select(any_of(c("source", "prompt_method", "estimate", "lci", "uci", "estimate_link", "lci_link", "uci_link"))),
    pairwise = pairwise_table,
    control = control_table
  )
}

build_control_contrasts <- function(
  pairwise_raw,
  cell_lookup,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  ref_label <- paste(reference_source_name, reference_prompt_method_name, sep = " | ")
  if (!ref_label %in% as.character(cell_lookup$cell)) {
    return(tibble())
  }

  pairwise_raw %>%
    separate_wider_delim(contrast, delim = " - ", names = c("cell_a", "cell_b"), cols_remove = FALSE) %>%
    mutate(
      cell_a = str_remove_all(cell_a, "^\\(|\\)$"),
      cell_b = str_remove_all(cell_b, "^\\(|\\)$"),
      ref_is_a = cell_a == ref_label,
      ref_is_b = cell_b == ref_label
    ) %>%
    filter(ref_is_a | ref_is_b) %>%
    mutate(
      sign = if_else(ref_is_b, 1, -1),
      target_cell = if_else(ref_is_b, cell_a, cell_b),
      estimate = sign * estimate,
      lci_tmp = sign * lower.CL,
      uci_tmp = sign * upper.CL,
      lci = pmin(lci_tmp, uci_tmp),
      uci = pmax(lci_tmp, uci_tmp),
      p_adj = p.adjust(p.value, method = "BH"),
      stars = significance_stars(p_adj)
    ) %>%
    separate_wider_delim(target_cell, delim = " | ", names = c("source", "prompt_method")) %>%
    select(source, prompt_method, estimate, lci, uci, p.value, p_adj, stars)
}

empty_outcome_fit <- function(source_levels_present = model_levels) {
  list(
    summary = tibble(
      source = factor(character(), levels = source_levels_present),
      prompt_method = factor(character()),
      estimate = numeric(),
      lci = numeric(),
      uci = numeric()
    ),
    pairwise = tibble(),
    control = tibble()
  )
}

fit_outcomes <- function(
  data,
  outcomes = outcome_specs,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  fits <- map(
    seq_len(nrow(outcomes)),
    function(i) {
      spec <- outcomes[i, ]
      if (!(spec$name %in% names(data))) return(empty_outcome_fit(levels(droplevels(data$source))))
      fit_outcome_frequentist(
        data = data,
        outcome_col = spec$name,
        family = spec$family,
        transform_name = spec$transform,
        reference_source_name = reference_source_name,
        reference_prompt_method_name = reference_prompt_method_name
      )
    }
  )

  list(
    summary = bind_rows(map2(
      fits, seq_len(nrow(outcomes)),
      ~ .x$summary %>% mutate(outcome = outcomes$label[.y], outcome_name = outcomes$name[.y])
    )),
    pairwise = bind_rows(map2(
      fits, seq_len(nrow(outcomes)),
      ~ .x$pairwise %>% mutate(outcome = outcomes$label[.y], outcome_name = outcomes$name[.y])
    )),
    control = bind_rows(map2(
      fits, seq_len(nrow(outcomes)),
      ~ {
        tbl <- .x$control
        if (is.null(tbl) || nrow(tbl) == 0) return(tibble())
        tbl %>% mutate(outcome = outcomes$label[.y], outcome_name = outcomes$name[.y])
      }
    ))
  )
}

compute_categorical_contrasts <- function(
  data,
  var_name,
  separator = NA_character_,
  top_n = NULL,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method,
  source_levels_present = levels(droplevels(data$source))
) {
  ref_source <- reference_source_name
  ref_pm <- reference_prompt_method_name

  if (!is.na(separator)) {
    items <- data %>%
      select(prompt_method, source, all_of(var_name)) %>%
      filter(!is.na(.data[[var_name]])) %>%
      separate_rows(all_of(var_name), sep = separator) %>%
      filter(.data[[var_name]] != "")
  } else {
    items <- data %>%
      select(prompt_method, source, all_of(var_name)) %>%
      filter(!is.na(.data[[var_name]])) %>%
      mutate(!!var_name := as.character(.data[[var_name]]))
  }

  if (!is.null(top_n)) {
    top_items <- items %>%
      count(.data[[var_name]], sort = TRUE) %>%
      slice_head(n = top_n) %>%
      pull(1)
    items <- items %>% filter(.data[[var_name]] %in% top_items)
  }

  ref_dist <- items %>%
    filter(source == ref_source, prompt_method == ref_pm) %>%
    count(.data[[var_name]]) %>%
    rename(category = !!var_name) %>%
    mutate(prop = n / sum(n))

  if (nrow(ref_dist) == 0) return(tibble())

  cells <- items %>%
    distinct(prompt_method, source) %>%
    filter(!(source == ref_source & prompt_method == ref_pm))

  result <- pmap_dfr(
    cells,
    function(prompt_method, source) {
      cell_dist <- items %>%
        filter(prompt_method == !!prompt_method, source == !!source) %>%
        count(.data[[var_name]]) %>%
        rename(category = !!var_name)
      merged <- ref_dist %>%
        select(category, expected_prop = prop) %>%
        full_join(cell_dist, by = "category") %>%
        mutate(
          n = if_else(is.na(n), 0L, as.integer(n)),
          expected_prop = if_else(is.na(expected_prop), 0, expected_prop)
        ) %>%
        mutate(expected_prop = pmax(expected_prop, 1e-6)) %>%
        mutate(expected_prop = expected_prop / sum(expected_prop))
      total_n <- sum(merged$n)
      if (total_n == 0) return(tibble())
      test <- suppressWarnings(chisq.test(x = merged$n, p = merged$expected_prop))
      tibble(
        source = source,
        prompt_method = prompt_method,
        n = total_n,
        chi_sq = unname(test$statistic),
        df = unname(test$parameter),
        p.value = test$p.value
      )
    }
  )

  if (nrow(result) == 0) return(result)

  result %>%
    mutate(
      p_adj = p.adjust(p.value, method = "BH"),
      stars = significance_stars(p_adj),
      source = factor(source, levels = source_levels_present),
      prompt_method = factor(prompt_method, levels = levels(data$prompt_method))
    )
}

compute_chroma_contrasts <- function(
  data,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method,
  source_levels_present = levels(droplevels(data$source))
) {
  ref_source <- reference_source_name
  ref_pm <- reference_prompt_method_name

  agg <- make_chroma_long(data) %>%
    filter(!is.na(energy_scaled)) %>%
    group_by(prompt_method, source, chroma_bin) %>%
    summarize(total = sum(energy_scaled), .groups = "drop")

  ref_vec <- agg %>%
    filter(source == ref_source, prompt_method == ref_pm) %>%
    arrange(chroma_bin) %>%
    pull(total)

  if (length(ref_vec) == 0 || sum(ref_vec) == 0) return(tibble())
  ref_prop <- ref_vec / sum(ref_vec)
  ref_prop <- pmax(ref_prop, 1e-6)
  ref_prop <- ref_prop / sum(ref_prop)

  cells <- agg %>%
    distinct(prompt_method, source) %>%
    filter(!(source == ref_source & prompt_method == ref_pm))

  result <- pmap_dfr(
    cells,
    function(prompt_method, source) {
      obs_vec <- agg %>%
        filter(source == !!source, prompt_method == !!prompt_method) %>%
        arrange(chroma_bin) %>%
        pull(total)
      if (length(obs_vec) == 0 || sum(obs_vec) == 0) return(tibble())
      counts <- round(obs_vec / sum(obs_vec) * 1e5)
      test <- suppressWarnings(chisq.test(x = counts, p = ref_prop))
      tibble(
        source = source,
        prompt_method = prompt_method,
        chi_sq = unname(test$statistic),
        df = unname(test$parameter),
        p.value = test$p.value
      )
    }
  )

  if (nrow(result) == 0) return(result)

  result %>%
    mutate(
      p_adj = p.adjust(p.value, method = "BH"),
      stars = significance_stars(p_adj),
      source = factor(source, levels = source_levels_present),
      prompt_method = factor(prompt_method, levels = levels(data$prompt_method))
    )
}

summarize_outcomes <- function(data, outcomes = outcome_specs) {
  map_dfr(
    seq_len(nrow(outcomes)),
    function(i) {
      spec <- outcomes[i, ]
      if (!(spec$name %in% names(data))) return(tibble())
      transformed <- if (spec$family == "binary") {
        as.numeric(data[[spec$name]])
      } else {
        apply_transform(data[[spec$name]], spec$transform)
      }
      working <- data %>%
        mutate(.value = transformed) %>%
        filter(!is.na(.value))

      working %>%
        group_by(prompt_method, source) %>%
        summarize(
          n = n(),
          estimate = mean(.value),
          sd_value = sd(.value),
          se = sd_value / sqrt(n),
          lci = estimate - 1.96 * se,
          uci = estimate + 1.96 * se,
          q10 = quantile(.value, 0.10),
          q25 = quantile(.value, 0.25),
          median = quantile(.value, 0.50),
          q75 = quantile(.value, 0.75),
          q90 = quantile(.value, 0.90),
          .groups = "drop"
        ) %>%
        mutate(
          outcome = spec$label,
          outcome_name = spec$name,
          family = spec$family,
          transform = spec$transform
        )
    }
  )
}

summarize_support_against_reference <- function(
  data,
  outcomes = outcome_specs,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  summaries <- map_dfr(
    seq_len(nrow(outcomes)),
    function(i) {
      spec <- outcomes[i, ]
      if (!(spec$name %in% names(data))) return(tibble())

      raw_value <- if (spec$family == "binary") as.numeric(data[[spec$name]]) else data[[spec$name]]

      data %>%
        mutate(.value = raw_value) %>%
        filter(!is.na(.value)) %>%
        group_by(prompt_method, source) %>%
        summarize(
          n = n(),
          estimate = mean(.value),
          minimum = min(.value),
          q10 = quantile(.value, 0.10),
          q25 = quantile(.value, 0.25),
          median = quantile(.value, 0.50),
          q75 = quantile(.value, 0.75),
          q90 = quantile(.value, 0.90),
          maximum = max(.value),
          .groups = "drop"
        ) %>%
        mutate(
          iqr = q75 - q25,
          p80 = q90 - q10,
          outcome = spec$label,
          outcome_name = spec$name,
          family = spec$family,
          transform = spec$transform
        )
    }
  )

  reference_rows <- summaries %>%
    filter(source == reference_source_name, prompt_method == reference_prompt_method_name) %>%
    select(
      outcome, outcome_name,
      ref_estimate = estimate, ref_minimum = minimum,
      ref_q10 = q10, ref_q25 = q25, ref_median = median,
      ref_q75 = q75, ref_q90 = q90, ref_maximum = maximum,
      ref_iqr = iqr, ref_p80 = p80
    )

  summaries %>%
    left_join(reference_rows, by = c("outcome", "outcome_name")) %>%
    mutate(
      estimate_diff = estimate - ref_estimate,
      median_diff = median - ref_median,
      iqr_ratio = if_else(ref_iqr > 0, iqr / ref_iqr, NA_real_),
      p80_ratio = if_else(ref_p80 > 0, p80 / ref_p80, NA_real_),
      q10_overlap = pmax(q10, ref_q10),
      q90_overlap = pmin(q90, ref_q90),
      overlap_width = pmax(0, q90_overlap - q10_overlap),
      overlap_ratio = if_else(ref_p80 > 0, overlap_width / ref_p80, NA_real_),
      min_diff = minimum - ref_minimum,
      max_diff = maximum - ref_maximum
    )
}

summarize_tail_support_against_reference <- function(
  data,
  outcomes = outcome_specs,
  tail_probabilities = c(0.05),
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  stopifnot(all(tail_probabilities > 0), all(tail_probabilities < 0.5))

  map_dfr(
    tail_probabilities,
    function(tail_probability) {
      map_dfr(
        seq_len(nrow(outcomes)),
        function(i) {
          spec <- outcomes[i, ]
          if (!(spec$name %in% names(data)) || spec$family != "continuous") return(tibble())

          working <- data %>%
            transmute(prompt_method, source, .value = .data[[spec$name]]) %>%
            filter(!is.na(.value))

          reference_row <- working %>%
            filter(source == reference_source_name, prompt_method == reference_prompt_method_name) %>%
            summarize(
              ref_lower_tail = quantile(.value, tail_probability),
              ref_upper_tail = quantile(.value, 1 - tail_probability)
            )

          if (nrow(reference_row) == 0) return(tibble())

          working %>%
            group_by(prompt_method, source) %>%
            summarize(
              n = n(),
              lower_tail_share = mean(.value <= reference_row$ref_lower_tail[1]),
              upper_tail_share = mean(.value >= reference_row$ref_upper_tail[1]),
              .groups = "drop"
            ) %>%
            mutate(
              lower_tail_ratio = lower_tail_share / tail_probability,
              upper_tail_ratio = upper_tail_share / tail_probability,
              tail_probability = tail_probability,
              outcome = spec$label,
              outcome_name = spec$name,
              transform = spec$transform
            )
        }
      )
    }
  )
}

summarize_human_region_coverage <- function(
  data,
  outcomes = outcome_specs,
  region_specs = human_region_specs,
  reference_source_name = reference_source,
  reference_prompt_method_name = reference_prompt_method
) {
  map_dfr(
    seq_len(nrow(outcomes)),
    function(i) {
      spec <- outcomes[i, ]
      if (!(spec$name %in% names(data)) || spec$family != "continuous") return(tibble())

      working <- data %>%
        transmute(prompt_method, source, .value = .data[[spec$name]]) %>%
        filter(!is.na(.value))

      map_dfr(
        seq_len(nrow(region_specs)),
        function(j) {
          region <- region_specs[j, ]

          reference_row <- working %>%
            filter(source == reference_source_name, prompt_method == reference_prompt_method_name) %>%
            summarize(
              ref_lower = quantile(.value, region$lower_prob),
              ref_upper = quantile(.value, region$upper_prob)
            )

          if (nrow(reference_row) == 0) return(tibble())

          working %>%
            group_by(prompt_method, source) %>%
            summarize(
              n = n(),
              inside_share = mean(.value >= reference_row$ref_lower[1] & .value <= reference_row$ref_upper[1]),
              outside_share = 1 - inside_share,
              .groups = "drop"
            ) %>%
            mutate(
              expected_inside_share = region$upper_prob - region$lower_prob,
              inside_ratio = inside_share / expected_inside_share,
              region_name = region$region_name,
              lower_prob = region$lower_prob,
              upper_prob = region$upper_prob,
              outcome = spec$label,
              outcome_name = spec$name
            )
        }
      )
    }
  )
}

make_raw_tempo_bin <- function(values) {
  breaks <- c(seq(0, 240, by = 20), Inf)
  labels <- c(paste(head(seq(0, 220, by = 20)), seq(20, 240, by = 20), sep = "-"), "240+")
  cut(values, breaks = breaks, labels = labels, include.lowest = TRUE, right = FALSE)
}

make_folded_tempo_bin <- function(values) {
  breaks <- c(seq(0, 60, by = 5), Inf)
  labels <- c(paste(head(seq(0, 55, by = 5)), seq(5, 60, by = 5), sep = "-"), "60+")
  cut(values, breaks = breaks, labels = labels, include.lowest = TRUE, right = FALSE)
}

make_instrument_count_bin <- function(values) {
  if_else(is.na(values), "Missing", as.character(as.integer(values)))
}

make_time_signature_category <- function(values) {
  if_else(is.na(values), "Missing", as.character(as.integer(values)))
}

make_key_mode_category <- function(values) {
  if_else(is.na(values), "Missing", values)
}

summarize_category_concentration <- function(category_data, family_name) {
  category_data %>%
    filter(!is.na(category_value), category_value != "") %>%
    count(prompt_method, source, category_value) %>%
    group_by(prompt_method, source) %>%
    mutate(prop = n / sum(n), rank = row_number(desc(prop))) %>%
    summarize(
      total_n = sum(n),
      top_1_share = max(prop),
      top_3_share = sum(prop[rank <= 3]),
      entropy = -sum(prop * log(prop)),
      effective_category_count = exp(entropy),
      top_category = category_value[which.max(prop)],
      top_3_categories = paste(category_value[order(desc(prop))][seq_len(min(3, n()))], collapse = ", "),
      family_name = family_name,
      .groups = "drop"
    )
}

summarize_dominant_modes <- function(
  data,
  source_levels_present = levels(droplevels(data$source))
) {
  bind_rows(
    data %>%
      transmute(prompt_method, source, category_value = as.character(make_raw_tempo_bin(tempo))) %>%
      summarize_category_concentration("Raw Tempo"),
    data %>%
      transmute(prompt_method, source, category_value = as.character(make_folded_tempo_bin(tempo_fold_60))) %>%
      summarize_category_concentration("Folded Tempo"),
    data %>%
      transmute(prompt_method, source, category_value = make_instrument_count_bin(n_instruments)) %>%
      summarize_category_concentration("Instrument Count"),
    data %>%
      transmute(prompt_method, source, category_value = make_time_signature_category(time_signature)) %>%
      summarize_category_concentration("Time Signature"),
    data %>%
      transmute(prompt_method, source, category_value = make_key_mode_category(canonical_song_key)) %>%
      summarize_category_concentration("Key")
  ) %>%
    mutate(
      source = factor(source, levels = source_levels_present),
      prompt_method = factor(prompt_method, levels = levels(data$prompt_method))
    )
}

compute_neighbor_indices_within_clusters <- function(
  feature_matrix,
  neighbor_k,
  cluster_assignments
) {
  neighbor_list <- vector("list", nrow(feature_matrix))

  for (cluster_id in sort(unique(cluster_assignments))) {
    idx <- which(cluster_assignments == cluster_id)
    if (length(idx) < 2) next

    distance_matrix <- as.matrix(dist(feature_matrix[idx, , drop = FALSE]))
    diag(distance_matrix) <- Inf

    for (row_index in seq_along(idx)) {
      row_distances <- distance_matrix[row_index, ]
      row_order <- order(row_distances)
      neighbor_list[[idx[row_index]]] <- idx[row_order[seq_len(min(neighbor_k, length(row_order)))]]
    }
  }

  neighbor_list
}

summarize_neighbor_overlap <- function(
  feature_table,
  neighbor_k = 10,
  reference_source_name = reference_source
) {
  working <- feature_table %>%
    filter(distance_complete) %>%
    mutate(distance_row = row_number())

  if (nrow(working) < 2) return(tibble())

  feature_matrix <- working %>%
    select(all_of(paste0("fs_", distance_feature_names))) %>%
    as.matrix()

  set.seed(20260418)
  prototype_count <- min(max(32, floor(sqrt(nrow(working)))), 256, nrow(working))
  clustering <- kmeans(feature_matrix, centers = prototype_count, iter.max = 50, nstart = 1)
  neighbor_ids <- compute_neighbor_indices_within_clusters(feature_matrix, neighbor_k, clustering$cluster)

  neighbor_rows <- map_dfr(
    seq_len(nrow(working)),
    function(i) {
      idx <- neighbor_ids[[i]]
      if (is.null(idx) || length(idx) == 0) return(tibble())

      neighbor_meta <- working[idx, ]
      source_prop <- neighbor_meta %>% count(source) %>% mutate(prop = n / sum(n))
      source_entropy <- -sum(source_prop$prop * log(source_prop$prop))

      tibble(
        prompt_method = working$prompt_method[i],
        source = working$source[i],
        same_source_share = mean(neighbor_meta$source == working$source[i]),
        same_prompt_method_share = mean(neighbor_meta$prompt_method == working$prompt_method[i]),
        reference_share = mean(neighbor_meta$source == reference_source_name),
        neighbor_source_entropy = source_entropy,
        effective_neighbor_source_count = exp(source_entropy)
      )
    }
  )

  neighbor_rows %>%
    group_by(prompt_method, source) %>%
    summarize(
      rows = n(),
      same_source_share = mean(same_source_share),
      same_prompt_method_share = mean(same_prompt_method_share),
      reference_share = mean(reference_share),
      neighbor_source_entropy = mean(neighbor_source_entropy),
      effective_neighbor_source_count = mean(effective_neighbor_source_count),
      .groups = "drop"
    )
}

compute_prompt_aligned_pair_correlation <- function(pair_data, source_1, source_2) {
  if (nrow(pair_data) < 3) {
    return(tibble(shared_prompts = nrow(pair_data), rho = NA_real_, p.value = NA_real_))
  }
  test <- suppressWarnings(cor.test(pair_data[[source_1]], pair_data[[source_2]], method = "spearman", exact = FALSE))
  tibble(shared_prompts = nrow(pair_data), rho = unname(test$estimate), p.value = test$p.value)
}

summarize_model_feature_similarity_by_prompt_method <- function(
  data,
  feature_names = feature_similarity_feature_names,
  pair_specs = model_pair_specs
) {
  feature_labels <- outcome_specs %>%
    filter(name %in% feature_names) %>%
    select(feature_name = name, feature = label)

  prompt_aligned <- data %>%
    filter(source %in% generator_levels) %>%
    select(prompt_method, prompt_id, source, all_of(feature_names)) %>%
    pivot_longer(cols = all_of(feature_names), names_to = "feature_name", values_to = "value") %>%
    filter(!is.na(value)) %>%
    group_by(prompt_method, prompt_id, source, feature_name) %>%
    summarize(value = mean(value), .groups = "drop")

  pairwise_similarity <- map_dfr(
    seq_len(nrow(pair_specs)),
    function(i) {
      pair_spec <- pair_specs[i, ]

      prompt_aligned %>%
        filter(source %in% c(pair_spec$source_1, pair_spec$source_2)) %>%
        pivot_wider(names_from = source, values_from = value) %>%
        filter(!is.na(.data[[pair_spec$source_1]]), !is.na(.data[[pair_spec$source_2]])) %>%
        group_by(prompt_method, feature_name) %>%
        group_modify(
          ~ compute_prompt_aligned_pair_correlation(.x, pair_spec$source_1, pair_spec$source_2)
        ) %>%
        ungroup() %>%
        mutate(
          source_1 = pair_spec$source_1,
          source_2 = pair_spec$source_2,
          pair_label = pair_spec$pair_label,
          pair_rank = pair_spec$pair_rank,
          later_year = pair_spec$later_year
        )
    }
  )

  pairwise_similarity %>%
    left_join(feature_labels, by = "feature_name") %>%
    mutate(
      feature = factor(feature, levels = feature_similarity_outcome_levels),
      prompt_method = factor(prompt_method, levels = levels(data$prompt_method)),
      pair_label = factor(pair_label, levels = pair_specs$pair_label)
    ) %>%
    select(prompt_method, feature_name, feature, source_1, source_2, pair_label, pair_rank, later_year, shared_prompts, rho, p.value)
}

summarize_model_feature_similarity <- function(prompt_method_similarity) {
  prompt_method_similarity %>%
    mutate(
      fisher_weight = pmax(shared_prompts - 3, 1),
      rho_clipped = pmin(pmax(rho, -0.999999), 0.999999),
      fisher_z = atanh(rho_clipped)
    ) %>%
    filter(!is.na(fisher_z)) %>%
    group_by(feature_name, feature, source_1, source_2, pair_label, pair_rank, later_year) %>%
    summarize(
      prompt_methods = n(),
      shared_prompts = sum(shared_prompts),
      pooled_weight = sum(fisher_weight),
      rho = tanh(weighted.mean(fisher_z, fisher_weight)),
      lci = tanh(weighted.mean(fisher_z, fisher_weight) - 1.96 / sqrt(sum(fisher_weight))),
      uci = tanh(weighted.mean(fisher_z, fisher_weight) + 1.96 / sqrt(sum(fisher_weight))),
      .groups = "drop"
    ) %>%
    mutate(
      pair_label = factor(pair_label, levels = levels(prompt_method_similarity$pair_label)),
      feature = factor(feature, levels = levels(prompt_method_similarity$feature))
    )
}

compute_feature_pair_correlation <- function(pair_data, feature_1, feature_2) {
  if (nrow(pair_data) < 3) {
    return(tibble(shared_prompts = nrow(pair_data), rho = NA_real_, p.value = NA_real_))
  }
  test <- suppressWarnings(cor.test(pair_data[[feature_1]], pair_data[[feature_2]], method = "spearman", exact = FALSE))
  tibble(shared_prompts = nrow(pair_data), rho = unname(test$estimate), p.value = test$p.value)
}

summarize_within_model_feature_correlation_by_prompt_method <- function(
  data,
  feature_names = feature_similarity_feature_names
) {
  feature_labels <- outcome_specs %>%
    filter(name %in% feature_names) %>%
    select(feature_name = name, feature = label)

  feature_pair_specs <- as_tibble(t(combn(feature_names, 2)), .name_repair = "minimal") %>%
    setNames(c("feature_1", "feature_2"))

  prompt_aligned <- data %>%
    filter(source %in% generator_levels) %>%
    select(prompt_method, prompt_id, source, all_of(feature_names)) %>%
    group_by(prompt_method, prompt_id, source) %>%
    summarize(across(all_of(feature_names), \(x) mean(x, na.rm = TRUE)), .groups = "drop")

  pairwise_similarity <- map_dfr(
    seq_len(nrow(feature_pair_specs)),
    function(i) {
      pair_spec <- feature_pair_specs[i, ]

      prompt_aligned %>%
        select(prompt_method, prompt_id, source, all_of(c(pair_spec$feature_1, pair_spec$feature_2))) %>%
        filter(!is.na(.data[[pair_spec$feature_1]]), !is.na(.data[[pair_spec$feature_2]])) %>%
        group_by(source, prompt_method) %>%
        group_modify(
          ~ compute_feature_pair_correlation(.x, pair_spec$feature_1, pair_spec$feature_2)
        ) %>%
        ungroup() %>%
        mutate(feature_1 = pair_spec$feature_1, feature_2 = pair_spec$feature_2)
    }
  )

  pairwise_similarity %>%
    left_join(feature_labels, by = c("feature_1" = "feature_name")) %>%
    rename(feature_label_1 = feature) %>%
    left_join(feature_labels, by = c("feature_2" = "feature_name")) %>%
    rename(feature_label_2 = feature) %>%
    mutate(
      source = factor(as.character(source), levels = progression_source_levels),
      prompt_method = factor(prompt_method, levels = levels(data$prompt_method)),
      feature_label_1 = factor(feature_label_1, levels = feature_similarity_outcome_levels),
      feature_label_2 = factor(feature_label_2, levels = feature_similarity_outcome_levels)
    )
}

summarize_within_model_feature_correlation <- function(prompt_method_similarity) {
  prompt_method_similarity %>%
    mutate(
      fisher_weight = pmax(shared_prompts - 3, 1),
      rho_clipped = pmin(pmax(rho, -0.999999), 0.999999),
      fisher_z = atanh(rho_clipped)
    ) %>%
    filter(!is.na(fisher_z)) %>%
    group_by(source, feature_1, feature_2, feature_label_1, feature_label_2) %>%
    summarize(
      prompt_methods = n(),
      shared_prompts = sum(shared_prompts),
      pooled_weight = sum(fisher_weight),
      rho = tanh(weighted.mean(fisher_z, fisher_weight)),
      lci = tanh(weighted.mean(fisher_z, fisher_weight) - 1.96 / sqrt(sum(fisher_weight))),
      uci = tanh(weighted.mean(fisher_z, fisher_weight) + 1.96 / sqrt(sum(fisher_weight))),
      .groups = "drop"
    ) %>%
    mutate(source = factor(as.character(source), levels = progression_source_levels))
}
