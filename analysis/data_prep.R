library(tidyverse)

build_feature_space_table <- function(
  data,
  feature_names = feature_space_feature_names,
  distance_names = distance_feature_names
) {
  feature_table <- data %>%
    mutate(row_id = row_number()) %>%
    select(
      row_id,
      join_key,
      prompt_method,
      source,
      all_of(feature_names),
      canonical_song_key,
      time_signature
    )

  for (feature_name in feature_names) {
    feature_values <- feature_table[[feature_name]]
    feature_mean <- mean(feature_values, na.rm = TRUE)
    feature_sd <- sd(feature_values, na.rm = TRUE)
    scaled_name <- paste0("fs_", feature_name)
    if (is.na(feature_sd) || feature_sd == 0) {
      feature_table[[scaled_name]] <- if_else(is.na(feature_values), NA_real_, 0)
    } else {
      feature_table[[scaled_name]] <- (feature_values - feature_mean) / feature_sd
    }
  }

  feature_table %>%
    mutate(
      distance_complete = if_all(
        all_of(paste0("fs_", distance_names)),
        ~ !is.na(.x)
      )
    )
}

cap_feature_space_rows <- function(
  feature_table,
  max_rows_per_cell = 2500
) {
  if (is.null(max_rows_per_cell)) {
    return(feature_table %>% filter(distance_complete))
  }

  feature_table %>%
    filter(distance_complete) %>%
    arrange(row_id) %>%
    group_by(prompt_method, source) %>%
    slice_head(n = max_rows_per_cell) %>%
    ungroup()
}

apply_transform <- function(values, transform_name) {
  if (transform_name == "identity") return(values)
  if (transform_name == "log10") return(if_else(values > 0, log10(values), NA_real_))
  if (transform_name == "log1p") return(if_else(values >= 0, log1p(values), NA_real_))
  stop("Unknown transform: ", transform_name)
}

inverse_transform_fn <- function(transform_name) {
  if (transform_name == "identity") return(function(x) x)
  if (transform_name == "log10") return(function(x) 10^x)
  if (transform_name == "log1p") return(function(x) exp(x) - 1)
  stop("Unknown transform: ", transform_name)
}

make_chroma_long <- function(data) {
  data %>%
    pivot_longer(
      cols = starts_with("chromasum_"),
      names_to = "chroma_bin",
      values_to = "energy"
    ) %>%
    mutate(
      chroma_bin = factor(
        chroma_labels[as.integer(str_remove(chroma_bin, "chromasum_")) + 1],
        levels = chroma_labels,
        ordered = TRUE
      ),
      energy_scaled = energy * 1000
    )
}

summarize_chroma_profile <- function(data) {
  make_chroma_long(data) %>%
    group_by(prompt_method, source, chroma_bin) %>%
    summarize(
      energy_scaled = mean(energy_scaled),
      energy_sd = sd(energy_scaled),
      n = n(),
      energy_se = energy_sd / sqrt(n),
      lci = energy_scaled - 1.96 * energy_se,
      uci = energy_scaled + 1.96 * energy_se,
      .groups = "drop"
    )
}
