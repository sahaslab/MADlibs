library(tidyverse)

ensure_output_dir <- function(output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

save_plot_pdf <- function(
  plot_object,
  output_path,
  width,
  height
) {
  ggsave(
    filename = output_path,
    plot = plot_object,
    width = width,
    height = height
  )
}

save_table_outputs <- function(table_object, output_prefix) {
  writeLines(format_markdown_table(table_object), paste0(output_prefix, ".md"))
  writeLines(format_latex_table(table_object), paste0(output_prefix, ".tex"))
}

format_markdown_table <- function(table_object) {
  if (nrow(table_object) == 0) {
    return(c("| Note |", "| --- |", "| Empty table |"))
  }
  formatted_table <- format_table_values(table_object)
  header <- paste0("| ", paste(names(formatted_table), collapse = " | "), " |")
  separator <- paste0("| ", paste(rep("---", ncol(formatted_table)), collapse = " | "), " |")
  rows <- apply(formatted_table, 1, \(row) paste0("| ", paste(row, collapse = " | "), " |"))
  c(header, separator, rows)
}

format_latex_table <- function(table_object) {
  formatted_table <- format_table_values(table_object)
  alignment <- paste(rep("l", ncol(formatted_table)), collapse = "")
  header <- paste(names(formatted_table), collapse = " & ")
  body <- apply(formatted_table, 1, \(row) paste0(paste(row, collapse = " & "), " \\\\"))
  c(
    paste0("\\begin{tabular}{", alignment, "}"),
    "\\hline",
    paste0(header, " \\\\"),
    "\\hline",
    body,
    "\\hline",
    "\\end{tabular}"
  )
}

format_table_values <- function(table_object) {
  formatted_table <- table_object
  for (column_name in names(formatted_table)) {
    if (!is.numeric(formatted_table[[column_name]])) {
      next
    }
    formatted_table[[column_name]] <- format_numeric_table_column(
      formatted_table[[column_name]],
      column_name
    )
  }

  formatted_table %>%
    mutate(across(everything(), as.character)) %>%
    mutate(across(everything(), escape_table_value))
}

format_numeric_table_column <- function(values, column_name) {
  integer_columns <- c(
    "n",
    "rows",
    "df",
    "prompt_count",
    "audio_count",
    "shared_prompts",
    "prompt_methods",
    "rare_categories_total",
    "rare_categories_touched",
    "rare_categories_unreached",
    "total_n"
  )
  integer_like <- all(is.na(values) | values == round(values))

  if (column_name %in% integer_columns || integer_like) {
    return(format(round(values), big.mark = "", trim = TRUE, scientific = FALSE))
  }

  format(round(values, 4), nsmall = 4, trim = TRUE)
}

escape_table_value <- function(value) {
  value %>%
    str_replace_all("\\|", "\\\\|") %>%
    str_replace_all("_", "\\\\_") %>%
    replace_na("")
}
