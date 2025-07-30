# Cell Type Proportions Analysis - Complete Pipeline
# ==================================================

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(readr)

# HELPER FUNCTIONS
# ================

#' Load and prepare cell type count data
#' @param file_path Path to the cell type counts CSV file
#' @param exclude_patient Optional patient ID to exclude
#' @return Cleaned data frame with standardized column names
load_cell_type_data <- function(file_path, exclude_patient = NULL) {
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Clean column names and convert aneuploidy to factor
  df <- df %>% 
    mutate(aneuploidy = as.factor(aneuploidy)) %>% 
    rename_with(~ gsub(" ", "_", tolower(.)))
  
  # Exclude specific patient if specified
  if (!is.null(exclude_patient)) {
    df <- df %>% filter(patient_id != exclude_patient)
  }
  
  return(df)
}

#' Calculate cell type proportions per patient
#' @param df Data frame with cell type counts
#' @param exclude_columns Vector of column names to exclude from proportion calculation
#' @return Data frame with proportions
calculate_cell_proportions <- function(df, exclude_columns = c("unclassified", "stroma_sma", "stroma")) {
  # Remove specified columns
  df_clean <- df %>% select(-any_of(exclude_columns))
  
  # Calculate proportions
  props_df <- df_clean %>%
    rowwise() %>%
    mutate(
      row_total = sum(c_across(where(is.numeric) & !matches("id"))),
      across(where(is.numeric) & !matches("id"), 
             ~ .x / row_total, 
             .names = "{.col}_prop")
    ) %>%
    ungroup() %>%
    select(patient_id, aneuploidy, ends_with("prop"))
  
  return(props_df)
}

#' Convert proportions to long format
#' @param props_df Data frame with proportions in wide format
#' @return Data frame in long format
proportions_to_long <- function(props_df) {
  long_props_df <- props_df %>%
    pivot_longer(cols = ends_with("_prop"),
                 names_to = "phenotype",
                 values_to = "proportion") %>%
    mutate(phenotype = str_remove(phenotype, "_prop")) %>%
    filter(phenotype != "row_total")
  
  return(long_props_df)
}

#' Calculate summary statistics for proportions
#' @param long_props_df Data frame in long format
#' @param group_vars Vector of grouping variables (default: c("aneuploidy", "phenotype"))
#' @return Data frame with summary statistics
calculate_proportion_stats <- function(long_props_df, group_vars = c("aneuploidy", "phenotype")) {
  summary_df <- long_props_df %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      mean_proportion = mean(proportion, na.rm = TRUE),
      sd_proportion = sd(proportion, na.rm = TRUE),
      se = sd(proportion, na.rm = TRUE) / sqrt(n()),
      n = n(),
      ci95 = 1.96 * se,
      .groups = "drop"
    )
  
  return(summary_df)
}

#' Calculate p-values for group comparisons
#' @param long_props_df Data frame in long format
#' @param group_var Variable to compare groups (default: "aneuploidy")
#' @param method Statistical method to use ('t.test' or 'betareg')
#' @return Data frame with p-values
calculate_proportion_pvalues <- function(long_props_df, group_var = "aneuploidy", method = c('t.test', 'betareg')) {
  # Validate method
  method <- match.arg(method)
  
  if (method == 't.test') {
    pval_df <- long_props_df %>% 
      group_by(phenotype) %>%
      summarise(
        p_value = t.test(proportion ~ get(group_var))$p.value,
        .groups = "drop"
      )
  } else if (method == 'betareg') {
    # Check if betareg package is installed
    if (!requireNamespace("betareg", quietly = TRUE)) {
      stop("Package 'betareg' needed for this method. Please install it.")
    }
    
    pval_df <- long_props_df %>% 
      group_by(phenotype) %>%
      summarise(
        p_value = {
          # Fit beta regression model
          model <- betareg::betareg(proportion ~ get(group_var), data = cur_data())
          # Extract p-value for group_var coefficient
          summary(model)$coefficients$mean[2,4]
        },
        .groups = "drop"
      )
  }
  
  # Add significance labels and formatting
  pval_df <- pval_df %>%
    mutate(
      p_label = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**", 
        p_value < 0.05 ~ "*",
        TRUE ~ "ns"
      ),
      p_numeric = signif(p_value, 3),
      significance = p_value < 0.05,
      method = method  # Add method used for tracking
    )
  
  return(pval_df)
}

#' Create cell type proportions plot
#' @param summary_df Data frame with summary statistics
#' @param plot_title Title for the plot
#' @param facet_var Optional faceting variable
#' @param error_type Type of error bars ("se" or "ci95")
#' @param angle Text angle for x-axis labels
#' @return ggplot object
create_proportions_plot <- function(summary_df, plot_title, facet_var = NULL, 
                                    error_type = "se", angle = 45) {
  
  # Select error bar values
  error_col <- switch(error_type,
                      "se" = "se",
                      "ci95" = "ci95",
                      "se")
  
  # Base plot
  p <- summary_df %>%
    filter(!str_detect(phenotype, "unclassified|row_total")) %>%
    ggplot(aes(x = phenotype, y = mean_proportion, fill = aneuploidy)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    geom_errorbar(aes(
      ymin = mean_proportion - get(error_col),
      ymax = mean_proportion + get(error_col)
    ),
    width = 0.2,
    position = position_dodge(width = 0.9)
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = angle, hjust = 1),
      legend.position = "bottom"
    ) +
    labs(
      title = plot_title,
      x = "Cell Type",
      y = "Mean Proportion",
      fill = "Aneuploidy"
    )
  
  # Add faceting if specified
  if (!is.null(facet_var)) {
    p <- p + facet_grid(~ get(facet_var))
  }
  
  return(p)
}


# MAIN PIPELINE FUNCTION
# ======================

#' Complete cell type proportions analysis pipeline
#' @param file_path Path to the cell type counts CSV file
#' @param exclude_columns Vector of column names to exclude from proportion calculation
#' @param exclude_patient Optional patient ID to exclude from analysis
#' @param plot_title Title for the generated plot
#' @param group_vars Vector of grouping variables for summary statistics
#' @param facet_var Optional variable for plot faceting
#' @param statistical_methods Vector of statistical methods to use ('t.test', 'betareg', or both)
#' @param plot_angle Angle for x-axis labels in plot
#' @param error_type Type of error bars ("se" or "ci95")
#' @param verbose Whether to print intermediate results
#' @return List containing all analysis results
run_cell_type_analysis <- function(file_path,
                                   exclude_columns = STANDARD_EXCLUSIONS,
                                   exclude_patient = NULL,
                                   plot_title = "Cell Type Proportions by Aneuploidy Status",
                                   group_vars = c("aneuploidy", "phenotype"),
                                   facet_var = NULL,
                                   statistical_methods = c("t.test", "betareg"),
                                   plot_angle = 45,
                                   error_type = "se",
                                   verbose = TRUE) {
  
  if (verbose) cat("Loading and preparing data...\n")
  
  # Load data
  df <- load_cell_type_data(file_path, exclude_patient = exclude_patient)
  
  if (verbose) cat("Calculating proportions...\n")
  
  # Calculate proportions
  props_df <- calculate_cell_proportions(df, exclude_columns = exclude_columns)
  
  if (verbose) cat("Converting to long format...\n")
  
  # Convert to long format
  long_props_df <- proportions_to_long(props_df)
  
  if (verbose) cat("Calculating summary statistics...\n")
  
  # Calculate summary statistics
  summary_df <- calculate_proportion_stats(long_props_df, group_vars = group_vars)
  
  # Calculate p-values for each requested method
  pval_results <- list()
  for (method in statistical_methods) {
    if (verbose) cat(paste("Calculating p-values using", method, "...\n"))
    pval_results[[method]] <- calculate_proportion_pvalues(long_props_df, method = method)
  }
  
  if (verbose) cat("Creating plot...\n")
  
  # Create plot
  plot <- create_proportions_plot(summary_df, 
                                  plot_title = plot_title,
                                  facet_var = facet_var,
                                  error_type = error_type,
                                  angle = plot_angle)
  
  # Compile results
  results <- list(
    raw_counts_data = df,
    proportions_data = long_props_df,
    summary_stats = summary_df,
    p_values = pval_results,
    plot = plot,
    analysis_params = list(
      file_path = file_path,
      exclude_columns = exclude_columns,
      exclude_patient = exclude_patient,
      group_vars = group_vars,
      statistical_methods = statistical_methods
    )
  )
  
  if (verbose) {
    cat("\nAnalysis complete!\n")
    cat("Results include:\n")
    cat("- raw_data: Original loaded data\n")
    cat("- proportions_wide: Proportions in wide format\n")
    cat("- proportions_long: Proportions in long format\n")
    cat("- summary_stats: Summary statistics by group\n")
    cat("- p_values: Statistical test results\n")
    cat("- plot: ggplot object\n")
    cat("- analysis_params: Parameters used in analysis\n")
  }
  
  return(results)
}

# USAGE EXAMPLES
# ==============

# Set working directory
setwd('/Volumes/scratch/DIMA/chiodin/tests/patients_analysis')
# data_file = "data/cell_type_counts_aneuploidy_NEW.csv"
data_file = 'data/phenotype_counts_20250730.csv'


# 1. STANDARD ANALYSIS
# --------------------
STANDARD_EXCLUSIONS <- c("unclassified", "stroma_sma", "stroma")
results_standard <- run_cell_type_analysis(
  file_path = data_file,
  exclude_columns = STANDARD_EXCLUSIONS,
  plot_title = "Cell Type Proportions by Aneuploidy Status",
  plot_angle = 90
)

# 2. UNCLASSIFIED EXCLUSION ANALYSIS
# ------------------------------
# Minimal exclusions (only unclassified)
UNCLASSIFIED_EXCLUSION <- c("unclassified")
results_unclassified <- run_cell_type_analysis(
  file_path = data_file,
  exclude_columns = UNCLASSIFIED_EXCLUSION,
  plot_title = "Cell Type Proportions (Minimal Exclusions)",
  plot_angle = 90
)

# 3. EXCLUDE SPECIFIC PATIENT
# ---------------------------
results_no_patient <- run_cell_type_analysis(
  file_path = data_file,
  exclude_columns = UNCLASSIFIED_EXCLUSION,
  exclude_patient = "1818600",
  plot_title = "Cell Type Proportions (Patient 1818600 Excluded)",
  plot_angle = 90
)

# 4. CLINICAL DATA WITH TREATMENT ARM (requires modified load function)
# Note: This would need a separate pipeline for clinical data
# or modification of the load_cell_type_data function to handle clinical data

# 5. DISPLAY AND ACCESS RESULTS
# -----------------------------

# Display plots
print(results_standard$plot)
print(results_unclassified$plot)
print(results_no_patient$plot)

# Access p-values
print("Standard analysis - T-test p-values:")
print(results_standard$p_values$t.test)

print("Standard analysis - Beta regression p-values:")
print(results_standard$p_values$betareg)

# Access summary statistics
print("Summary statistics:")
print(results_standard$summary_stats)
