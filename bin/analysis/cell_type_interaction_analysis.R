# Cell Type Interactions Analysis - Functions and Setup
# =====================================================

# Load required libraries
library(tidyverse)
library(ggpubr)

# FUNCTIONS
# =========

#' Clean and prepare interaction data
#' @param interactions_file Path to interactions CSV file
#' @param clinical_data Clinical data frame with patient_id, aneuploidy, treatment_arm
#' @return Cleaned interactions data frame
prepare_interaction_data <- function(interactions_file, clinical_data) {
  interactions <- read_csv(interactions_file, show_col_types = FALSE)
  
  interactions_clean <- interactions %>% 
    mutate(patient_id = as.character(patient_id)) %>% 
    select(patient_id, everything()) %>%
    left_join(
      clinical_data %>% select(patient_id, aneuploidy, treatment_arm),
      by = "patient_id"
    ) %>% 
    mutate(interaction = as.factor(interaction))
  
  return(interactions_clean)
}

#' Calculate interaction proportions
#' @param interactions_data Cleaned interactions data frame
#' @return Data frame with interaction proportions
calculate_interaction_proportions <- function(interactions_data) {
  interactions_summary <- interactions_data %>% 
    group_by(interaction, aneuploidy, patient_id) %>% 
    summarize(tot_pair_interactions = sum(interaction_count), .groups = "drop") %>%
    separate(interaction, into = c("start_point", "end_point"), sep = "-", remove = FALSE) %>% 
    group_by(start_point, aneuploidy, patient_id) %>% 
    mutate(
      tot_start_point_interactions = sum(tot_pair_interactions),
      interactions_prop = tot_pair_interactions / tot_start_point_interactions
    ) %>% 
    ungroup()
  
  return(interactions_summary)
}

#' Calculate summary statistics for plotting
#' @param interactions_summary Data frame with interaction proportions
#' @return Data frame with mean and SD for each interaction-aneuploidy combination
calculate_plot_statistics <- function(interactions_summary) {
  interactions_plot_data <- interactions_summary %>% 
    group_by(interaction, aneuploidy) %>% 
    summarize(
      mean_interactions_prop = mean(interactions_prop, na.rm = TRUE),
      sd_interactions_prop = sd(interactions_prop, na.rm = TRUE),
      n_patients = n_distinct(patient_id),
      .groups = "drop"
    )
  
  return(interactions_plot_data)
}

#' Create interaction plot
#' @param plot_data Data frame with plotting statistics
#' @param interactions_filter Vector of interaction names to include
#' @param plot_title Title for the plot
#' @param free_scales Whether to use free y-axis scales (default: TRUE)
#' @return ggplot object
create_interaction_plot <- function(plot_data, interactions_filter, plot_title, free_scales = TRUE) {
  scales_setting <- if(free_scales) "free_y" else "fixed"
  
  plot_data %>% 
    filter(interaction %in% interactions_filter) %>% 
    ggplot(aes(x = aneuploidy, y = mean_interactions_prop, fill = aneuploidy)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    geom_errorbar(aes(ymin = mean_interactions_prop - sd_interactions_prop,
                      ymax = mean_interactions_prop + sd_interactions_prop),
                  width = 0.4, position = position_dodge(width = 0.9)) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1),
      legend.position = "bottom"
    ) +
    labs(
      title = plot_title, 
      y = "Mean Interaction Proportion", 
      x = "Aneuploidy Status",
      fill = "Aneuploidy"
    ) +
    facet_wrap(~interaction, scales = scales_setting)
}


#' Calculate statistical p-values
#' @param interactions_summary Data frame with interaction proportions
#' @param interactions_filter Vector of interaction names to test
#' @param method Statistical test method: 't.test' or 'betareg'
#' @return Data frame with p-values and significance labels
calculate_pvalues <- function(interactions_summary, interactions_filter, method = c('t.test', 'betareg')) {
  method <- match.arg(method)
  
  filtered_data <- interactions_summary %>% 
    filter(interaction %in% interactions_filter)
  
  if (method == 't.test') {
    pval_df <- filtered_data %>%
      group_by(interaction) %>%
      summarise(
        p_value = t.test(interactions_prop ~ aneuploidy)$p.value,
        .groups = "drop"
      )
  } else if (method == 'betareg') {
    if (!requireNamespace("betareg", quietly = TRUE)) {
      stop("Package 'betareg' needed for this method. Please install it.")
    }
    
    pval_df <- filtered_data %>%
      group_by(interaction) %>%
      summarise(
        p_value = {
          model <- betareg::betareg(interactions_prop ~ aneuploidy, data = cur_data())
          summary(model)$coefficients$mean[2, 4]
        },
        .groups = "drop"
      )
  }
  
  pval_df %>%
    mutate(
      p_label = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ "ns"
      ),
      p_numeric = signif(p_value, 3),
      significance = p_value < 0.05
    )
}





#' Generate summary statistics
#' @param interactions_summary Data frame with interaction proportions
#' @return Summary statistics by aneuploidy status
generate_summary_stats <- function(interactions_summary) {
  interactions_summary %>% 
    group_by(aneuploidy) %>% 
    summarize(
      n_patients = n_distinct(patient_id),
      n_interactions = n_distinct(interaction),
      mean_total_interactions = mean(tot_pair_interactions, na.rm = TRUE),
      median_total_interactions = median(tot_pair_interactions, na.rm = TRUE),
      mean_proportion = mean(interactions_prop, na.rm = TRUE),
      .groups = "drop"
    )
}


# PIPELINE FUNCTION (single filter only)
run_interaction_pipeline <- function(interactions_file, interactions_filter, filter_name = "Selected Interactions") {
  # Load dependencies and clinical data
  source('./scripts/clean_clinical_data.R')
  
  # Prepare interaction data
  interactions_clean <- prepare_interaction_data(interactions_file, clinical_data_selected)
  
  # Calculate interaction proportions and statistics
  interactions_summary <- calculate_interaction_proportions(interactions_clean)
  plot_data <- calculate_plot_statistics(interactions_summary)
  
  # Plot
  plot <- create_interaction_plot(
    plot_data,
    interactions_filter,
    paste(filter_name, "by Aneuploidy Status")
  )
  
  # Statistical tests
  pvals_ttest <- calculate_pvalues(interactions_summary, interactions_filter, method='t.test')
  pvals_betareg <- calculate_pvalues(interactions_summary, interactions_filter, method='betareg')
  
  # Output
  print("Statistical test results:")
  print("T test results:")
  print(pvals_ttest)
  print("Beta regression results:")
  print(pvals_betareg)
  
  print("Plot:")
  print(plot)
  
  # Summary statistics
  summary_stats <- generate_summary_stats(interactions_summary)
  print("Summary statistics:")
  print(summary_stats)
}


# EXAMPLE USAGE
# =========

# Set working directory and input file
setwd('/Volumes/scratch/DIMA/chiodin/tests/patients_analysis')
interactions_file <- 'data/phenotypes_interactions_20250730.csv'

# Interaction filters
T_CELL_TUMOR <- c("T regulatory-Tumor", "T helper-Tumor", "T cytotoxic-Tumor", 
                  "T cell-Tumor", "Immune-Tumor")

# Macrophage interactions with tumor
MACROPHAGE_TUMOR <- c("M1-Tumor", "M2-Tumor", "Macrophages-Tumor")

# Marker-specific interactions with tumor
MARKER_TUMOR <- c("PAX2+-Tumor", "L1CAM+-Tumor")

# Homotypic interactions
HOMOTYPIC <- c("Tumor-Tumor", "Immune-Immune", "T regulatory-T regulatory", 
               "T helper-T helper", "T cytotoxic-T cytotoxic", "T cell-T cell")

# Run pipeline
run_interaction_pipeline(interactions_file, T_CELL_TUMOR, "T Cell Interactions with Tumor")
run_interaction_pipeline(interactions_file, MACROPHAGE_TUMOR, "Macropages Interactions with Tumor")
run_interaction_pipeline(interactions_file, MARKER_TUMOR, "L1CAM+ and PAX2+ Interactions with Tumor")
run_interaction_pipeline(interactions_file, HOMOTYPIC, "Homotypic interactions (immune cells)")
