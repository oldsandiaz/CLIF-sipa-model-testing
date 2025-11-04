# Load necessary libraries
print("Initializing Table1 Script")
library(arrow)
library(tidyverse)
library(stringr)
library(data.table)
library(tictoc)
library(DiagrammeR)
library(glue)
library(vtree)


tic()
# Clear env
rm(list = ls())

# Load data
source("utils/config.R")
output_path <- config$output_path
site_name <- config$site_name


print("Creating STROBE Diagram")
# Construct a STROBE diagram
inclusion_table <- read.csv(file.path(output_path, "exportable", "inclusion_table.csv"))

strobe_diagram <- DiagrammeR::grViz(glue("
  digraph STROBE {{
    graph [layout = dot, rankdir = TB]
    
    # Define node styles
    node [shape = box, style = filled, fillcolor = white, fontname = Arial]
    
    # Define nodes
    A [label = 'Total Hospitalizations\n(n = {inclusion_table[1,2]})']
    B [shape = point, style = invisible, width = 0, height = 0]
    C [label = 'Hospitalizations excluded\n(age < 18)\n(n = {inclusion_table[1,2] - inclusion_table[2,2]})', fillcolor = lightcoral]
    D [label = 'Age >= 18\n(n = {inclusion_table[2,2]})', fillcolor = lightblue]
    
    E [shape = point, style = invisible, width = 0, height = 0]
    F [label = 'Hospitalizations excluded\n(No ICU stay)\n(n = {inclusion_table[2,2] - inclusion_table[3,2]})', fillcolor = lightcoral]
    G [label = 'Went to the ICU\n(n = {inclusion_table[3,2]})', fillcolor = lightblue]

    H [shape = point, style = invisible, width = 0, height = 0]
    I [label = 'Hospitalizations excluded\n(No respiratory support data)\n(n = {inclusion_table[3,2] - inclusion_table[4,2]})', fillcolor = lightcoral]
    J [label = 'With respiratory support data\n(n = {inclusion_table[4,2]})', fillcolor = lightblue]

    K [shape = point, style = invisible, width = 0, height = 0]
    L [label = 'Hospitalizations excluded\n(No lab data)\n(n = {inclusion_table[4,2] - inclusion_table[5,2]})', fillcolor = lightcoral]
    M [label = 'With lab data\n(n = {inclusion_table[5,2]})', fillcolor = lightblue]

    N [shape = point, style = invisible, width = 0, height = 0]
    O [label = 'Hospitalizations excluded\n(No vitals data)\n(n = {inclusion_table[5,2] - inclusion_table[6,2]})', fillcolor = lightcoral]
    P [label = 'With vitals data\n(n = {inclusion_table[6,2]})', fillcolor = lightblue]

    Q [shape = point, style = invisible, width = 0, height = 0]
    R [label = 'Hospitalizations excluded\n(No meds data)\n(n = {inclusion_table[6,2] - inclusion_table[7,2]})', fillcolor = lightcoral]
    S [label = 'With meds data\n(n = {inclusion_table[7,2]})', fillcolor = lightblue]

    T [shape = point, style = invisible, width = 0, height = 0]
    U [label = 'Hospitalizations excluded\n(No GCS data)\n(n = {inclusion_table[7,2] - inclusion_table[8,2]})', fillcolor = lightcoral]
    V [label = 'With GCS data\n(n = {inclusion_table[8,2]})', fillcolor = lightblue]

    W [shape = point, style = invisible, width = 0, height = 0]
    X [label = 'Hospitalizations excluded\n(Not on life support for 6 consecutive hours)\n(n = {inclusion_table[8,2] - inclusion_table[9,2]})', fillcolor = lightcoral]
    Y [label = 'Final Cohort\n(n = {inclusion_table[9,2]})', fillcolor = lightblue]

    # Ranks
    {{rank = same; B; C}}
    {{rank = same; E; F}}
    {{rank = same; H; I}}
    {{rank = same; K; L}}
    {{rank = same; N; O}}
    {{rank = same; Q; R}}
    {{rank = same; T; U}}
    {{rank = same; W; X}}

    # Edges
    A -> B [arrowhead = none] 
    B -> C 
    B -> D
    D -> E [arrowhead = none]
    E -> F
    E -> G
    G -> H [arrowhead = none]
    H -> I
    H -> J
    J -> K [arrowhead = none]
    K -> L
    K -> M
    M -> N [arrowhead = none]
    N -> O
    N -> P
    P -> Q [arrowhead = none]
    Q -> R
    Q -> S
    S -> T [arrowhead = none]
    T -> U
    T -> V
    V -> W [arrowhead = none]
    W -> X
    W -> Y
  }}
"))

# Create the exportable directory if it doesn't exist
dir.create(file.path(output_path, "exportable"), showWarnings = FALSE, recursive = TRUE)

# Save the diagram as a PNG file
DiagrammeR::export_graph(strobe_diagram, file_name = file.path(output_path, "exportable", "strobe_diagram.png"))
print("STROBE diagram exported.")

# Load the SIPA features dataset
data <- read_parquet(file.path(output_path, "final", "sipa_features.parquet"))

# Correct column names by removing trailing underscore
setnames(data, names(data), sub("__", "_", names(data)))

# Average _pre and _post columns
pre_cols <- names(data)[grepl("_pre$", names(data))]
post_cols <- names(data)[grepl("_post$", names(data))]

base_vars <- sub("_pre$", "", pre_cols)

for (var in base_vars) {
  pre_col <- paste0(var, "_pre")
  post_col <- paste0(var, "_post")
  if (pre_col %in% names(data) && post_col %in% names(data)) {
    data <- data %>%
      mutate(!!var := rowMeans(select(., all_of(c(pre_col, post_col))), na.rm = TRUE))}}

# Helper for median (IQR), 10th/90th deciles, and NA count
summary_stats_na <- function(x) {
  non_na_count <- sum(!is.na(x))
  na_count <- sum(is.na(x))
  if (non_na_count == 0) {
    return(paste0("NA: ", na_count))
  }
  paste0(formatC(median(x, na.rm = TRUE), digits = 1, format = "f"),
         " (", 
         formatC(quantile(x, 0.25, na.rm = TRUE), digits = 1, format = "f"), ", ",
         formatC(quantile(x, 0.75, na.rm = TRUE), digits = 1, format = "f"), ")",
         "; ",
         formatC(quantile(x, 0.1, na.rm = TRUE), digits = 1, format = "f"), "-",
         formatC(quantile(x, 0.9, na.rm = TRUE), digits = 1, format = "f"),
         "; ", na_count)
}

# Summarize to hospitalization_id (ICU encounter) level
data_summary <- data %>%
  group_by(hospitalization_id) %>%
  summarise(
    patient_id = first(patient_id),
    in_hospital_mortality = first(in_hospital_mortality),
    sex_category = first(sex_category),
    race_category = first(race_category),
    ethnicity_category = first(ethnicity_category),
    age_at_admission = first(age_at_admission),
    p_f = median(p_f, na.rm = TRUE),
    s_f = median(s_f, na.rm = TRUE),
    platelets = median(platelets, na.rm = TRUE),
    bilirubin = median(bilirubin, na.rm = TRUE),
    map = median(map, na.rm = TRUE),
    dobutamine = median(dobutamine, na.rm = TRUE),
    dopamine = median(dopamine, na.rm = TRUE),
    norepinephrine = median(norepinephrine, na.rm = TRUE),
    phenylephrine = median(phenylephrine, na.rm = TRUE),
    epinephrine = median(epinephrine, na.rm = TRUE),
    gcs = median(gcs, na.rm = TRUE),
    creatinine = median(creatinine, na.rm = TRUE),
    sofa_score = median(sofa_score, na.rm = TRUE)
  ) %>%
  ungroup()

# Patient-level summary for demographics
patient_summary <- data_summary %>%
  group_by(patient_id) %>%
  summarise(
    race_category = first(race_category),
    ethnicity_category = first(ethnicity_category),
    sex_category = first(sex_category),
    in_hospital_mortality = first(in_hospital_mortality)
  )

# Define new variable display names
var_display_names <- list(
  "age_at_admission" = "Age (Median, IQR; 10th-90th Decile; NA)",
  "p_f" = "PaO2/FiO2 (Median, IQR; 10th-90th Decile; NA)",
  "s_f" = "SpO2/FiO2 (Median, IQR; 10th-90th Decile; NA)",
  "platelets" = "Platelet Count (Median, IQR; 10th-90th Decile; NA)",
  "bilirubin" = "Bilirubin (Median, IQR; 10th-90th Decile; NA)",
  "map" = "Mean Arterial Pressure (Median, IQR; 10th-90th Decile; NA)",
  "dobutamine" = "Dobutamine (Median, IQR; 10th-90th Decile; NA)",
  "dopamine" = "Dopamine (Median, IQR; 10th-90th Decile; NA)",
  "norepinephrine" = "Norepinephrine (Median, IQR; 10th-90th Decile; NA)",
  "phenylephrine" = "Phenylephrine (Median, IQR; 10th-90th Decile; NA)",
  "epinephrine" = "Epinephrine (Median, IQR; 10th-90th Decile; NA)",
  "gcs" = "Glasgow Coma Scale (Median, IQR; 10th-90th Decile; NA)",
  "creatinine" = "Creatinine (Median, IQR; 10th-90th Decile; NA)",
  "sofa_score" = "SOFA Score (Median, IQR; 10th-90th Decile; NA)"
)

table1 <- list()

# Number of ICU Encounters (N)
table1[["ICU Encounters (N)"]] <- nrow(data_summary)

# Number of Patients (N)
table1[["Patients (N)"]] <- nrow(patient_summary)

# In-Hospital Mortality Rate
mortality_n <- sum(patient_summary$in_hospital_mortality, na.rm = TRUE)
mortality_pct <- 100 * mean(patient_summary$in_hospital_mortality, na.rm = TRUE)
table1[["In-Hospital Mortality Rate (N, %)"]] <- paste0(mortality_n, " (", formatC(mortality_pct, digits = 1, format = "f"), "%)")

# Sex (N, % female) at patient level
n_female <- sum(patient_summary$sex_category == "Female", na.rm = TRUE)
pct_female <- 100 * mean(patient_summary$sex_category == "Female", na.rm = TRUE)
table1[["Sex (N, % Female)"]] <- paste0(n_female, " (", formatC(pct_female, digits = 1, format = "f"), "%)")

# Blank Race header row
table1[["Race (N, %)"]] <- ""

race_levels <- c("White", "Asian", "Native Hawaiian or Other Pacific Islander", 
                 "Black or African American", "Unknown", "Other", "American Indian or Alaska Native")
race_tab <- patient_summary %>%
  filter(race_category %in% race_levels) %>%
  group_by(race_category) %>%
  summarise(N = n(), .groups = "drop") %>%
  mutate(Total = sum(N),
         Percent = 100 * N / Total,
         Race = paste0(N, " (", formatC(Percent, digits = 1, format = "f"), "%)")) %>%
  select(race_category, Race)

for (level in race_levels) {
  val <- race_tab$Race[race_tab$race_category == level]
  table1[[paste0("  ", level)]] <- if (length(val) > 0) val else "0 (0.0%)"
}

# Blank Ethnicity header row
table1[["Ethnicity (N, %)"]] <- ""
ethnicity_levels <- c("Hispanic", "Non-Hispanic", "Unknown")
eth_tab <- patient_summary %>%
  filter(ethnicity_category %in% ethnicity_levels) %>%
  group_by(ethnicity_category) %>%
  summarise(N = n(), .groups = "drop") %>%
  mutate(Total = sum(N),
         Percent = 100 * N / Total,
         Eth = paste0(N, " (", formatC(Percent, digits = 1, format = "f"), "%)")) %>%
  select(ethnicity_category, Eth)

for (level in ethnicity_levels) {
  val <- eth_tab$Eth[eth_tab$ethnicity_category == level]
  table1[[paste0("  ", level)]] <- if (length(val) > 0) val else "0 (0.0%)"
}

# Numeric variables
numeric_vars <- c("age_at_admission", "p_f", "s_f", "platelets", "bilirubin", "map",
                  "dobutamine", "dopamine", "norepinephrine", 
                  "phenylephrine", "epinephrine", "gcs", "creatinine", "sofa_score")
for (v in numeric_vars) {
  display_name <- var_display_names[[v]]
  table1[[display_name]] <- summary_stats_na(data_summary[[v]])
}

# Build the table in the desired order
desired_order <- c(
  "ICU Encounters (N)",
  "Patients (N)",
  "In-Hospital Mortality Rate (N, %)",
  "Sex (N, % Female)",
  "Race (N, %)",
  paste0("  ", race_levels),
  "Ethnicity (N, %)",
  paste0("  ", ethnicity_levels),
  unname(unlist(var_display_names))
)

# Defensive extraction: always return blank if not found or not named
table1_df <- tibble::tibble(
  Variable = desired_order,
  !!site_name := sapply(desired_order, function(x) table1[[x]] %||% "")
)

# Export as a separate CSV file
write.csv(table1_df, file.path(output_path, "exportable", "table1.csv"), row.names = FALSE)
print("Table 1 exported as CSV to output/exportable")
toc()
