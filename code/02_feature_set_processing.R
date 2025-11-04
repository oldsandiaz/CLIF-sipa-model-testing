# Feature set creation for model development

# Load necessary libraries
print("Initialized Feature Set Processing Script")

library(arrow)
library(tidyverse)
library(stringr)
library(data.table)
library(tictoc)

# Clear env
rm(list = ls())

# Path
source("utils/config.R")
output_path <- config$output_path

# Load data
data <- read_parquet(file.path(output_path, "intermediate", "sipa_clif_cohort.parquet"))

# Convert to data.table
setDT(data)

# Vectorized SOFA score calculation

tic()
compute_sofa_score_vec <- function(p_f, s_f, platelets, bilirubin, map, dopamine, dobutamine, norepinephrine, epinephrine, gcs, creatinine) {
  
  # Respiration (PaO2/FiO2 or SaO2/FiO2 ratio)
  resp <- case_when(
    !is.na(p_f) & p_f >= 400 ~ 0,
    !is.na(p_f) & p_f >= 300 ~ 1,
    !is.na(p_f) & p_f >= 200 ~ 2,
    !is.na(p_f) & p_f >= 100 ~ 3,
    !is.na(p_f) ~ 4,
    !is.na(s_f) & s_f > 301 ~ 0,
    !is.na(s_f) & s_f >= 221 ~ 1,
    !is.na(s_f) & s_f >= 142 ~ 2,
    !is.na(s_f) & s_f >= 67 ~ 3,
    !is.na(s_f) ~ 4,
    TRUE ~ NA_real_
  )
  
  # Coagulation (platelet count)
  coag <- case_when(
    !is.na(platelets) & platelets >= 150 ~ 0,
    !is.na(platelets) & platelets >= 100 ~ 1,
    !is.na(platelets) & platelets >= 50 ~ 2,
    !is.na(platelets) & platelets >= 20 ~ 3,
    !is.na(platelets) ~ 4,
    TRUE ~ NA_real_
  )
  
  # Liver (bilirubin, mg/dl)
  liver <- case_when(
    !is.na(bilirubin) & bilirubin < 1.2 ~ 0,
    !is.na(bilirubin) & bilirubin < 2.0 ~ 1,
    !is.na(bilirubin) & bilirubin < 6.0 ~ 2,
    !is.na(bilirubin) & bilirubin < 12.0 ~ 3,
    !is.na(bilirubin) ~ 4,
    TRUE ~ NA_real_
  )
  
  # Cardiovascular
  cv <- case_when(
    (!is.na(dopamine) & dopamine > 15) | (!is.na(norepinephrine) & norepinephrine > 0.1) | (!is.na(epinephrine) & epinephrine > 0.1) ~ 4,
    (!is.na(dopamine) & dopamine > 5) | (!is.na(norepinephrine) & norepinephrine > 0 & norepinephrine <= 0.1) | (!is.na(epinephrine) & epinephrine > 0 & epinephrine <= 0.1) ~ 3,
    (!is.na(dopamine) & dopamine > 0 & dopamine <= 5) | (!is.na(dobutamine) & dobutamine > 0) ~ 2,
    !is.na(map) & map < 70 ~ 1,
    TRUE ~ 0
  )
  
  # CNS (GCS)
  cns <- case_when(
    !is.na(gcs) & gcs == 15 ~ 0,
    !is.na(gcs) & gcs >= 13 ~ 1,
    !is.na(gcs) & gcs >= 10 ~ 2,
    !is.na(gcs) & gcs >= 6 ~ 3,
    !is.na(gcs) ~ 4,
    TRUE ~ NA_real_
  )
  
  # Renal (creatinine, mg/dl)
  renal <- case_when(
    !is.na(creatinine) & creatinine < 1.2 ~ 0,
    !is.na(creatinine) & creatinine < 2.0 ~ 1,
    !is.na(creatinine) & creatinine < 3.5 ~ 2,
    !is.na(creatinine) & creatinine < 5.0 ~ 3,
    !is.na(creatinine) ~ 4,
    TRUE ~ NA_real_
  )
  
  rowSums(cbind(resp, coag, liver, cv, cns, renal), na.rm = TRUE)
}

# Calculate SOFA score for each hour
data[, sofa_score := compute_sofa_score_vec(p_f_imputed, s_f, min_plt_count, max_bilirubin, min_map, dopamine, dobutamine, norepinephrine, epinephrine, gcs_total, max_creatinine)]


# Summarize features

# Create datetime column
data[, meas_dttm := as.POSIXct(paste(meas_date, sprintf("%02d:00:00", meas_hour)), format = "%Y-%m-%d %H:%M:%S", tz = "UTC")]

# Define pre/post life support periods
data[, period := ifelse(meas_dttm < life_support_start, "pre", "post")]

# Define summary functions
summary_functions <- list(
  creatinine = function(x) max(x, na.rm = TRUE),
  platelets = function(x) min(x, na.rm = TRUE),
  p_f = function(x) min(x, na.rm = TRUE),
  dobutamine = function(x) max(x, na.rm = TRUE),
  phenylephrine = function(x) max(x, na.rm = TRUE),
  dopamine = function(x) max(x, na.rm = TRUE),
  epinephrine = function(x) max(x, na.rm = TRUE),
  gcs = function(x) min(x, na.rm = TRUE),
  bilirubin = function(x) max(x, na.rm = TRUE),
  map = function(x) min(x, na.rm = TRUE),
  s_f = function(x) min(x, na.rm = TRUE),
  norepinephrine = function(x) max(x, na.rm = TRUE),
  sofa_score = function(x) max(x, na.rm = TRUE),
  vasopressin = function(x) max(x, na.rm = TRUE),
  milrinone = function(x) max(x, na.rm = TRUE),
  angiotensin = function(x) max(x, na.rm = TRUE),
  metaraminol = function(x) max(x, na.rm = TRUE),
  norepinephrine_eq = function(x) max(x, na.rm = TRUE)
)

# Columns to summarize
vars_to_summarize <- c("max_creatinine", "min_plt_count", "p_f_imputed", 
                       "dobutamine", "phenylephrine", "dopamine", 
                       "epinephrine", "gcs_total", "max_bilirubin", 
                       "min_map", "s_f", "norepinephrine", 
                       "sofa_score", "vasopressin", "milrinone", "angiotensin", "metaraminol", "norepinephrine_eq")
names(vars_to_summarize) <- c("creatinine", "platelets", "p_f", "dobutamine", 
                              "phenylephrine", "dopamine", "epinephrine", 
                              "gcs", "bilirubin", "map", "s_f", "norepinephrine", 
                              "sofa_score", "vasopressin", "milrinone", "angiotensin", "metaraminol", "norepinephrine_eq")

# Summarize data
summarized_data <- data[, 
  lapply(names(vars_to_summarize), function(var) summary_functions[[var]](.SD[[vars_to_summarize[var]]])),
  by = .(hospitalization_id, period),
  .SDcols = vars_to_summarize
]

setnames(summarized_data, old = paste0("V", 1:length(vars_to_summarize)), new = names(vars_to_summarize))

# Reshape data to wide format
wide_data <- dcast(summarized_data, hospitalization_id ~ period, value.var = names(vars_to_summarize))


# Clean up infinite values
for (col in names(wide_data)) {
  if (is.numeric(wide_data[[col]])) {
    set(wide_data, i = which(is.infinite(wide_data[[col]])), j = col, value = NA)
  }
}

# Get patient demographics
patient_data <- unique(data[, .(patient_id, hospitalization_id, ethnicity_category, age_at_admission, race_category, sex_category, in_hospital_mortality)])

# Join summarized data with patient data
final_data <- merge(patient_data, wide_data, by = "hospitalization_id", all.x = TRUE)

# Create the directory if it doesn't exist
dir.create(file.path(output_path, "final"), showWarnings = FALSE, recursive = TRUE)

# Export final data
write_parquet(final_data, file.path(output_path, "final", "sipa_features.parquet"))
toc()
print(paste("Feature set exported to", file.path(output_path, "final")))
