# Cohort Identification script for subsequent feature set processing
# REQUIRES 0a_respiratory_support_waterfall.R 

# Load libraries

library(knitr)
library(here)
library(tidyverse)
library(arrow)
library(gtsummary)
library(stringr)
library(duckdb)
library(DBI)
library(tictoc)
library(dtplyr)
library(data.table)
library(jsonlite)
library(fst)


print("Initialized Cohort Identification Script")
tic("Script Completion")

rm(list = ls())

# Helper function to create DuckDB connection
write_files_to_duckdb <- function(files, tables_location, file_type, con) {
  for (file_name in files) {
    table_name <- tools::file_path_sans_ext(file_name)
    file_path <- file.path(tables_location, paste0(file_name, file_type))
    if (grepl("\\.parquet$", file_type)) {
      DBI::dbExecute(con, sprintf(
        "CREATE OR REPLACE TABLE %s AS SELECT * FROM read_parquet('%s')",
        table_name, file_path
      ))
      message("Loaded parquet table: ", table_name)
    } else if (grepl("\\.csv$", file_type)) {
      DBI::dbExecute(con, sprintf(
        "CREATE OR REPLACE TABLE %s AS SELECT * FROM read_csv_auto('%s')",
        table_name, file_path
      ))
      message("Loaded csv table: ", table_name)
    } else if (grepl("\\.fst$", file_type)) {
      # fst not directly supported: read into R then write to DuckDB
      data <- fst::read.fst(file_path)
      duckdb::dbWriteTable(con, table_name, data, overwrite = TRUE)
      rm(data)
      message("Loaded fst table: ", table_name)
    } else {
      stop("Unsupported file format")
    }
  }
}

# Needed tables
true_tables <- list("clif_medication_admin_continuous", "clif_patient",
                    "clif_hospitalization", "clif_adt", "clif_respiratory_support",
                    "clif_labs", "clif_vitals", "clif_patient_assessments")

# Dates REQUIRED YYYY-MM-DD format
admission_date_min <-"2018-01-01"
admission_date_max <- "2023-12-31"

# Set paths

source("utils/config.R")

# Access configuration parameters
site_name <- config$site_name
tables_path <- config$tables_path
file_type <- config$file_type
output_path <- config$output_path


tic("Setting up DuckDB and loading data")
# Set up DuckDB connection
con <- duckdb::dbConnect(duckdb::duckdb(), dbdir = tempfile())
DBI::dbExecute(con, "INSTALL icu")
DBI::dbExecute(con, "LOAD icu")
write_files_to_duckdb(true_tables, tables_path, file_type, con)
toc()

# Function to track sample size for different slices
cohort_table <- data.frame(Reason = character(), N = integer(), stringsAsFactors = FALSE)
cohort_tracking <- function(exclusion_reason, current_data, cohort_table){
  n <- as.numeric(n_distinct(current_data$hospitalization_id))
  new_row <- data.frame(Reason = exclusion_reason, N = n)
  bind_rows(cohort_table, new_row)}

tic()
# Exclusion and Inclusion Criteria

hospitalization <- tbl(con, "clif_hospitalization") %>%
  select(hospitalization_id, patient_id, admission_dttm, discharge_dttm, age_at_admission, discharge_name, discharge_category) %>% 
  filter(discharge_dttm >= admission_dttm) %>% 
  filter(!is.na(discharge_dttm) & !is.na(admission_dttm))%>%
  # Filter by admission date -- not applicable for MIMIC 
  #filter(admission_dttm >= as.Date(admission_date_min) & admission_dttm <= as.Date(admission_date_max)) %>%
  collect()

# Start cohort tracking
cohort_table <- cohort_tracking("Total Hospitalizations", hospitalization, cohort_table)

# Age
# First slice: Exclusion criteria: Age < 18
hospitalization <- hospitalization %>%
  filter(age_at_admission >= 18)

# Pull IDs
hosp_ids <- hospitalization %>% 
  select(hospitalization_id)

# Save to cohort tracking table
cohort_table <- cohort_tracking("Age >= 18", hosp_ids, cohort_table)

# Location
adt_adults <- hosp_ids %>%
  inner_join(tbl(con, "clif_adt"), by = "hospitalization_id", copy = TRUE) %>%
  select(hospitalization_id, location_category) %>%
  distinct() %>%
  mutate(count = 1) %>% 
  pivot_wider(names_from = location_category, 
              values_from = count, values_fill = list(count = 0)) %>% 
  filter(icu == 1) %>% 
  select(hospitalization_id) %>% 
  distinct()

# Second slice: find all hospitalization IDs associated with an ICU or Ward stay ONLY
adt_adults <- hosp_ids %>%
  inner_join(tbl(con, "clif_adt"), by = "hospitalization_id", copy = TRUE) %>%
  select(hospitalization_id, location_category) %>%
  distinct() %>%
  mutate(count = 1) %>% 
  pivot_wider(names_from = location_category, 
              values_from = count, values_fill = list(count = 0)) %>% 
  filter(icu == 1) %>% 
  select(hospitalization_id) %>% 
  distinct()

hospitalization <- adt_adults %>% 
  left_join(hospitalization, by = "hospitalization_id")

hosp_ids <- hospitalization %>% 
  select(hospitalization_id)

# Save to cohort tracking table
cohort_table <- cohort_tracking("Went to the ICU", hosp_ids, cohort_table)

# Free up memory
rm(adt_adults)
## Create Hourly Timestamps for Each Hospitalization
# Get first and last vital times to get hourly timestamps for hospitalization
## Doing this by minimum and maximum vital time
vital_min_max_time <- tbl(con, "clif_vitals") %>% 
  inner_join(hosp_ids, copy = TRUE) %>% 
  group_by(hospitalization_id) %>% 
  summarise(
    min_dttm = min(recorded_dttm),
    max_dttm = max(recorded_dttm),
    .groups = "drop"
  ) %>%
  collect()

# Create a large dataframe that will store hourly data for each hospitalization ID

hosp_by_hour <- vital_min_max_time %>% 
  rowwise() %>%
  mutate(txnDt = list(seq.POSIXt(min_dttm, max_dttm, by = "hour"))) %>%
  unnest(txnDt) %>%
  mutate(meas_hour = hour(txnDt),
         meas_date = as.Date(txnDt)) %>%
  select(hospitalization_id, txnDt, meas_date, meas_hour) %>%
  arrange(hospitalization_id, txnDt, meas_date, meas_hour) %>%
  select(hospitalization_id, meas_date, meas_hour)

tic("Respiratory Support Data Processing")

# Respiratory Support
## Load data and rename columns
## The loaded file was processed by the script: 0a_respiratory_support_waterfall.R
resp_support <- read_parquet(file.path(output_path, "intermediate", "clif_respiratory_support_processed.parquet"))

# Convert to data.table
setDT(resp_support)

# Rename columns
setnames(resp_support, "recorded_date", "meas_date")
setnames(resp_support, "recorded_hour", "meas_hour")

## There are duplicates in meas_date, hour and hospitalization ID combinations
## For each hour, select the row that has the least number of NAs
resp_support[, n_na := rowSums(is.na(.SD))]
resp_support_min_na <- resp_support[resp_support[, .I[n_na == min(n_na)], by = .(hospitalization_id, meas_hour, meas_date)]$V1]
resp_support_min_na[, n_na := NULL]

## Remove remaining duplicates
resp_support_min_na <- unique(resp_support_min_na, by = c("hospitalization_id", "meas_hour", "meas_date"))

## Remove CPAP
resp_support_final <- resp_support_min_na[!(device_category == "CPAP" & (is.na(fio2_approx) | fio2_approx < 0.3))]

## Create a new device category column
vent_modes <- c("SIMV", "Pressure-Regulated Volume Control", 
                "Assist Control-Volume Control", "Pressure Support/CPAP", 
                "Pressure Control", "Volume Support", "assist control-volume control")

resp_support_final[, device_category_2 := fcase(
  mode_category %in% vent_modes, "Vent",
  device_category == "IMV", "Vent",
  !is.na(device_category), device_category,
  is.na(device_category) & fio2_approx == 0.21 & is.na(lpm_set) & is.na(peep_set) & is.na(tidal_volume_set), "Room Air",
  is.na(device_category) & is.na(fio2_approx) & lpm_set == 0 & is.na(peep_set) & is.na(tidal_volume_set), "Room Air",
  is.na(device_category) & is.na(fio2_approx) & lpm_set <= 20 & lpm_set > 0 & is.na(peep_set) & is.na(tidal_volume_set), "Nasal Cannula",
  is.na(device_category) & is.na(fio2_approx) & lpm_set > 20 & is.na(peep_set) & is.na(tidal_volume_set), "High Flow NC",
  device_category == "Nasal Cannula" & is.na(fio2_approx) & lpm_set > 20, "High Flow NC",
  default = device_category
)]

## Select needed columns from resp_support
resp_support_final <- resp_support_final[, .(hospitalization_id, device_category_2, recorded_dttm, fio2_approx, lpm_set, meas_hour, meas_date)]

## Create Device Category Ranking
## First, rank devices in order to select most intense device used for each hour
## Second, determine max FiO2 for each hour
## Third, associate most intense device with FiO2 for each hour

device_rank_lookup <- c(
  'Vent' = 1,
  'NIPPV' = 2,
  'CPAP' = 3,
  'High Flow NC' = 4,
  'Face Mask' = 5,
  'Trach Collar' = 6,
  'Nasal Cannula' = 7,
  'Other' = 8,
  'Room Air' = 9
)

# Apply device rankings
resp_support_final[, device_rank := device_rank_lookup[device_category_2]]

### Max fio2 per hour
fio2_max <- resp_support_final[!is.na(fio2_approx), .(fio2_approx = max(fio2_approx, na.rm = TRUE)), by = .(hospitalization_id, meas_date, meas_hour)]

### Group by person, measurement date and measurement hour 
### Get most intense device within each hour
device_max <- resp_support_final[!is.na(device_rank), .(device_rank = min(device_rank, na.rm = TRUE)), by = .(hospitalization_id, meas_date, meas_hour)]

# Create the reverse lookup vector
device_category_lookup <- names(device_rank_lookup)
names(device_category_lookup) <- device_rank_lookup

### Revert back to categories from rank
device_max[, device_category := device_category_lookup[as.character(device_rank)]]


# Free up memory
rm(resp_support_final)
rm(resp_support_min_na)
rm(resp_support)

### Add to encounters by hour
hosp_by_hour <- merge(hosp_by_hour, fio2_max, by = c("hospitalization_id", "meas_hour", "meas_date"), all.x = TRUE)
hosp_by_hour <- merge(hosp_by_hour, device_max, by = c("hospitalization_id", "meas_hour", "meas_date"), all.x = TRUE)

toc()

### Clear memory
rm(device_max)
rm(fio2_max)
rm(hospitalization)


## Carry Forward Device Name and FiO2
## Carry forward device name until another device is recorded 
## or the end of the measurement time window
hosp_by_hour <- hosp_by_hour %>%
  group_by(hospitalization_id) %>%
  fill(device_category, .direction = "down") %>%
  mutate(device_filled = ifelse(is.na(device_category), NA, device_category))

## Carry forward FiO2 measurement until another device is recorded 
## or the end of the measurement time window
hosp_by_hour <- hosp_by_hour %>%
  mutate(fio2_approx = as.double(fio2_approx)) %>%
  group_by(hospitalization_id, device_filled) %>%
  fill(fio2_approx, .direction = "down") %>%
  mutate(fio2_filled = ifelse(is.na(fio2_approx), NA, fio2_approx))

hosp_by_hour <- hosp_by_hour %>% 
  distinct() %>% 
  arrange(hospitalization_id, meas_date, meas_hour) %>% 
  select(hospitalization_id, meas_date, meas_hour, device_filled, fio2_filled)

cohort_table <- cohort_tracking("Adding Respiratory Support Data", hosp_by_hour, cohort_table)

tic("Adding Lab Data")
# Labs
required_lab_categories <- c("po2_arterial", "bilirubin_total", "platelet_count", "creatinine")

labs <- tbl(con, "clif_labs") %>%
  filter(lab_category %in% required_lab_categories) %>% 
  filter(
    (lab_category == "platelet_count" & lab_value_numeric > 0 & lab_value_numeric <= 832) | 
      (lab_category == "bilirubin_total" & lab_value_numeric > 0 & lab_value_numeric <= 50) |
      (lab_category == "po2_arterial" & lab_value_numeric > 30 & lab_value_numeric <= 800) |
      (lab_category == "creatinine" & lab_value_numeric > 0 & lab_value_numeric <= 30)
  ) %>% 
  collect()

## Next, for each lab, determine the worst value in a particular hour
### PaO2: Minimum and then forward filling

pao2 <- labs %>%
  filter(lab_category == "po2_arterial") %>% 
  mutate(meas_hour = hour(lab_result_dttm),
         meas_date = as.Date(lab_result_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, lab_category, lab_value_numeric) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(min_pao2 = min(lab_value_numeric, na.rm = TRUE)) %>%
  ungroup()

ids_by_hour <- hosp_by_hour %>% 
  select(hospitalization_id, meas_date, meas_hour)

pao2_hours <- left_join(ids_by_hour, pao2) %>%
  arrange(hospitalization_id, meas_date, meas_hour) %>%
  distinct() %>% 
  collect()

pao2_hours <- pao2_hours %>%
  mutate(dttm = as_datetime(paste0(meas_date, " ", meas_hour, ":00"), format = "%Y-%m-%d %H:%M"))

pao2_hours <- pao2_hours %>%
  mutate(last_measure = if_else(!is.na(min_pao2), dttm, NA)) %>%
  group_by(hospitalization_id) %>%
  fill(last_measure, .direction = "down") %>%
  ungroup()

# Calculate time difference between the hour we're trying to fill and the most recent PaO2, filter to only 3 additional hrs (4 total)
pao2_hours <- pao2_hours %>%
  mutate(hour_diff = as.numeric(difftime(dttm, last_measure, units = "hours"))) %>%
  filter(hour_diff >= 0 & hour_diff <= 3)

# Fill PaO2 forward
pao2_hours <- pao2_hours %>%
  group_by(hospitalization_id, last_measure) %>%
  fill(min_pao2, .direction = "down") %>%
  mutate(pao2_filled = if_else(is.na(min_pao2), NA_real_, min_pao2)) %>%
  ungroup()

pao2_filled <- pao2_hours %>%
  select(hospitalization_id, meas_date, meas_hour, pao2_filled)


### Bilirubin
bilirubin <- labs %>% 
  filter(lab_category == "bilirubin_total") %>% 
  mutate(meas_hour = hour(lab_result_dttm),
         meas_date = as.Date(lab_result_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, lab_category, lab_value_numeric) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(max_bilirubin = max(lab_value_numeric, na.rm = TRUE)) %>%
  ungroup()

### Platelets
plt_count <- labs %>% 
  filter(lab_category == "platelet_count") %>% 
  mutate(meas_hour = hour(lab_result_dttm),
         meas_date = as.Date(lab_result_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, lab_category, lab_value_numeric) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(min_plt_count = min(lab_value_numeric, na.rm = TRUE)) %>%
  ungroup()

### Creatinine
creatinine <- labs %>% 
  filter(lab_category == "creatinine") %>% 
  mutate(meas_hour = hour(lab_result_dttm),
         meas_date = as.Date(lab_result_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, lab_category, lab_value_numeric) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(max_creatinine = max(lab_value_numeric, na.rm = TRUE)) %>%
  ungroup()

## Add all lab data to main df

hosp_by_hour <- hosp_by_hour %>% 
  left_join(pao2_filled, by = c("hospitalization_id", "meas_hour", "meas_date")) %>% 
  left_join(creatinine, by = c("hospitalization_id", "meas_hour", "meas_date")) %>% 
  left_join(bilirubin, by = c("hospitalization_id", "meas_hour", "meas_date")) %>% 
  left_join(plt_count, by = c("hospitalization_id", "meas_hour", "meas_date"))

cohort_table <- cohort_tracking("Adding Lab Data", hosp_by_hour, cohort_table)

toc()
### Save some memory
rm(labs)
rm(bilirubin)
rm(creatinine)
rm(pao2_filled)
rm(pao2_hours)
rm(pao2)
rm(plt_count)
rm(vital_min_max_time)


tic("Adding Vitals")
# Vitals: SpO2 and MAP
# Need: SpO2 and MAP
required_vitals <- c("map", "spo2", "weight_kg")

vitals <- tbl(con, "clif_vitals") %>%
  filter(vital_category %in% required_vitals) %>% 
  filter(
    (vital_category == "spo2" & vital_value > 60 & vital_value <= 100) |
      (vital_category == "map" & vital_value > 0 ) |
      (vital_category == "weight_kg" & vital_value > 0 & vital_value <= 700)
  ) %>%
  inner_join(hosp_ids, by = "hospitalization_id", copy = TRUE) %>% 
  select(hospitalization_id, recorded_dttm, vital_category, vital_value) %>%
  collect()

# Get weight_kg 

weight_kg <- vitals %>% 
  filter(vital_category == "weight_kg") %>%
  select(hospitalization_id, recorded_dttm, weight_kg = vital_value) %>%
  mutate(
    meas_hour = hour(recorded_dttm),
    meas_date = as.Date(recorded_dttm)
  ) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarize(
    weight_kg = mean(as.numeric(weight_kg), na.rm = TRUE),
    .groups = "drop"
  )

# Get min SpO2 per hour
spo2 <- vitals %>%
  filter(vital_category == "spo2") %>% 
  mutate(meas_hour = hour(recorded_dttm),
         meas_date = as.Date(recorded_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, vital_category, vital_value) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(min_spo2 = min(vital_value, na.rm = TRUE)) %>%
  ungroup()

# Get min MAP per hour
map <- vitals %>%
  filter(vital_category == "map") %>% 
  mutate(meas_hour = hour(recorded_dttm),
         meas_date = as.Date(recorded_dttm)) %>%
  select(hospitalization_id, meas_date, meas_hour, vital_category, vital_value) %>%
  group_by(hospitalization_id, meas_date, meas_hour) %>%
  summarise(min_map = min(vital_value, na.rm = TRUE)) %>%
  ungroup()

## We can use SpO2 -> PaO2 conversion. This can help with PaO2 NAs.
## Source: Non-linear estimation from Brown et al in Critical Care Medicine August 2017 
## DOI: 10.1097/CCM.0000000000002514

calc_pao2 <- function(spo2) {
  spo2 <- spo2 / 100
  # Return NA for spo2 >= 1 or spo2 <= 0 to avoid divide by zero or log of negative
  invalid <- spo2 >= 1 | spo2 <= 0 | is.na(spo2)
  a <- 11700 / ((1 / spo2) - 1)
  b <- sqrt((50^3) + (a^2))
  pao2 <- ((b + a)^(1/3)) - ((b - a)^(1/3))
  pao2[invalid] <- NA_real_
  return(pao2)
}

# Add PaO2 calculation
spo2 <- spo2 %>%
  mutate(pao2_imputed = calc_pao2(min_spo2))

# Replace with NA if SpO2 >= 97
spo2 <- spo2 %>%
  mutate(pao2_imputed = ifelse(min_spo2>=97, NA, pao2_imputed))

# Join to main table
hosp_by_hour <- hosp_by_hour %>% 
  left_join(map, by = c("hospitalization_id", "meas_date", "meas_hour")) %>% 
  left_join(spo2, by = c("hospitalization_id", "meas_date", "meas_hour")) %>%
  left_join(weight_kg, by = c("hospitalization_id", "meas_date", "meas_hour")) %>%
  distinct()

cohort_table <- cohort_tracking("Adding SpO2, PaO2, and Weight", hosp_by_hour, cohort_table)
cohort_table

## Free up memory
rm(vitals)
rm(map)
rm(weight_kg)
rm(spo2)

toc()

# Create PaO2/FiO2 and SpO2/FiO2 ratio columns
hosp_by_hour <- hosp_by_hour %>%
  mutate(p_f = ifelse(!is.na(fio2_filled) & !is.na(pao2_filled), pao2_filled / fio2_filled, NA),
         p_f_imputed = ifelse(!is.na(fio2_filled) & !is.na(pao2_imputed), pao2_imputed / fio2_filled, NA),
         s_f = ifelse(!is.na(fio2_filled) & !is.na(min_spo2), min_spo2 / fio2_filled, NA)) %>%
  distinct()

# Forward fill weight_kg
hosp_by_hour <- hosp_by_hour %>%
  group_by(hospitalization_id) %>%
  fill(weight_kg, .direction = "down") %>%
  ungroup()


tic("Adding Medications")
# MEDICATIONS

# The following medication columns must be used across the consortium:
med_vars <- c(
  "norepinephrine",
  "epinephrine",
  "phenylephrine",
  "dopamine",
  "metaraminol",
  "vasopressin",
  "angiotensin",
  "dobutamine")

med_vars_ne_eq <- paste0(med_vars, "_ne_eq")

# Function to add missing medication columns
add_missing_meds <- function(df, meds) {
  missing <- setdiff(meds, names(df))
  df[missing] <- 0
  df
}

pressors <- tbl(con, "clif_medication_admin_continuous") %>%
  filter(med_group == "vasoactives") %>%
  mutate(meas_hour = hour(admin_dttm),
         meas_date = as.Date(admin_dttm)) %>%
  select(hospitalization_id, meas_hour, meas_date, med_category, med_dose, med_dose_unit) %>%
  distinct() %>%
  inner_join(hosp_ids, by = "hospitalization_id", copy = TRUE) %>% 
  collect()

# Add weight_kg
pressors_with_weight <- pressors %>%
  left_join(hosp_by_hour %>% select(hospitalization_id, meas_date, meas_hour, weight_kg), 
            by = c("hospitalization_id", "meas_date", "meas_hour")) %>% 
  distinct()

# Function to get norepinephrine equivalent conversion factors
get_conversion_factor <- function(med_category, med_dose_unit, weight_kg) {
  # 1) Retrieve medication info (if missing, return NA)
  med_info <- med_unit_info[[med_category]]
  if (is.null(med_info)) {
    return(NA_real_)
  }
  # 2) Convert the incoming unit to lowercase
  med_dose_unit <- tolower(med_dose_unit)
  # 3) Check if it's an acceptable unit; if not, return NA
  if (!(med_dose_unit %in% med_info$acceptable_units)) {
    return(NA_real_)
  }
  # 4) Determine conversion factor for each med+unit
  factor <- NA_real_
  # Group 1: norepinephrine, epinephrine, phenylephrine, dopamine, metaraminol, dobutamine
  #   required_unit: "mcg/kg/min"
  if (med_category %in% c("norepinephrine", "epinephrine", "phenylephrine",
                          "dopamine", "milrinone", "dobutamine", "metaraminol")) {
    if (med_dose_unit == "mcg/kg/min") {
      factor <- 1
    } else if (med_dose_unit == "mcg/kg/hr") {
      factor <- 1 / 60
    } else if (med_dose_unit == "mg/kg/hr") {
      factor <- 1000 / 60
    } else if (med_dose_unit == "mcg/min") {
      factor <- 1 / weight_kg
    } else if (med_dose_unit == "mg/hr") {
      factor <- (1000 / 60) / weight_kg
    } else {
      return(NA_real_)
    }
    # Group 2: angiotensin
    #   required_unit: "mcg/kg/min"
  } else if (med_category == "angiotensin") {
    if (med_dose_unit == "ng/kg/min") {
      factor <- 1 / 1000
    } else if (med_dose_unit == "ng/kg/hr") {
      factor <- 1 / 1000 / 60
    } else {
      return(NA_real_)
    }
    # Group 3: vasopressin
    #   required_unit: "units/min"
  } else if (med_category == "vasopressin") {
    if (med_dose_unit == "units/min") {
      factor <- 1
    } else if (med_dose_unit == "units/hr") {
      factor <- 1 / 60
    } else if (med_dose_unit == "milliunits/min") {
      factor <- 1 / 1000
    } else if (med_dose_unit == "milliunits/hr") {
      factor <- 1 / 1000 / 60
    } else {
      return(NA_real_)
    }
    # If none of the above, return NA
  } else {
    return(NA_real_)
  }
  return(factor)
}

# Unit information for each medication
med_unit_info <- list(
  norepinephrine = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")),
  epinephrine = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")),
  phenylephrine = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")),
  vasopressin = list(
    required_unit = "units/min",
    acceptable_units = c("units/min", "units/hr", "milliunits/min", "milliunits/hr")),
  dopamine = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")),
  angiotensin = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("ng/kg/min", "ng/kg/hr")),
  dobutamine = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")),
  milrinone = list(
    required_unit = "mcg/kg/min",
    acceptable_units = c("mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr")))

# Convert medication doses to norepinephrine equivalents
# Name: med name + ne_equiv
# Source: Vasopressor dose equivalence: A scoping review and suggested formula by Goradia et al. from Journal of Critical Care Medicine
# DOI: 10.1016/j.jcrc.2020.11.002

## First convert med_dose_unit to lowercase
pressors_with_weight <- pressors_with_weight %>%
  mutate(med_dose_unit = tolower(med_dose_unit))

# Apply conversion function to get conversion factor
pressors_with_weight <- pressors_with_weight %>%
  mutate(
    conversion_factor = mapply(get_conversion_factor, med_category, med_dose_unit, weight_kg),
    med_dose_converted = round(med_dose * conversion_factor, 3))

# Pivot wide for normal med doses
med_dose_wide <- pressors_with_weight %>%
  select(hospitalization_id, med_category, med_dose_converted, 
         meas_hour, meas_date) %>%
  pivot_wider(
    names_from = med_category,
    values_from = med_dose_converted,
    values_fn = max) %>% 
  add_missing_meds(med_vars) %>% 
  arrange(hospitalization_id, meas_date, meas_hour)

# Pivot wide for norepinephrine equivalents
# Adds norepinephrine_eq formula 
med_dose_converted_wide <- pressors_with_weight %>%
  mutate(med_category_ne_eq = paste0(med_category, "_ne_eq")) %>%
  select(hospitalization_id, med_category_ne_eq, 
         med_dose_converted, meas_hour, meas_date) %>%
  pivot_wider(
    names_from = med_category_ne_eq,
    values_from = med_dose_converted,
    values_fn = max) %>% 
  add_missing_meds(med_vars_ne_eq) %>%
  mutate(norepinephrine_eq =
      coalesce(norepinephrine_ne_eq, 0) +
      coalesce(epinephrine_ne_eq, 0) +
      coalesce(phenylephrine_ne_eq, 0) / 10 +
      coalesce(dopamine_ne_eq, 0) / 100 +
      coalesce(metaraminol_ne_eq, 0) / 8 +
      coalesce(vasopressin_ne_eq, 0) * 2.5 +
      coalesce(angiotensin_ne_eq, 0) * 10) %>% 
  arrange(hospitalization_id, meas_date, meas_hour) %>% 
  select(hospitalization_id, meas_date, meas_hour, norepinephrine_eq)

# Combine both wide dataframes
pressors_wide <- med_dose_wide %>%
  left_join(med_dose_converted_wide, 
            by = c("hospitalization_id", "meas_hour", "meas_date")) %>%
  distinct()

## Get max num pressors (cap at 4) per hour
## Keep dobutamine alone
max_pressors <- pressors %>%
  group_by(hospitalization_id, meas_hour, meas_date) %>%
  summarise(
    med_list = unique(med_category),
    num_pressors = n_distinct(med_category),
    num_pressors = ifelse(num_pressors > 4, 4, num_pressors),
    dobutamine_alone = as.integer(all(med_list == "dobutamine") & length(med_list) == 1),
    .groups = "drop")


## Merge max number of pressors and pressor doses to main dataframe hosp_by_hour!
hosp_by_hour <- hosp_by_hour %>% 
  left_join(pressors_wide, by = c("hospitalization_id", 
                                  "meas_date", "meas_hour")) %>% 
  left_join(max_pressors, by = c("hospitalization_id", 
                                 "meas_date", "meas_hour")) %>%
  distinct()

hosp_by_hour <- hosp_by_hour %>% 
  select(-med_list) %>% 
  distinct()

cohort_table <- cohort_tracking("Adding Meds", hosp_by_hour, cohort_table)
cohort_table

## Free up memory
rm(max_pressors)
rm(pressors_with_weight)
rm(pressors_wide)
rm(med_dose_wide)
rm(med_dose_converted_wide)
rm(pressors)

toc()


tic("Adding Assessments")
# Assessments
# Need: GCS
## Load data
assessments <- tbl(con, "clif_patient_assessments") %>% 
  filter(assessment_category == "gcs_total") %>%
  filter(numerical_value > 0) %>%
  mutate(meas_hour = hour(recorded_dttm),
         meas_date = as.Date(recorded_dttm)) %>% 
  select(hospitalization_id, meas_hour, meas_date, 
         assessment_category, numerical_value) %>% 
  distinct() %>% 
  inner_join(hosp_ids, by = "hospitalization_id", copy = TRUE) %>% 
  collect()

assessments <- assessments %>% 
  pivot_wider(names_from = assessment_category, 
              values_from = numerical_value,
              values_fn = min)

## Merge with main dataframe hosp_by_hour!
hosp_by_hour <- hosp_by_hour %>% 
  left_join(assessments, by = c("hospitalization_id", 
                                "meas_date", "meas_hour"))

rm(assessments)

cohort_table <- cohort_tracking("Adding GCS", hosp_by_hour, cohort_table)

toc()


tic("Flagging Life Support")
# Life Support
## Flag if on life support in a given hour
hosp_by_hour <- hosp_by_hour %>%
  mutate(
    on_life_support = case_when(
      device_filled %in% c('NIPPV', 'Vent', 'High Flow NC') ~ 1,
      p_f < 200 ~ 1,
      p_f_imputed < 200 ~ 1,
      num_pressors >= 1 ~ 1,
      TRUE ~ 0
    ),
    life_support_reason = case_when(device_filled %in% c('NIPPV', 'Vent', 'High Flow NC') ~ device_filled,
    p_f < 200 | p_f_imputed < 200 ~ "P:F < 200",
    num_pressors >= 1 ~ "Pressors",
    TRUE ~ NA_character_)) %>% 
  group_by(hospitalization_id) %>%
  fill(life_support_reason, .direction = "down") %>%
  ungroup()


## Create leading flags to identify first episode of 6 consecutive hours of life support
hosp_by_hour <- hosp_by_hour %>%
  arrange(hospitalization_id, meas_date, meas_hour) %>%
  group_by(hospitalization_id) %>%
  mutate(
    lead_1 = lead(on_life_support, 1),
    lead_2 = lead(on_life_support, 2),
    lead_3 = lead(on_life_support, 3),
    lead_4 = lead(on_life_support, 4),
    lead_5 = lead(on_life_support, 5)) %>%
  mutate(
    across(c(on_life_support, lead_1, lead_2, lead_3, lead_4, lead_5), ~ replace_na(.x, 0))) %>%
  mutate(
    lead_values = rowSums(across(c(on_life_support, lead_1, lead_2, lead_3, lead_4, lead_5))),
    life_support_sum = ifelse(lead_values == 6, 6, 0)) %>%
  ungroup()

## Get time of first episode of 6 consecutive hours of life support started
ls_encs <- hosp_by_hour %>%
  filter(life_support_sum == 6) %>%
  mutate(life_support_start = as.POSIXct(paste0(meas_date, " ", meas_hour, ":00"), 
                                         format = "%Y-%m-%d %H:%M", tz="UTC")) %>%
  group_by(hospitalization_id) %>%
  summarise(life_support_start = min(life_support_start, na.rm=T)) %>%
  ungroup() %>%
  mutate(window_start = as.POSIXct((life_support_start - hours(42)), tz="UTC"),
         window_end = as.POSIXct((life_support_start + hours(5)), tz="UTC"))

## Explode between window start and window end to get all hourly timestamps
ls_encs_hours <- ls_encs %>%
  rowwise() %>%
  mutate(txnDt = list(seq.POSIXt(window_start, window_end, by = "hour"))) %>%
  unnest(txnDt) %>%
  mutate(meas_hour = hour(txnDt),
         meas_date = as.Date(txnDt)) %>% 
  arrange(hospitalization_id, txnDt) 

## The window that we care about is 42 hours pre and 6 hours post life support start
## where patient is on life support for at least 6 hours after starting

hosp_by_hour_ls <- ls_encs_hours %>% 
  left_join(hosp_by_hour, by = c("hospitalization_id", "meas_hour", "meas_date"))

cohort_table <- cohort_tracking("Life Support For At Least 6 Hours", hosp_by_hour_ls, cohort_table)

toc()

# Cohort Demographics
# Bring in patient demographics
hosp_patient_ids <- tbl(con, "clif_hospitalization") %>% 
  select(hospitalization_id, patient_id, age_at_admission,
         admission_dttm, discharge_dttm) %>% 
  distinct() %>% 
  collect()

patient <-  tbl(con, "clif_patient") %>%
  select(patient_id, race_category, ethnicity_category, sex_category, death_dttm) %>% 
  distinct() %>% 
  collect()  

# Determine in-hospital mortality
hosp_with_mortality <- hosp_patient_ids %>% 
  left_join(patient, by = "patient_id") %>% 
  mutate(in_hospital_mortality = if_else(
    !is.na(death_dttm) & death_dttm >= admission_dttm & death_dttm <= discharge_dttm,
    1, 0
  )) %>% 
  distinct()

# Finalize cohort and export
cohort <- hosp_by_hour_ls %>% 
  left_join(hosp_with_mortality, by = "hospitalization_id")

cohort_final <- cohort %>% 
  select(-lead_1, -lead_2, -lead_3, -lead_4, -lead_5, 
         -life_support_sum, -lead_values, -pao2_filled, -p_f) %>%
  distinct()


cohort_table <- cohort_tracking("Final", cohort_final, cohort_table)

toc()

# Create the directories if they don't exist
dir.create(file.path(output_path, "exportable"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(output_path, "intermediate"), showWarnings = FALSE, recursive = TRUE)

write.csv(cohort_table, file.path(output_path, "exportable", "inclusion_table.csv"), row.names = FALSE)
write_parquet(cohort_final, file.path(output_path, "intermediate", paste0("sipa_clif_cohort", file_type)))
print(paste("Data exported as parquet to", file.path(output_path, "intermediate")))
print(paste("Cohort tracking table exported as csv to", file.path(output_path, "exportable")))
print(cohort_table)
duckdb::dbDisconnect(con, shutdown=TRUE)
