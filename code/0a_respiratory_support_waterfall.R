# Respiratory Support Waterfall Script
# Processes clif_respiratory_support table for use in cohort_identification
# and cohort_characterization

rm(list = ls())

# Access configuration parameters
source("utils/config.R")
site_name <- config$site_name
tables_path <- config$tables_path
file_type <- config$file_type
output_path <- config$output_path

# Load necessary libraries
library(lubridate)
library(arrow)
library(collapse)
library(tictoc)
library(data.table)
library(stringr)
library(tidyverse)

# Load the respiratory support data
clif_respiratory_support <- read_parquet(file.path(tables_path, paste0("clif_respiratory_support", file_type)))
clif_adt <- read_parquet(file.path(tables_path, paste0("clif_adt", file_type)))

# Process the data
# Courtesy of Nick Ingraham
#~~~~~~~~~~~~~~~~
##~~ getting an hour sequence so we can fill in the gaps
#~~~~~~~~~~~~~~~~
## This is just encounter ID and recorded times at xx:59:59
## data that occurs last in the hour when there are multiple data points 
## in the end ... we will want this to be the data we use to fill the next hour... if time is NOT unified... you could have hour sequence that is 12:01, 1:01 everywhere and even when there IS data you risk filling in from the hour before and not getting the NEW data during that hour.
# if we set all the new seq hours to 59:59 then you can fill those in without risking other data when you fill in and do distinct (take the first of the hour for everything).  Remember.  Even hours with 1 data will have a new hour seq row that may be before or after the data... so doing the 59:59 puts it at the end!!!
# 

hour_sequence <- clif_respiratory_support |> 
  group_by(hospitalization_id)  |> 
  reframe(recorded_dttm = seq(fmin(recorded_dttm), fmax(recorded_dttm), by = "1 hour")) |>
  # Adjust to the last second of the hour using lubridate's floor_date
  mutate(recorded_dttm = floor_date(recorded_dttm, "hour") + minutes(59) + seconds(59)) |> 
  # Create date and hour columns efficiently
  mutate(recorded_date = as_date(recorded_dttm),
         recorded_hour = hour(recorded_dttm)) |> 
  ungroup()

## Quality Check & Clean + Waterfall

#~~~~~~~~~~~~~~~~
##~~ Quick QA and fixing missing values throughout
#~~~~~~~~~~~~~~~~
tic()
df_resp_support_1  <- clif_respiratory_support |> 
  dplyr::select(hospitalization_id, hospitalization_id, recorded_dttm, device_category, device_name, mode_category, mode_name, 
                fio2_set, lpm_set, tidal_volume_set, peep_set,  pressure_support_set, resp_rate_set, tracheostomy, 
                starts_with("tidal_volume_obs"), peak_inspiratory_pressure_obs, 
                starts_with("minute_vent_obs"), 
                plateau_pressure_obs, starts_with("mean_airway_pressure_obs") 
  )

if ("mean_airway_pressure_obs" %in% names(df_resp_support_1)){
  df_resp_support_1 <- df_resp_support_1 |> 
    mutate(
      
      # mean_airway_pressure_obs
      mean_airway_pressure_obs = fcase(mean_airway_pressure_obs > 60, NA_real_, rep_len(TRUE, length(mean_airway_pressure_obs)), mean_airway_pressure_obs),
      mean_airway_pressure_obs = fcase(mean_airway_pressure_obs <  0, NA_real_, rep_len(TRUE, length(mean_airway_pressure_obs)), mean_airway_pressure_obs)
      
    )
  print("QC for MAIRP done!!")
}


df_resp_support_1 <- df_resp_support_1 |> 
  mutate(
    device_name = tolower(device_name),
    # fio2_set
    fio2_set = fcase(fio2_set > 1, NA_real_, rep_len(TRUE, length(fio2_set)), fio2_set),
    fio2_set = fcase(fio2_set <  .21, NA_real_, rep_len(TRUE, length(fio2_set)), fio2_set),
    
    # Set tidal_volume_set
    tidal_volume_set = fcase(tidal_volume_set > 2500, NA_real_, rep_len(TRUE, length(tidal_volume_set)), tidal_volume_set),
    tidal_volume_set = fcase(tidal_volume_set <   50, NA_real_, rep_len(TRUE, length(tidal_volume_set)), tidal_volume_set),
    
    # peep_set
    peep_set = fcase(peep_set > 30, NA_real_, rep_len(TRUE, length(peep_set)), peep_set),
    peep_set = fcase(peep_set <  0, NA_real_, rep_len(TRUE, length(peep_set)), peep_set),
    
    # pressure_support_set (sometimes APRV may be in here ... so limit ~ 50??)
    pressure_support_set = fcase(pressure_support_set > 50, NA_real_, rep_len(TRUE, length(pressure_support_set)), pressure_support_set),
    pressure_support_set = fcase(pressure_support_set <  0, NA_real_, rep_len(TRUE, length(pressure_support_set)), pressure_support_set),
    
    # resp_rate_set
    resp_rate_set = fcase(resp_rate_set > 60, NA_real_, rep_len(TRUE, length(resp_rate_set)), resp_rate_set),
    resp_rate_set = fcase(resp_rate_set <  0, NA_real_, rep_len(TRUE, length(resp_rate_set)), resp_rate_set),
    
    # tidal_volume_obs
    # tidal_volume_obs = fcase(tidal_volume_obs > 2500, NA_real_, rep_len(TRUE, length(tidal_volume_obs)), tidal_volume_obs),
    # tidal_volume_obs = fcase(tidal_volume_obs <    0, NA_real_, rep_len(TRUE, length(tidal_volume_obs)), tidal_volume_obs),
    
    # peak_inspiratory_pressure_obs
    peak_inspiratory_pressure_obs = fcase(peak_inspiratory_pressure_obs > 60, NA_real_, rep_len(TRUE, length(peak_inspiratory_pressure_obs)), peak_inspiratory_pressure_obs),
    peak_inspiratory_pressure_obs = fcase(peak_inspiratory_pressure_obs <  0, NA_real_, rep_len(TRUE, length(peak_inspiratory_pressure_obs)), peak_inspiratory_pressure_obs),
    
    # # minute_vent_obs
    # minute_vent_obs = fcase(minute_vent_obs > 30, NA_real_, rep_len(TRUE, length(minute_vent_obs)), minute_vent_obs),
    # minute_vent_obs = fcase(minute_vent_obs <  0, NA_real_, rep_len(TRUE, length(minute_vent_obs)), minute_vent_obs),
    # 
    
  ) |>
  
  # getting data and hour information
  mutate(recorded_date = date(recorded_dttm),
         recorded_hour = hour(recorded_dttm)) |> 
  
  # getting hospital ID for each hour
  dplyr::left_join( #tidy_table doesn't like it when you use join_by() with between (Dropped tidy_table 2_2024) 
    clif_adt |> 
      dplyr::select(hospitalization_id, hospital_id, location_name, location_category, in_dttm, out_dttm),
    by = join_by(hospitalization_id, between(recorded_dttm, in_dttm, out_dttm))
  ) |> 
  
  # order for filling things in
  arrange(hospitalization_id, recorded_dttm) |> 
  
  
  
  # Fixing when: the mode and category are there with device_name and device_cat not filled in.  fixing with the below
  mutate(
    device_category = 
      fcase(
        is.na(device_category) & is.na(device_name) &
          str_detect(mode_category, "Assist-Control Volume-Control|SIMV|Pressure Control"),
        "IMV",
        rep_len(TRUE, length(device_category)), device_category
      ),
    device_name = 
      fcase(
        str_detect(device_category, "IMV") & is.na(device_name) &
          str_detect(mode_category, "Assist-Control Volume-Control|SIMV|Pressure Control"),
        "mechanical ventilator",
        rep_len(TRUE, length(device_name)), device_name
        
      ),
  ) |>
  
  # fixing other vent things
  #     If device before is VENT + normal vent things ... its VENT too 
  mutate(device_category = fcase(is.na(device_category) & 
                                   lag(device_category == "IMV") & 
                                   tidal_volume_set > 1 & 
                                   resp_rate_set > 1 & 
                                   peep_set > 1, 
                                 "IMV", 
                                 rep_len(TRUE, length(device_category)), device_category)) |>
  
  #     If device after is VENT + normal vent things ... its VENT too 
  mutate(device_category = fcase(is.na(device_category) & 
                                   lead(device_category == "IMV") & 
                                   tidal_volume_set > 1 & 
                                   resp_rate_set > 1 & 
                                   peep_set > 1, 
                                 "IMV", 
                                 rep_len(TRUE, length(device_category)), device_category)) |>
  
  # same as above for device_name ^^^^^^^^^^^
  mutate(device_name = fcase(is.na(device_name) & lag(device_category == "IMV") & tidal_volume_set > 1 & resp_rate_set > 1 & peep_set > 1, 
                             "mechanical ventilation", 
                             rep_len(TRUE, length(device_name)), device_name)) |> 
  
  mutate(device_name = fcase(is.na(device_name) & lead(device_category == "IMV") & tidal_volume_set > 1 & resp_rate_set > 1 & peep_set > 1, 
                             "mechanical ventilation", 
                             rep_len(TRUE, length(device_name)), device_name)) |> 
  
  
  # doing this for BiPAP as well 
  mutate(device_category = fcase(is.na(device_category) & 
                                   lag(device_category == "NIPPV") & 
                                   # minute_vent_obs > 1 & 
                                   peak_inspiratory_pressure_obs > 1 & 
                                   pressure_support_set > 1, 
                                 "NIPPV", 
                                 rep_len(TRUE, length(device_category)), device_category)) |>
  
  mutate(device_category = fcase(is.na(device_category) & 
                                   lead(device_category == "NIPPV") & 
                                   # minute_vent_obs > 1 & 
                                   peak_inspiratory_pressure_obs > 1 & 
                                   pressure_support_set > 1, 
                                 "NIPPV", 
                                 rep_len(TRUE, length(device_category)), device_category)) |>
  
  
  
  # there are times when its clearly back to CMV (resp set and volume is set but no one puts a mode back in... just leaves it blank)
  # this is usually after pressure support ... we need to classify this now as CMV. 
  # only exception to this should be when it says trach
  # There are also some without device_cat or name and they have all the variables... these should be changed too 
  mutate(
    device_category = 
      fcase(
        is.na(device_category) & 
          (lag(device_category == "IMV") | lead(device_category == "IMV")) & 
          !str_detect(device_name, "trach") &
          tidal_volume_set > 0 & 
          resp_rate_set > 0,
        "IMV",
        rep_len(TRUE, length(device_category)), device_category),
    device_name = 
      fcase(
        is.na(device_name) & 
          (lag(device_category == "IMV") | lead(device_category == "IMV")) & 
          !str_detect(device_name, "trach") &
          tidal_volume_set > 0 & 
          resp_rate_set > 0,
        "mechanical ventilator",
        rep_len(TRUE, length(device_name)), device_name),
    mode_category = 
      fcase(
        is.na(mode_category) & 
          (lag(device_category == "IMV") | lead(device_category == "IMV")) & 
          !str_detect(device_name, "trach") &
          tidal_volume_set > 0 & 
          resp_rate_set > 0,
        "Assist Control Volume-Control",
        rep_len(TRUE, length(mode_category)), mode_category),
    mode_name = 
      fcase(
        is.na(mode_name) & 
          (lag(device_category == "IMV") | lead(device_category == "IMV")) & 
          !str_detect(device_name, "trach") &
          tidal_volume_set > 0 & 
          resp_rate_set > 0,
        "cmv/ac",
        rep_len(TRUE, length(mode_name)), mode_name)
  ) |> 
  
  
  # when there are duplicate times...
  group_by(hospitalization_id, recorded_dttm) |> 
  
  # when bipap is part of a duplicate we need to get rid of it... 
  #     its usually when a vent is STARTED and device is carried over but it goes to a new line with lots of NAs
  #     the NA line above has the vent settings.  Its best to just drop the NIPPV line when its a duplicate
  #     if we don't do this... the vent settings get sent backwards across all bipap
  
  mutate(n = n()) |>  
  filter(
    #  essentially this is... DROP if n>1 and device_cat == NIPPV
    !(n > 1 & device_category == "NIPPV")) |> 
  
  # redo n so we keep vent settings from above... now NAs are bad around other things and we should just drop
  mutate(n = n()) |> 
  filter(
    #  essentially this is... DROP if n>1 and device_cat == NA
    !(n > 1 & is.na(device_category))) |> 
  
  # random carried over bipap sometimes when there is trach next and there is vent before
  filter(
    !(device_category == "NIPPV" & lead(device_category == "Trach Collar") & lag(device_category != "NIPPV"))
  ) |>
  
  
  # filter if missing everything  
  filter(
    #  essentially this is... DROP if everything missing
    !(is.na(device_category) & 
        is.na(device_name) &
        is.na(mode_category) &
        is.na(mode_name) & 
        is.na(fio2_set) &        # keeps informative fio2_set data around
        is.na(tidal_volume_set)    # keeps vent data around... this happens sort of often
    )) |> 
  
  
  
  # dropping duplicates for everything else but just taking the first one
  #       ffirst works WAY faster than fill up and down and slicing(1)
  ffirst() |> 
  
  ungroup() |> # technically don't need this  
  
  # random nasal cannula surrounded by IMV before and after
  group_by(hospitalization_id) |>
  arrange(hospitalization_id, recorded_dttm) |>
  filter(
    !(device_category == "Nasal Cannula" & lead(device_category == "IMV") & lag(device_category == "IMV"))
  ) |>
  ###########
# TEMP STOP #
  ungroup()
###########
###########

toc()

## Resp support 

tic()

df_resp_support <- df_resp_support_1 |> 
  
  # bring in hour sequences
  bind_rows(hour_sequence) |> 
  
  
  #~~~~~~~~~~~~~~~~
  ##~~ Filling in data based on a waterfall of categories to ensure accuracy
  #~~~~~~~~~~~~~~~~
  # organizing
  arrange(hospitalization_id, recorded_dttm) |> 
  relocate(hospitalization_id, recorded_dttm, recorded_date, recorded_hour) |> 
  
  # fill forward device category
  group_by(hospitalization_id) |> 
  arrange(hospitalization_id, recorded_dttm) |> 
  fill(device_category) |>
  ungroup() |> 
  
  # Record a new device_category when either (a) a new encounter, or (b) preceded by a...   
  # different device category
  mutate(
    # need to have NA as something so it gets an ID
    device_cat_f = fcase(is.na(device_category), "missing", rep_len(TRUE, length(device_category)), device_category), # cant have anything with NAs when factoring
    device_cat_f = as.integer(as.factor(device_cat_f)), # need an integer for this
    
    # getting IDs
    device_cat_id = fcumsum((
      hospitalization_id != flag(hospitalization_id, fill = TRUE) |           # (a)
        device_cat_f  != flag(device_cat_f, fill = TRUE)))) |>       # (b)
  
  relocate(device_cat_id, .after = recorded_hour) |> 
  
  # fill device name
  #         changed some failsafes above 4/2024 so its ok to do downup with this now
  group_by(hospitalization_id, device_cat_id) |> 
  arrange(hospitalization_id, recorded_dttm) |> 
  fill(device_name, .direction = "downup") |> 
  ungroup() |>
  
  
  # Record a new device_id when either (a) a new encounter, or 
  #                                    (b) preceded by a different device name.
  mutate(
    # need to have NA as something so it gets an ID
    device_name_f = fifelse(is.na(device_name), "missing", device_name), # cant have anything with NAs when factoring
    device_name_f = as.integer(as.factor(device_name_f)), # need an integer for this
    
    # getting IDs
    device_id = fcumsum((
      hospitalization_id    != flag(hospitalization_id, fill = TRUE) |           # (a)
        device_name_f != flag(device_name_f, fill = TRUE)))) |>      # (b)
  
  relocate(device_id, .after = recorded_hour) |> 
  
  # fill mode_category (downup)
  # there are PST that are being carried over to days before when ppl get REINTUBATED
  group_by(hospitalization_id, device_id) |> 
  arrange(hospitalization_id, recorded_dttm) |> 
  fill(mode_category, .direction = "downup") |> 
  ungroup() |> 
  
  # Create mode_id
  mutate(
    mode_cat_f = fifelse(is.na(mode_category), "missing", mode_category), # cant have anything with NAs when factoring
    mode_cat_f = as.integer(as.factor(mode_cat_f)), # need an integer for this
    
    mode_cat_id = fcumsum((
      device_id     != flag(device_id, fill = TRUE) |        # (a)
        mode_cat_f  != flag(mode_cat_f, fill = TRUE)))) |>   # (b)
  
  
  relocate(mode_cat_id, .after = recorded_hour) |> 
  
  # fill mode name (downup) 
  group_by(hospitalization_id, mode_cat_id) |> 
  arrange(hospitalization_id, recorded_dttm) |> 
  fill(mode_name, .direction = "downup") |> 
  ungroup() |>  
  
  # Create mode name id
  mutate(
    mode_name_f = fifelse(is.na(mode_name), "missing", mode_name), # cant have anything with NAs when factoring
    mode_name_f = as.integer(as.factor(mode_name_f)), # need an integer for this
    
    mode_name_id = fcumsum((
      mode_cat_id != flag(mode_cat_id, fill = TRUE) |               # (a)
        mode_name_f != flag(mode_name_f, fill = TRUE)))) |>         # (b)
  
  relocate(mode_name_id, .after = recorded_hour) |> 
  
  
  # changing fio2_set to 0.21 if room air as category
  mutate(fio2_set = if_else(is.na(fio2_set) & device_category == "Room Air", .21, fio2_set)) |> 
  # erroneous set volumes are in places where they shouldn't be for PS and trach_dome
  mutate(
    tidal_volume_set = fifelse(
      (
        mode_category == "Pressure Support/CPAP" &    # needs to be PS/CPAP
          !is.na(pressure_support_set)                    # needs to have a PS level
      ) |
        (
          is.na(mode_category) &                      # mode cat needs to be NA
            str_detect(device_name, "trach")          # only when trach stuff
        ) |
        (
          mode_category == "Pressure Support/CPAP" &  # needs to be PS/CPAP
            str_detect(device_name, "trach")          # only when trach stuff
        ),
      NA_integer_,
      tidal_volume_set),
    
    
  ) |>
  
  # there are ppl with t-piece that should be blow_by
  mutate(mode_category = fifelse(
    (is.na(mode_category) & 
       str_detect(device_name, "t-piece")),
    "Blow by",
    mode_category
  )) |> 
  
  # carry forward the rest
  group_by(hospitalization_id, mode_name_id) |>  # mode_name_id is the most granular, can go up and down
  arrange(hospitalization_id, recorded_dttm) |> 
  
  # took trach out of this so we don't fill back up 3/2024
  fill(c(fio2_set, lpm_set, peep_set, tidal_volume_set, pressure_support_set, resp_rate_set, 
         # tidal_volume_obs, 
         peak_inspiratory_pressure_obs,  
         # minute_vent_obs, 
         hospital_id, location_name, location_category, in_dttm, out_dttm
  ), .direction = "downup"
  ) |>
  
  # making trach the same for everyone
  mutate(tracheostomy = fifelse(tracheostomy == 1, 1, NA)) |> 
  
  # fill trach... only down
  fill(c(tracheostomy), .direction = "down") |>
  ungroup() |> 
  
  
  # need to get rid of duplicates 
  distinct() |> 
  dplyr::select(
    hospitalization_id,
    recorded_dttm,
    recorded_date,
    recorded_hour,
    mode_name_id,
    device_category,
    device_name,
    mode_category,
    mode_name,
    mode_cat_id,
    device_id,
    device_cat_id,
    fio2_set,
    lpm_set,
    peep_set,
    tracheostomy,
    tidal_volume_set,
    pressure_support_set,
    resp_rate_set,
    starts_with("tidal_volume_obs"),
    starts_with("mean_airway_pressure_obs"),
    peak_inspiratory_pressure_obs,
    plateau_pressure_obs,
    # obs_resp_rate,
    # minute_vent_obs,
    hospital_id,
    location_name,
    location_category,
    # in_dttm,
    # out_dttm,
    # device_cat_f,
    # device_name_f,
    # mode_cat_f,
    # mode_name_f,
  ) 

toc()

rm(df_resp_support_1)
rm(hour_sequence)
rm(clif_adt)
rm(clif_respiratory_support)


# Load the device_category to ranges mapping table
category_values <- read_csv("lookup-tables/device_category_to_conversion.csv")

# Ensure device_category is trimmed of whitespace and merge with mapping
df_resp_support <- df_resp_support %>%
  mutate(device_category = str_trim(device_category)) %>%
  left_join(category_values, by = "device_category")

# Check ranges for fio2_set
df_resp_support_conv <- df_resp_support %>% 
  mutate(fio2_set = case_when(
    !is.na(fio2_set) & !is.na(range_lower) & fio2_set < range_lower ~ range_lower,
    !is.na(fio2_set) & !is.na(range_upper) & fio2_set > range_upper ~ range_upper,
    TRUE ~ fio2_set))

# Impute FiO2 values when fio2_set is NA and lpm_set is not NA
df_resp_support_conv <- df_resp_support_conv %>%
  mutate(fio2_set = case_when(
    is.na(fio2_set) & !is.na(lpm_set) & !is.na(conversion) ~ {
      fio2_imp <- 0.21 + lpm_set * conversion
      pmin(pmax(fio2_imp, range_lower), range_upper)
    },
    TRUE ~ fio2_set))

# Rename fio2_set to fio2_approx
df_resp_support_conv <- df_resp_support_conv %>% 
  rename(fio2_approx = fio2_set)

summary(df_resp_support$fio2_set)
summary(df_resp_support_conv$fio2_approx)

# If there are still NA values in fio2_approx, set them to range_lower if available
df_resp_support_conv <- df_resp_support_conv %>%
  mutate(fio2_approx = ifelse(is.na(fio2_approx) & !is.na(range_lower), range_lower, fio2_approx)) %>% 
  # Ensure fio2_approx is 0.21 for Room Air, as lookup table doesn't provide range_lower
  mutate(fio2_approx = ifelse(device_category == "Room Air", 0.21, fio2_approx)) 

summary(df_resp_support_conv$fio2_approx)

# Define the output directory
intermediate_output_path <- file.path(output_path, "intermediate")

# Create the directory if it doesn't exist
dir.create(intermediate_output_path, showWarnings = FALSE, recursive = TRUE)

# Save the processed data
write_parquet(df_resp_support_conv, file.path(intermediate_output_path, paste0("clif_respiratory_support_processed", file_type)))
print(paste("Table exported as parquet to", intermediate_output_path))
