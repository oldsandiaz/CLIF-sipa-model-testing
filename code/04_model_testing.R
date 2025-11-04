# Load data 
library(arrow)
library(ggplot2)
library(caret)
library(pROC)
library(tictoc)
library(mgcv)
library(lightgbm)
library(tidyverse)
library(glmnet)
library(wesanderson)

# Clear environment
rm(list = ls())

source("utils/config.R")
output_path <- config$output_path


set.seed(42) # the meaning of life

# Load testing data
data <- read_parquet(paste0(output_path, "/sipa_features.parquet"))

sofa_age_all <- data %>%
   select(p_f_pre, s_f_pre, platelets_pre, bilirubin_pre, map_pre, gcs_pre, creatinine_pre, norepinephrine_eq_pre, p_f_post, s_f_post, platelets_post, bilirubin_post, map_post, gcs_post, creatinine_post, norepinephrine_eq_post, age_at_admission) %>%
    mutate(across(everything(), ~replace_na(.x, 0)))

# Load all models contained in the models directory
model_files <- list.files("/Users/cdiaz/Desktop/SRP/clif-sipa-model-testing/models", full.names = TRUE)

# Models will either be .txt or .rds files. Load based on file extension.

# Function to load models based on file extension
load_model <- function(file_path) {
  ext <- tools::file_ext(file_path)
  if (ext == "txt") {
    return(lightgbm::lgb.load(file_path))
  } else if (ext == "rds") {
    return(readRDS(file_path))
  } else {
    warning(paste("Unknown extension for model file:", file_path))
    return(NULL)
  }
}

# Load all models into a named list
models <- lapply(model_files, load_model)
names(models) <- basename(model_files)

# Helper function to run prediction based on model type
predict_model <- function(model, feature_matrix) {
  # LightGBM
  if (inherits(model, "lgb.Booster")) {
    return(as.numeric(predict(model, newdata = as.matrix(feature_matrix))))
  }
  # GLM (logistic regression, etc.)
  if (inherits(model, "glm")) {
    return(as.numeric(predict(model, newdata = feature_matrix, type = "response")))
  }
  # GAM (mgcv package)
  if (inherits(model, "gam")) {
    return(as.numeric(predict(model, newdata = feature_matrix, type = "response")))
  }
  # GLMnet (lasso/ridge regression)
  if (inherits(model, "elnet")) {
    # Use lambda that was used in training (usually model$lambda.min or model$lambda.1se)
    # If not present, default to lambda=0.01
    lambda_val <- if (!is.null(model$lambda.min)) model$lambda.min else 0.01
    pred <- predict(model, newx = as.matrix(feature_matrix), type = "response", s = lambda_val)
    return(as.numeric(pred))
  }
  # Try generic prediction for other model types
  return(as.numeric(predict(model, newdata = feature_matrix, type = "response")))
}

# Run predictions for all models
pred_matrix <- sapply(models, predict_model, feature_matrix = sofa_age_all)

# SIMPLE ENSEMBLE: AVERAGE ACROSS ALL PREDICTIONS
ensemble_pred <- rowMeans(pred_matrix)

# Calculate AUC for the ensemble model
auc_ensemble <- roc(data$in_hospital_mortality, ensemble_pred)$auc

cat(sprintf("Ensemble AUC: %.4f\n", auc_ensemble))

# Calibration plot for ensemble

# Ensure race_group is defined
data$race_group <- ifelse(data$race == "Black or African American", "Black", "Non-Black")

# Ensemble predictions (assuming pred_matrix is already calculated)
ensemble_pred <- rowMeans(pred_matrix)

# Output variable for calibration (must be factor)
output <- factor(ifelse(data$in_hospital_mortality == 1, "Dead", "Alive"), levels = c("Alive", "Dead"))

# Calibration plot for ensemble, stratified by race
cal_data_ensemble <- data.frame(
  prob = ensemble_pred,
  y = output,
  race = data$race_group
)

cal_obj_black <- caret::calibration(y ~ prob, data = subset(cal_data_ensemble, race == "Black"), class = "Dead")
cal_obj_non_black <- caret::calibration(y ~ prob, data = subset(cal_data_ensemble, race == "Non-Black"), class = "Dead")

plot_data_ensemble <- rbind(
  data.frame(cal_obj_black$data, race = "Black Patients"),
  data.frame(cal_obj_non_black$data, race = "Non-Black Patients")
)

p_ensemble <- ggplot(plot_data_ensemble, aes(x = midpoint, y = Percent, color = race)) +
  geom_line() +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Calibration Plot for Ensemble Model",
    x = "Predicted Probability",
    y = "Observed Frequency"
  ) +
  theme_classic() +
  scale_color_manual(values = wesanderson::wes_palette("GrandBudapest2", n = 2))

ggsave(p_ensemble, filename = paste0(output_path, "/calibration_ensemble.png"), width = 6, height = 4)

# Calculate allocation efficiency for ensemble model vs random allocation

calculate_efficiency <- function(predicted_mortality, true_mortality, s_grid = seq(0, 1, by = 0.01)) {
  N <- length(true_mortality)
  survivors <- 1 - true_mortality
  res <- purrr::map_dfr(s_grid, function(s) {
    V <- floor(s * N)
    if (V == 0) {
      eff <- NA_real_
    } else {
      selected_idx <- order(predicted_mortality)[1:V]
      survivors_allocated <- sum(survivors[selected_idx])
      eff <- survivors_allocated / V
    }
    tibble(s = s, efficiency = eff)
  })
  res
}

# The true mortality vector
true_mortality_vector <- data$in_hospital_mortality

# Ensemble prediction (already calculated)
ensemble_pred <- rowMeans(pred_matrix)

# Efficiency for ensemble
eff_ensemble <- calculate_efficiency(ensemble_pred, true_mortality_vector)

# Efficiency for random allocation
eff_random <- calculate_efficiency(runif(length(true_mortality_vector)), true_mortality_vector)

# Combine results for plotting
eff_all <- bind_rows(
  eff_random %>% mutate(model = "Random Allocation"),
  eff_ensemble %>% mutate(model = "Ensemble Model")
)

# Plot
allocation_plot <- ggplot(eff_all, aes(x = s, y = efficiency, color = model)) +
  geom_line(size = 1.2, na.rm = TRUE) +
  scale_x_continuous(name = "Life support supply, s", breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(name = "Allocation Efficiency", limits = c(0, 1)) +
  ggtitle("Allocation Efficiency: Ensemble vs Random") +
  theme_classic() +
  theme(text = element_text(size = 14)) +
  scale_color_manual(values = wesanderson::wes_palette("GrandBudapest2", n = 2))

# Save plot
saveRDS(allocation_plot, file.path(output_path, "/ensemble_vs_random_allocation_efficiency_plot.rds"))

print("Finished Ensemble vs Random Allocation Efficiency Analysis.")

