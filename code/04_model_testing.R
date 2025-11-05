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

sofa_score_only <- data %>%
  select(sofa_score_pre, sofa_score_post) %>%
  mutate(across(everything(), ~replace_na(.x, 0))) %>%
  rowwise() %>%
  mutate(worst_sofa_score = max(sofa_score_pre, sofa_score_post)) %>%
  select(worst_sofa_score)

# Load all models contained in the models directory
model_files <- list.files("/Users/cdiaz/Desktop/SRP/clif-sipa-model-testing/models", full.names = TRUE)

# Models will either be .txt or .rds files. Load based on file extension.
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
all_models <- lapply(model_files, load_model)
names(all_models) <- basename(model_files)

# Separate models
sofa_glm_models <- all_models[grep("glm_sofa_score_.*.rds", names(all_models))]
ensemble_models <- all_models[!names(all_models) %in% names(sofa_glm_models)]


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
  # Caret train object
  if (inherits(model, "train")) {
    return(as.numeric(predict(model, newdata = feature_matrix, type = "prob")[,2])) # Assuming binary classification and taking probability of the second class
  }
  # Try generic prediction for other model types
  return(as.numeric(predict(model, newdata = feature_matrix, type = "response")))
}

# Run predictions for SOFA models
sofa_pred_matrix <- sapply(sofa_glm_models, predict_model, feature_matrix = sofa_score_only)
sofa_glm_pred <- rowMeans(sofa_pred_matrix)

# Run predictions for Ensemble models
ensemble_pred_matrix <- sapply(ensemble_models, predict_model, feature_matrix = sofa_age_all)
ensemble_pred <- rowMeans(ensemble_pred_matrix)


# Calculate AUC for both models
roc_sofa <- roc(data$in_hospital_mortality, sofa_glm_pred, ci = TRUE)
roc_ensemble <- roc(data$in_hospital_mortality, ensemble_pred, ci = TRUE)

# Create a data frame for plotting
roc_sofa_data <- data.frame(
  fpr = 1 - roc_sofa$specificities,
  tpr = roc_sofa$sensitivities,
  model = "SOFA Score Only"
)

roc_ensemble_data <- data.frame(
  fpr = 1 - roc_ensemble$specificities,
  tpr = roc_ensemble$sensitivities,
  model = "Ensemble Model"
)

roc_data <- rbind(roc_sofa_data, roc_ensemble_data)

# Calculate confidence intervals for the ROC curves
ci_sofa <- ci.sp(roc_sofa, sensitivities=seq(0, 1, .01), boot.n=100)
ci_ensemble <- ci.sp(roc_ensemble, sensitivities=seq(0, 1, .01), boot.n=100)

ci_sofa_data <- data.frame(
  tpr = as.numeric(rownames(ci_sofa)),
  fpr_low = 1 - ci_sofa[, 3],
  fpr_high = 1 - ci_sofa[, 1],
  model = "SOFA Score Only"
)

ci_ensemble_data <- data.frame(
  tpr = as.numeric(rownames(ci_ensemble)),
  fpr_low = 1 - ci_ensemble[, 3],
  fpr_high = 1 - ci_ensemble[, 1],
  model = "Ensemble Model"
)

ci_data <- rbind(ci_sofa_data, ci_ensemble_data)

# Get AUC values
auc_sofa <- format(roc_sofa$auc, digits = 2)
auc_ensemble <- format(roc_ensemble$auc, digits = 2)

# Create legend labels
legend_labels <- c(
  `SOFA Score Only` = paste0("SOFA Score Only (AUC = ", auc_sofa, ")"),
  `Ensemble Model` = paste0("Ensemble Model (AUC = ", auc_ensemble, ")")
)

# Plot AUROC
p_auroc <- ggplot() +
  geom_ribbon(data = ci_data, aes(y = tpr, xmin = fpr_low, xmax = fpr_high, fill = model), alpha = 0.2) +
  geom_line(data = roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_abline(linetype = "dashed") +
  labs(
    title = "Comparing AUROC: Ensemble Model vs SOFA Score Only",
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "Model",
    fill = "Model"
  ) +
  theme_classic() +
  scale_color_manual(values = wesanderson::wes_palette("GrandBudapest2", n = 2), labels = legend_labels) +
  scale_fill_manual(values = wesanderson::wes_palette("GrandBudapest2", n = 2), guide = "none") +
  coord_fixed(ratio = 1) +
  theme(legend.position = "bottom")

ggsave(p_auroc, filename = paste0(output_path, "/auroc_comparison.png"), width = 8, height = 8)

p_auroc

print("Finished AUC Comparison Analysis.")
