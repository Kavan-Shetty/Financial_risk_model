print("\nStarting Financial Risk Model Pipeline...")

import os
import numpy as np


# Phase 1
from src.phase1_data.data_loader import load_data
from src.phase1_data.data_validation import validate_data
from src.phase1_data.temporal_split import temporal_split
from src.phase1_data.prepare_dataset import split_features_target


# Phase 2
from src.phase2_features.feature_engineering import engineer_features
from src.phase2_features.feature_validator import validate_features


# Phase 3
from src.phase3_model.imbalance_handler import compute_class_weight
from src.phase3_model.train_model import train_model
from src.phase3_model.predict import predict_probabilities


# Phase 4
from src.phase4_evaluation.evaluate_model import evaluate_model


# Phase 5
from src.phase5_scoring.credit_score import (
    generate_credit_scores,
    assign_risk_category,
    assign_fraud_label
)


# Phase 6
from src.phase6_explainability.shap_explainer import (
    generate_shap_explanations,
    get_feature_importance
)


# Phase 7
from src.phase7_deployment.model_saver import save_model
from src.phase7_deployment.model_loader import load_model


# Phase 8
from src.phase8_probability.probability_calibrator import (
    calibrate_model,
    get_calibrated_probabilities
)

from src.phase8_probability.threshold_selector import (
    find_optimal_threshold,
    classify_with_threshold
)


# ========================================
# Create directories
# ========================================

os.makedirs("models", exist_ok=True)


# ========================================
# PHASE 1: DATA
# ========================================

print("\nPHASE 1: DATA PREPARATION")

df = load_data("data/creditcard.csv")

df = validate_data(df)

train_df, test_df = temporal_split(df)

X_train, y_train = split_features_target(train_df)

X_test, y_test = split_features_target(test_df)


# ========================================
# PHASE 2: FEATURES
# ========================================

print("\nPHASE 2: FEATURE ENGINEERING")

X_train = engineer_features(X_train)

X_test = engineer_features(X_test)

validate_features(X_train)
validate_features(X_test)


# ========================================
# PHASE 3: TRAIN
# ========================================

print("\nPHASE 3: MODEL TRAINING")

class_weight = compute_class_weight(y_train)

model = train_model(
    X_train,
    y_train,
    class_weight
)


# Raw probabilities
raw_probabilities = predict_probabilities(
    model,
    X_test
)


# ========================================
# PHASE 4: EVALUATION
# ========================================

print("\nPHASE 4: MODEL EVALUATION")

roc_auc, pr_auc, ks = evaluate_model(
    y_test,
    raw_probabilities
)


# ========================================
# PHASE 6: SHAP (USE RAW MODEL)
# ========================================

print("\nPHASE 6: EXPLAINABILITY")

explainer, shap_values, X_sample = generate_shap_explanations(
    model,
    X_train,
    X_test,
    sample_size=1000
)

importance_df = get_feature_importance(
    shap_values,
    X_sample
)

print("\nTop Features:")
print(importance_df.head(10))


# ========================================
# PHASE 7: SAVE MODEL
# ========================================

print("\nPHASE 7: SAVING MODEL")

save_model(
    model,
    X_train.columns.tolist()
)

print("Model saved successfully")


# ========================================
# PHASE 8: CALIBRATION
# ========================================

print("\nPHASE 8: PROBABILITY CALIBRATION")

calibrated_model = calibrate_model(
    model,
    X_train,
    y_train
)

calibrated_probabilities = get_calibrated_probabilities(
    calibrated_model,
    X_test
)


threshold = find_optimal_threshold(
    y_test,
    calibrated_probabilities
)

predictions = classify_with_threshold(
    calibrated_probabilities,
    threshold
)


# ========================================
# PHASE 5: CREDIT SCORING
# ========================================

print("\nPHASE 5: CREDIT SCORING")

credit_scores = generate_credit_scores(
    calibrated_probabilities
)

risk_categories = assign_risk_category(
    credit_scores
)

fraud_labels = assign_fraud_label(
    calibrated_probabilities,
    threshold
)


# ========================================
# FINAL OUTPUT
# ========================================

print("\nPIPELINE COMPLETED")

print("\nMetrics:")
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
print("KS:", ks)

print("\nProbability range:")
print("Min:", calibrated_probabilities.min())
print("Max:", calibrated_probabilities.max())

print("\nScore range:")
print("Min:", credit_scores.min())
print("Max:", credit_scores.max())