"""
src/model_evaluation.py
========================
Comprehensive evaluation of trained financial-risk classifiers.

Metrics reported
----------------
Standard classification:
  Accuracy, Precision, Recall, F1 (macro + weighted)

Ranking / probabilistic:
  ROC-AUC         — primary model-selection criterion
  Average Precision (PR-AUC) — critical for highly imbalanced datasets
                   where PR curves are more informative than ROC curves

Visualisations
--------------
  * Confusion matrix (absolute counts + normalised)
  * ROC curves (all models on one plot)
  * Precision-Recall curves
  * Feature importance bar chart (tree-based models)
  * Calibration curve (reliability diagram)

Regulatory / model-risk context
--------------------------------
Banks and financial institutions are required by Basel II / IFRS 9 to:
  1. Demonstrate model discriminative power (ROC-AUC ≥ 0.70 is a common floor).
  2. Validate stability via Gini coefficient (= 2 × AUC − 1).
  3. Document feature importances for explainability / model-risk governance.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Use a clean, publication-quality style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute the full suite of evaluation metrics.

    Parameters
    ----------
    y_true     : Ground-truth binary labels.
    y_pred     : Hard class predictions (already thresholded).
    y_prob     : Predicted probability for the positive class.
    threshold  : Classification threshold (default 0.5).
    model_name : Display name used in logging.

    Returns
    -------
    Dict of metric_name → value.
    """
    # Re-threshold if desired threshold differs from 0.5
    if threshold != 0.50:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model":             model_name,
        "threshold":         threshold,
        "accuracy":          accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "avg_precision":     average_precision_score(y_true, y_prob),
        "gini":              2 * roc_auc_score(y_true, y_prob) - 1,  # Gini = 2*AUC − 1
    }

    logger.info(
        "\n[%s] Test Metrics\n"
        "  Accuracy   : %.4f\n"
        "  Precision  : %.4f\n"
        "  Recall     : %.4f\n"
        "  F1         : %.4f\n"
        "  ROC-AUC    : %.4f\n"
        "  Gini       : %.4f\n"
        "  PR-AUC     : %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["gini"],
        metrics["avg_precision"],
    )

    return metrics


def full_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> str:
    """Return sklearn's classification_report as a formatted string."""
    report = classification_report(
        y_true, y_pred,
        target_names=["Low Risk (0)", "High Risk (1)"],
        zero_division=0,
    )
    logger.info("\n%s — Classification Report:\n%s", model_name, report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot the confusion matrix with absolute counts and (optionally) fractions.

    Financial interpretation:
      * False Negatives (FN): defaulters classified as low-risk → credit losses.
      * False Positives (FP): safe borrowers rejected → opportunity cost / fairness.
    The trade-off between FN and FP is the core business decision for threshold setting.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved → %s", save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    models_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on a single axes.

    Parameters
    ----------
    models_probs : Dict mapping model_name → predicted_proba (positive class).
    y_true       : Ground-truth labels.
    save_path    : If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, y_prob) in enumerate(models_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc          = roc_auc_score(y_true, y_prob)
        color        = PALETTE[idx % len(PALETTE)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC=0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Financial Risk Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved → %s", save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Precision-Recall curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curves(
    models_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves.

    PR curves are more informative than ROC when the positive class (defaults)
    is rare, as they focus on the performance among predicted positives.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_true.mean()

    for idx, (name, y_prob) in enumerate(models_probs.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap    = average_precision_score(y_true, y_prob)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(recall, precision, color=color, lw=2, label=f"{name}  (AP={ap:.4f})")

    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"No-skill baseline ({baseline:.2%})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Financial Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve saved → %s", save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    estimator: Any,
    feature_names: List[str],
    top_n: int = 20,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Bar chart of the top-N most important features.

    Works for:
      * RandomForest / GradientBoosting (feature_importances_ attribute).
      * LogisticRegression (absolute coefficient values as importance proxy).

    Financial context
    -----------------
    Feature importances are required for:
      * Model explainability / SHAP values (next step).
      * Regulatory compliance (SR 11-7 model risk management guidance).
      * Communicating model drivers to credit risk committees.
    """
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        importance_type = "Gini Importance"
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
        importance_type = "|Coefficient|"
    else:
        logger.warning("Estimator has no feature_importances_ or coef_. Skipping plot.")
        return None

    # Sort and take top N
    indices  = np.argsort(importances)[::-1][:top_n]
    top_imp  = importances[indices]
    top_feat = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(
        range(len(top_feat)), top_imp[::-1],
        color=PALETTE[0], alpha=0.85, edgecolor="white",
    )
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat[::-1], fontsize=10)
    ax.set_xlabel(importance_type, fontsize=12)
    ax.set_title(
        f"{model_name} — Top {top_n} Feature Importances",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved → %s", save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(
    models_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Reliability diagram (calibration curve).

    A well-calibrated model's predicted probability of 0.7 should correspond
    to a true positive rate of ~70 % in that probability bucket.
    Calibration is critical when the model score is used as a direct
    probability of default (PD) for IFRS 9 / Basel expected-loss calculations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")

    for idx, (name, y_prob) in enumerate(models_probs.items()):
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform",
        )
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(mean_pred, fraction_pos, "s-", color=color, lw=2, label=name)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Actual Default Rate)", fontsize=12)
    ax.set_title("Calibration Curves — Probability of Default", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration curve saved → %s", save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    plots_dir: str = "reports/figures",
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Run end-to-end evaluation for all models, save plots, return metrics DataFrame.

    Parameters
    ----------
    trained_models : Dict of {name: fitted_estimator}.
    X_test         : Test feature matrix (preprocessed).
    y_test         : Test labels.
    feature_names  : Output feature names from the preprocessor.
    plots_dir      : Directory to save visualisation files.
    threshold      : Classification threshold.

    Returns
    -------
    pd.DataFrame with one row per model and all metric columns.
    """
    os.makedirs(plots_dir, exist_ok=True)
    all_metrics  = []
    models_probs = {}

    for name, estimator in trained_models.items():
        y_prob = estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        models_probs[name] = y_prob

        metrics = compute_metrics(y_test, y_pred, y_prob, threshold, model_name=name)
        all_metrics.append(metrics)
        full_classification_report(y_test, y_pred, model_name=name)

        # Per-model confusion matrix
        plot_confusion_matrix(
            y_test, y_pred,
            model_name=name,
            save_path=os.path.join(plots_dir, f"cm_{name.lower()}.png"),
        )
        plt.close("all")

        # Feature importance
        plot_feature_importance(
            estimator, feature_names, top_n=20, model_name=name,
            save_path=os.path.join(plots_dir, f"feature_importance_{name.lower()}.png"),
        )
        plt.close("all")

    # Shared plots across all models
    plot_roc_curves(
        models_probs, y_test,
        save_path=os.path.join(plots_dir, "roc_curves.png"),
    )
    plt.close("all")

    plot_precision_recall_curves(
        models_probs, y_test,
        save_path=os.path.join(plots_dir, "pr_curves.png"),
    )
    plt.close("all")

    plot_calibration_curve(
        models_probs, y_test,
        save_path=os.path.join(plots_dir, "calibration_curves.png"),
    )
    plt.close("all")

    results_df = pd.DataFrame(all_metrics).set_index("model")
    logger.info("\nFinal Model Comparison:\n%s", results_df.to_string())
    return results_df
