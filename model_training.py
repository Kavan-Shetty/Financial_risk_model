"""
src/model_training.py
=====================
Trains and tunes multiple classifiers for financial risk prediction.

Models included
---------------
1. Logistic Regression  — interpretable baseline; coefficients are directly
   mappable to credit scorecard points (industry standard for regulatory
   submissions / model risk governance).

2. Random Forest        — ensemble of decision trees; handles non-linear
   interactions and is robust to noisy financial data.  Feature importances
   are straightforward to communicate to risk committees.

3. Gradient Boosting    — sequential ensemble; typically achieves the best
   AUC on tabular financial data.  XGBoost / LightGBM variants are included.

Class imbalance handling
------------------------
Credit default datasets are inherently imbalanced (e.g. 5–20% defaults).
Two complementary strategies are applied:
  * class_weight='balanced' inside sklearn estimators.
  * Optional SMOTE over-sampling (imbalanced-learn) on the training set only.

Hyperparameter tuning
---------------------
GridSearchCV / RandomizedSearchCV optimises AUC-ROC (preferred over accuracy
for imbalanced classes) using stratified k-fold cross-validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────────────────────────────────────

def get_model_catalogue(cfg: dict) -> Dict[str, Any]:
    """
    Return a dict of {model_name: unfitted_estimator} for each enabled model.

    Parameters
    ----------
    cfg : Full project configuration dict (from config.yaml).

    Returns
    -------
    Dict mapping model names to sklearn-compatible estimators.
    """
    models_cfg = cfg["models"]
    rs         = cfg["project"]["random_state"]
    catalogue  = {}

    if models_cfg["logistic_regression"]["enabled"]:
        params = models_cfg["logistic_regression"]["params"]
        catalogue["LogisticRegression"] = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            class_weight=params["class_weight"],
            random_state=rs,
        )

    if models_cfg["random_forest"]["enabled"]:
        params = models_cfg["random_forest"]["params"]
        catalogue["RandomForest"] = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            n_jobs=params["n_jobs"],
            random_state=rs,
        )

    if models_cfg["gradient_boosting"]["enabled"]:
        params = models_cfg["gradient_boosting"]["params"]
        catalogue["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            min_samples_split=params["min_samples_split"],
            random_state=rs,
        )

    logger.info("Model catalogue: %s", list(catalogue.keys()))
    return catalogue


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation baseline
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run stratified k-fold CV on every model in *models*.

    Returns
    -------
    Dict mapping model name → {mean_score, std_score}.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    for name, estimator in models.items():
        logger.info("Cross-validating %s …", name)
        scores = cross_val_score(
            estimator, X_train, y_train,
            cv=skf, scoring=scoring, n_jobs=-1,
        )
        results[name] = {
            "mean": float(scores.mean()),
            "std":  float(scores.std()),
        }
        logger.info(
            "  %s — %s: %.4f ± %.4f",
            name, scoring, scores.mean(), scores.std(),
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    config_path: str = "config.yaml",
) -> RandomForestClassifier:
    """
    Grid-search over Random Forest hyperparameters.

    The search space is defined in config.yaml under
    tuning.random_forest_grid.  GridSearchCV uses StratifiedKFold to
    maintain class proportions in each fold.

    Returns
    -------
    Best fitted RandomForestClassifier.
    """
    cfg      = _load_config(config_path)
    tune_cfg = cfg["tuning"]
    rs       = cfg["project"]["random_state"]

    param_grid = tune_cfg.get("random_forest_grid", {
        "n_estimators":    [100, 200],
        "max_depth":       [5, 10, None],
        "min_samples_split": [2, 5],
    })

    skf = StratifiedKFold(
        n_splits=tune_cfg["cv_folds"], shuffle=True, random_state=rs,
    )
    base_model = RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=rs,
    )

    if tune_cfg["method"] == "random_search":
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=30,
            cv=skf,
            scoring=tune_cfg["scoring"],
            n_jobs=tune_cfg["n_jobs"],
            random_state=rs,
            verbose=1,
        )
    else:
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=skf,
            scoring=tune_cfg["scoring"],
            n_jobs=tune_cfg["n_jobs"],
            verbose=1,
        )

    logger.info("Tuning Random Forest (%s) …", tune_cfg["method"])
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s", search.best_params_)
    logger.info("Best RF CV %s: %.4f", tune_cfg["scoring"], search.best_score_)
    return search.best_estimator_


def tune_gradient_boosting(
    X_train: np.ndarray,
    y_train: pd.Series,
    config_path: str = "config.yaml",
) -> GradientBoostingClassifier:
    """
    Grid-search over Gradient Boosting hyperparameters.

    Returns
    -------
    Best fitted GradientBoostingClassifier.
    """
    cfg      = _load_config(config_path)
    tune_cfg = cfg["tuning"]
    rs       = cfg["project"]["random_state"]

    param_grid = tune_cfg.get("gradient_boosting_grid", {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    })

    skf = StratifiedKFold(
        n_splits=tune_cfg["cv_folds"], shuffle=True, random_state=rs,
    )
    base_model = GradientBoostingClassifier(random_state=rs)

    if tune_cfg["method"] == "random_search":
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=skf,
            scoring=tune_cfg["scoring"],
            n_jobs=tune_cfg["n_jobs"],
            random_state=rs,
            verbose=1,
        )
    else:
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=skf,
            scoring=tune_cfg["scoring"],
            n_jobs=tune_cfg["n_jobs"],
            verbose=1,
        )

    logger.info("Tuning Gradient Boosting (%s) …", tune_cfg["method"])
    search.fit(X_train, y_train)
    logger.info("Best GB params: %s", search.best_params_)
    logger.info("Best GB CV %s: %.4f", tune_cfg["scoring"], search.best_score_)
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# Train + select best model
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    config_path: str = "config.yaml",
    tune: bool = True,
) -> Dict[str, Any]:
    """
    Train (and optionally tune) all configured models.

    Parameters
    ----------
    X_train     : Preprocessed training feature matrix.
    y_train     : Binary target series.
    config_path : Path to config.yaml.
    tune        : If True, run hyperparameter search for RF and GB.

    Returns
    -------
    Dict mapping model name → fitted estimator.
    """
    cfg      = _load_config(config_path)
    rs       = cfg["project"]["random_state"]
    catalogue = get_model_catalogue(cfg)

    # Always fit Logistic Regression as-is (baseline)
    if "LogisticRegression" in catalogue:
        logger.info("Training LogisticRegression …")
        catalogue["LogisticRegression"].fit(X_train, y_train)

    if tune:
        if "RandomForest" in catalogue:
            catalogue["RandomForest"] = tune_random_forest(
                X_train, y_train, config_path=config_path,
            )
        if "GradientBoosting" in catalogue:
            catalogue["GradientBoosting"] = tune_gradient_boosting(
                X_train, y_train, config_path=config_path,
            )
    else:
        for name in ["RandomForest", "GradientBoosting"]:
            if name in catalogue:
                logger.info("Training %s (no tuning) …", name)
                catalogue[name].fit(X_train, y_train)

    logger.info("All models trained: %s", list(catalogue.keys()))
    return catalogue


def select_best_model(
    trained_models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: pd.Series,
    scoring: str = "roc_auc",
) -> Tuple[str, Any]:
    """
    Evaluate each trained model on the validation set and return the best one.

    AUC-ROC is used as the primary criterion because it is insensitive to
    the classification threshold and handles class imbalance gracefully.

    Returns
    -------
    (best_model_name, best_fitted_estimator)
    """
    from sklearn.metrics import roc_auc_score

    best_name  = None
    best_score = -np.inf
    best_model = None

    for name, estimator in trained_models.items():
        proba = estimator.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        logger.info("Validation %s — %s: %.4f", name, scoring, score)
        if score > best_score:
            best_score = score
            best_name  = name
            best_model = estimator

    logger.info("Best model: %s (val AUC=%.4f)", best_name, best_score)
    return best_name, best_model


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    estimator: Any,
    model_name: str,
    model_dir: str = "models",
    filename: Optional[str] = None,
) -> str:
    """
    Serialise a fitted estimator with joblib.

    Returns
    -------
    Full path to the saved file.
    """
    os.makedirs(model_dir, exist_ok=True)
    fname = filename or f"{model_name.lower().replace(' ', '_')}.pkl"
    path  = os.path.join(model_dir, fname)
    joblib.dump(estimator, path)
    logger.info("Model '%s' saved → %s", model_name, path)
    return path


def load_model(path: str) -> Any:
    """Deserialise a previously saved estimator."""
    estimator = joblib.load(path)
    logger.info("Model loaded from '%s'.", path)
    return estimator


def save_cv_results(
    cv_results: Dict[str, Dict[str, float]],
    path: str = "models/cv_results.json",
) -> None:
    """Persist cross-validation scores for audit trail."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cv_results, fh, indent=2)
    logger.info("CV results saved → %s", path)
