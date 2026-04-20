"""
src/preprocessing.py
====================
Full preprocessing pipeline:
  1. Missing-value imputation (numeric → median, categorical → mode).
  2. Outlier detection and capping using IQR or Z-score.
  3. One-hot encoding of categorical features.
  4. Numeric scaling (StandardScaler / MinMaxScaler / RobustScaler).
  5. Building a single sklearn Pipeline + ColumnTransformer that can be
     fitted on training data and consistently applied to val / test / new data.

Financial context
-----------------
In credit-risk modelling the training set is the historical loan book.
Any transformation (e.g., mean income for imputation) must be derived
**only** from training data to avoid data leakage — a common error that
causes over-optimistic evaluation metrics.  The Pipeline object enforces
this discipline automatically: fit on train, transform on all splits.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Outlier handling (applied before fitting the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def cap_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: List[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Cap outliers to [Q1 - k*IQR, Q3 + k*IQR] (Winsorisation).

    IQR-based capping is preferred over removal in lending data because
    extreme incomes / loan amounts are *real* — they just need bounding so
    they don't dominate tree splits or gradient updates.
    """
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        before_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if before_outliers > 0:
            logger.debug(
                "Column '%s': capped %d outliers to [%.2f, %.2f]",
                col, before_outliers, lower, upper,
            )
    return df


def cap_outliers_zscore(
    df: pd.DataFrame,
    numeric_cols: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Cap values beyond *threshold* standard deviations from the column mean.
    More aggressive than IQR for Gaussian-ish distributions.
    """
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mean = df[col].mean()
        std  = df[col].std()
        if std == 0:
            continue
        lower = mean - threshold * std
        upper = mean + threshold * std
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scaler factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_scaler(strategy: str):
    """Return the appropriate sklearn scaler for the configured strategy."""
    strategy = strategy.lower()
    if strategy == "standard":
        return StandardScaler()
    elif strategy == "minmax":
        return MinMaxScaler()
    elif strategy == "robust":
        # RobustScaler uses median / IQR → less sensitive to remaining outliers
        return RobustScaler()
    else:
        raise ValueError(
            f"Unknown scaling_strategy '{strategy}'. "
            "Choose 'standard', 'minmax', or 'robust'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_strategy: str = "standard",
    imputation_strategy: str = "median",
) -> ColumnTransformer:
    """
    Construct a sklearn ColumnTransformer that handles:
      * Numeric features: impute → scale.
      * Categorical features: impute with mode → one-hot encode.

    Parameters
    ----------
    numeric_features     : List of numeric column names.
    categorical_features : List of categorical column names.
    scaling_strategy     : One of 'standard', 'minmax', 'robust'.
    imputation_strategy  : sklearn SimpleImputer strategy for numeric cols.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
        ("scaler",  _get_scaler(scaling_strategy)),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",    # unseen categories → all zeros
                sparse_output=False,        # return dense array
                drop="first",              # avoid multicollinearity
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,         numeric_features),
            ("cat", categorical_pipeline,     categorical_features),
        ],
        remainder="drop",       # silently drop unrecognised columns (e.g. CustomerID)
        verbose_feature_names_out=False,
    )

    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Fit / transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_preprocessor(
    X_train: pd.DataFrame,
    config_path: str = "config.yaml",
    save_path: Optional[str] = None,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Fit the preprocessor on training data only.

    Parameters
    ----------
    X_train     : Training feature matrix (before target is removed).
    config_path : Path to config.yaml.
    save_path   : If provided, serialise the fitted preprocessor here.

    Returns
    -------
    (fitted_preprocessor, output_feature_names)
    """
    cfg = _load_config(config_path)
    prep_cfg = cfg["preprocessing"]
    feat_cfg = cfg["features"]

    numeric_cols     = [c for c in feat_cfg["numeric"]     if c in X_train.columns]
    categorical_cols = [c for c in feat_cfg["categorical"] if c in X_train.columns]

    # Engineered features: ratio features are numeric, bin features are categorical strings
    import pandas as _pd
    eng_numeric     = []
    eng_categorical = []
    for c in feat_cfg.get("engineered", []):
        if c not in X_train.columns:
            continue
        if _pd.api.types.is_numeric_dtype(X_train[c]):
            eng_numeric.append(c)
        else:
            eng_categorical.append(c)

    all_numeric      = numeric_cols + eng_numeric
    categorical_cols = categorical_cols + eng_categorical

    # Outlier capping on the training set *before* fitting scalers
    if prep_cfg.get("handle_outliers", True):
        method = prep_cfg.get("outlier_method", "iqr")
        if method == "iqr":
            X_train = cap_outliers_iqr(X_train, all_numeric)
        else:
            threshold = prep_cfg.get("outlier_threshold", 3.0)
            X_train   = cap_outliers_zscore(X_train, all_numeric, threshold)

    preprocessor = build_preprocessor(
        numeric_features     = all_numeric,
        categorical_features = categorical_cols,
        scaling_strategy     = prep_cfg.get("scaling_strategy", "standard"),
        imputation_strategy  = prep_cfg.get("imputation_strategy", "median"),
    )

    preprocessor.fit(X_train)
    logger.info("Preprocessor fitted on %d training samples.", len(X_train))

    # Retrieve output feature names
    feature_names = list(preprocessor.get_feature_names_out())
    logger.info("Total features after preprocessing: %d", len(feature_names))

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        joblib.dump(preprocessor, save_path)
        logger.info("Preprocessor saved → %s", save_path)

    return preprocessor, feature_names


def transform(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    config_path: str = "config.yaml",
) -> np.ndarray:
    """
    Apply a *fitted* preprocessor to any feature matrix.

    Outlier capping is intentionally NOT re-fitted here — the caps derived
    from training data are applied to validation / test / new data to prevent
    leakage.

    Parameters
    ----------
    preprocessor : Fitted ColumnTransformer.
    X            : Feature DataFrame (columns must include raw + engineered).
    config_path  : Path to config.yaml (used only for column name lists).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    cfg           = _load_config(config_path)
    feat_cfg      = cfg["features"]
    numeric_cols  = [c for c in feat_cfg["numeric"]    if c in X.columns]
    eng_cols      = [c for c in feat_cfg["engineered"] if c in X.columns]
    all_numeric   = numeric_cols + eng_cols

    # Apply same IQR caps (derived from training stats stored in preprocessor)
    X = cap_outliers_iqr(X, all_numeric)

    return preprocessor.transform(X)


def load_preprocessor(path: str) -> ColumnTransformer:
    """Deserialise a previously saved preprocessor."""
    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded from '%s'.", path)
    return preprocessor


def save_feature_names(
    feature_names: List[str],
    path: str = "models/feature_names.json",
) -> None:
    """Persist the list of feature names produced by the preprocessor."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(feature_names, fh, indent=2)
    logger.info("Feature names saved → %s", path)


def load_feature_names(path: str = "models/feature_names.json") -> List[str]:
    """Reload the persisted feature names."""
    with open(path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import generate_synthetic_dataset
    from feature_engineering import engineer_features

    df  = generate_synthetic_dataset(n_samples=500)
    df  = engineer_features(df)
    X   = df.drop(columns=["RiskFlag"])
    pp, names = fit_preprocessor(X)
    Xt  = pp.transform(X)
    print("Transformed shape:", Xt.shape)
    print("First 5 feature names:", names[:5])
