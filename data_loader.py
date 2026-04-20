"""
src/data_loader.py
==================
Responsible for:
  1. Reading the raw CSV dataset from disk.
  2. Validating that all expected columns are present.
  3. Performing initial type coercions and light cleaning.
  4. Splitting into train / validation / test sets.

Financial context
-----------------
A well-defined data-loading layer is critical in credit-risk pipelines because
raw data from banking / lending systems often arrives with:
  * Mixed data types (e.g. income stored as strings with currency symbols).
  * Duplicate records per customer.
  * Unrecognised sentinel values (e.g. -999 used for "missing").
All of these must be caught *before* modelling begins.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Columns we expect to find in the raw dataset.
# Extend this list when the dataset schema changes.
EXPECTED_COLUMNS = [
    "CustomerID",
    "Age",
    "Income",
    "LoanAmount",
    "LoanTenure",
    "CreditScore",
    "ExistingLoansCount",
    "MonthlyEMI",
    "TotalAssets",
    "TotalLiabilities",
    "NetMonthlyIncome",
    "EmploymentType",
    "MaritalStatus",
    "EducationLevel",
    "PropertyOwnership",
    "LoanPurpose",
    "RiskFlag",          # Target: 1 = default / high-risk, 0 = low-risk
]

# Values treated as missing during load
MISSING_SENTINELS = [-999, -1, "NA", "N/A", "nan", "none", "None", ""]


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str = "config.yaml") -> dict:
    """Load project-level configuration from YAML."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known sentinel / placeholder missing values with np.nan."""
    return df.replace(MISSING_SENTINELS, np.nan)


def _coerce_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Strip currency symbols / commas and coerce specified columns to float.
    Handles cases where 'Income' is stored as '$45,000' etc.
    """
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,£€\s]", "", regex=True)
                .replace("nan", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_duplicates(df: pd.DataFrame, key_col: str = "CustomerID") -> pd.DataFrame:
    """
    Remove duplicate rows.
    If a customer-ID column exists, keep the last occurrence (most recent record).
    """
    before = len(df)
    if key_col in df.columns:
        df = df.drop_duplicates(subset=[key_col], keep="last")
    else:
        df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.warning("Dropped %d duplicate rows.", removed)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(
    filepath: Optional[str] = None,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """
    Load the raw dataset from *filepath* (or fall back to config path).

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file.  If None the path from config.yaml is used.
    config_path : str
        Path to the project configuration YAML.

    Returns
    -------
    pd.DataFrame
        Raw, lightly-cleaned DataFrame ready for preprocessing.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found.
    ValueError
        If required columns are absent from the dataset.
    """
    cfg = _load_config(config_path)
    if filepath is None:
        filepath = cfg["data"]["raw_path"]

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{filepath}'. "
            "Please place the CSV in data/raw/ or update config.yaml."
        )

    logger.info("Loading raw data from '%s' …", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Raw shape: %s rows × %s columns", *df.shape)

    # ── Validate schema ──────────────────────────────────────────────────────
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            "Check EXPECTED_COLUMNS in data_loader.py."
        )

    # ── Light cleaning ───────────────────────────────────────────────────────
    df = _replace_sentinels(df)

    numeric_cols = cfg["features"]["numeric"]
    df = _coerce_numeric_columns(df, numeric_cols)

    df = _drop_duplicates(df)

    # Ensure target is integer (0 / 1)
    target_col = cfg["target"]["column"]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")

    logger.info("Clean shape after dedup / coerce: %s rows × %s columns", *df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "RiskFlag",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,  pd.Series,  pd.Series]:
    """
    Stratified train / validation / test split.

    Stratification on *target_col* ensures the minority class (defaulters)
    is proportionally represented in every split — critical for imbalanced
    credit-risk datasets where defaults may be < 10 % of records.

    Parameters
    ----------
    df          : Full cleaned DataFrame.
    target_col  : Name of the binary target column.
    test_size   : Fraction of data held out as the final test set.
    val_size    : Fraction of *training* data set aside for validation.
    random_state: Reproducibility seed.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Second split: carve out validation from training pool
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1.0 - test_size),
        stratify=y_train_val,
        random_state=random_state,
    )

    logger.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Default rate  → train: %.2f%% | val: %.2f%% | test: %.2f%%",
        y_train.mean() * 100,
        y_val.mean() * 100,
        y_test.mean() * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_synthetic_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    save_path: str = "data/raw/financial_risk_data.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic financial-risk dataset for demonstration / testing.

    This mirrors the structure of a typical retail-lending dataset:
      * Customer demographics (age, education, marital status).
      * Financial indicators (income, credit score, assets, liabilities).
      * Loan specifics (amount, tenure, EMI, purpose).
      * Binary target: RiskFlag (1 = high risk / default, 0 = low risk).

    The default rate is intentionally kept at ~20 % to simulate a realistic
    (though slightly elevated) credit portfolio.

    Parameters
    ----------
    n_samples    : Number of synthetic customer records.
    random_state : NumPy random seed.
    save_path    : Where to write the CSV (directories are created as needed).

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_state)

    employment_types    = ["Salaried", "Self-Employed", "Business Owner", "Unemployed"]
    marital_statuses    = ["Single", "Married", "Divorced", "Widowed"]
    education_levels    = ["High School", "Graduate", "Post-Graduate", "Doctorate"]
    property_ownership  = ["Owned", "Rented", "Mortgaged"]
    loan_purposes       = ["Home", "Car", "Education", "Personal", "Medical", "Business"]

    age             = rng.integers(22, 65, n_samples)
    income          = rng.integers(150_000, 2_500_000, n_samples)           # annual, INR
    loan_amount     = rng.integers(50_000, 5_000_000, n_samples)
    loan_tenure     = rng.integers(1, 30, n_samples)                        # years
    credit_score    = rng.integers(300, 900, n_samples)
    existing_loans  = rng.integers(0, 6, n_samples)
    monthly_emi     = (loan_amount / (loan_tenure * 12)).astype(int)
    total_assets    = rng.integers(100_000, 10_000_000, n_samples)
    total_liab      = rng.integers(0, 8_000_000, n_samples)
    net_monthly_inc = (income / 12).astype(int)

    emp_type        = rng.choice(employment_types, n_samples, p=[0.55, 0.25, 0.12, 0.08])
    marital_status  = rng.choice(marital_statuses, n_samples, p=[0.35, 0.50, 0.10, 0.05])
    education       = rng.choice(education_levels, n_samples, p=[0.20, 0.45, 0.28, 0.07])
    property_own    = rng.choice(property_ownership, n_samples, p=[0.30, 0.45, 0.25])
    loan_purpose    = rng.choice(loan_purposes, n_samples)

    # Construct a risk score based on domain logic, then threshold it
    risk_score = (
        - 0.003 * credit_score                          # higher credit score → less risk
        + 0.4   * (loan_amount / income)                # higher loan-to-income → more risk
        + 0.5   * existing_loans                        # more existing debt → more risk
        - 0.01  * (total_assets / 1_000_000)            # higher assets → less risk
        + 0.3   * (total_liab / np.maximum(total_assets, 1))  # leverage
        + rng.normal(0, 0.5, n_samples)                 # noise
    )
    risk_flag = (risk_score > risk_score.mean()).astype(int)

    df = pd.DataFrame({
        "CustomerID":         [f"CUST{i:05d}" for i in range(n_samples)],
        "Age":                age,
        "Income":             income,
        "LoanAmount":         loan_amount,
        "LoanTenure":         loan_tenure,
        "CreditScore":        credit_score,
        "ExistingLoansCount": existing_loans,
        "MonthlyEMI":         monthly_emi,
        "TotalAssets":        total_assets,
        "TotalLiabilities":   total_liab,
        "NetMonthlyIncome":   net_monthly_inc,
        "EmploymentType":     emp_type,
        "MaritalStatus":      marital_status,
        "EducationLevel":     education,
        "PropertyOwnership":  property_own,
        "LoanPurpose":        loan_purpose,
        "RiskFlag":           risk_flag,
    })

    os.makedirs(Path(save_path).parent, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info("Synthetic dataset saved → %s  (%d rows)", save_path, n_samples)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_synthetic_dataset(n_samples=5000)
    print(df.head())
    print("\nClass distribution:\n", df["RiskFlag"].value_counts())
