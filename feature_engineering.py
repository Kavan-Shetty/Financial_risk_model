"""
src/feature_engineering.py
===========================
Derives domain-specific financial features from the raw columns.

Financial risk modelling heavily relies on *ratio features* because they
normalise for scale — e.g., a £500,000 loan is low-risk for a millionaire
but catastrophic for someone earning £20,000.  The ratios computed here
mirror standard credit-bureau indicators:

  Debt-to-Income (DTI)     — used by Fannie Mae / Freddie Mac guidelines
  Loan-to-Income (LTI)     — Basel III liquidity risk metric
  Debt Service Coverage    — core corporate / retail lending metric
  Net Worth                — assets minus liabilities
  Financial Stress Index   — composite indicator
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small epsilon to avoid division by zero in ratio computations
_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature constructors
# ─────────────────────────────────────────────────────────────────────────────

def add_debt_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debt-to-Income ratio (DTI).

    DTI = Total Annual Debt Obligations / Annual Gross Income
    A DTI > 0.43 is typically a hard cut-off for qualified mortgages (CFPB).
    DTI > 0.50 is considered high-risk in most retail lending frameworks.
    """
    annual_emi = df["MonthlyEMI"] * 12
    df["DebtToIncome"] = annual_emi / (df["Income"] + _EPS)
    return df


def add_loan_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan-to-Income ratio (LTI).

    LTI = Total Loan Amount / Annual Gross Income
    Regulatory guidance in many jurisdictions caps LTI at 4–5×.
    """
    df["LoanToIncome"] = df["LoanAmount"] / (df["Income"] + _EPS)
    return df


def add_emi_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMI-to-Net-Monthly-Income ratio.

    Measures monthly cash-flow stress.  Ratios above 0.40–0.50 signal that
    debt repayment is consuming an unsustainable fraction of take-home pay.
    """
    df["EMIToIncome"] = df["MonthlyEMI"] / (df["NetMonthlyIncome"] + _EPS)
    return df


def add_asset_to_liability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asset Coverage Ratio.

    AssetToLiability = Total Assets / Total Liabilities
    Ratio < 1 indicates negative net equity (technically insolvent).
    Ratio in [1, 1.5] is fragile; > 2 is generally considered healthy.
    """
    df["AssetToLiability"] = df["TotalAssets"] / (df["TotalLiabilities"] + _EPS)
    return df


def add_net_worth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net Worth = Total Assets - Total Liabilities.

    Negative net worth is a strong predictor of default in retail lending.
    """
    df["NetWorth"] = df["TotalAssets"] - df["TotalLiabilities"]
    return df


def add_loan_burden_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan Burden Ratio = (LoanAmount * 12) / (NetMonthlyIncome * LoanTenure * 12).

    Estimates what fraction of lifetime earnings will be consumed by loan
    repayment.  High values indicate structural over-indebtedness.
    """
    lifetime_income = df["NetMonthlyIncome"] * df["LoanTenure"] * 12
    df["LoanBurdenRatio"] = df["LoanAmount"] / (lifetime_income + _EPS)
    return df


def add_financial_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Financial Stress Index (FSI).

    A weighted composite score blending:
      - DTI         : cash-flow pressure
      - LTI         : loan size relative to income
      - Liabilities : absolute debt load
      - ExistingLoans: existing credit obligations

    All components are normalised to [0, 1] via min-max within the dataset.
    This is a *rank* indicator — not an absolute score — and should not be
    compared across independently processed datasets.
    """
    def _minmax(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + _EPS)

    fsi = (
        0.35 * _minmax(df["DebtToIncome"])
        + 0.25 * _minmax(df["LoanToIncome"])
        + 0.25 * _minmax(df["TotalLiabilities"])
        + 0.15 * _minmax(df["ExistingLoansCount"])
    )
    df["FinancialStressIndex"] = fsi
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Binning features  (ordinal / categorical groupings)
# ─────────────────────────────────────────────────────────────────────────────

def add_credit_score_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin credit scores into standard industry bands.

    Band definitions (approximate — vary by bureau):
      Poor     : 300–579   (high risk)
      Fair     : 580–669   (sub-prime)
      Good     : 670–739   (near-prime)
      Very Good: 740–799   (prime)
      Excellent: 800–850   (super-prime)
    """
    bins   = [0, 579, 669, 739, 799, 900]
    labels = ["Poor", "Fair", "Good", "VeryGood", "Excellent"]
    df["CreditScoreBin"] = pd.cut(
        df["CreditScore"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers into life-stage cohorts.

    Life-stage mapping (financial behaviour proxy):
      Young Adult  : 18–29 (limited credit history)
      Adult        : 30–44 (peak earning / debt phase)
      Middle-Aged  : 45–59 (pre-retirement)
      Senior       : 60+   (fixed income / lower risk appetite)
    """
    bins   = [17, 29, 44, 59, 100]
    labels = ["YoungAdult", "Adult", "MiddleAged", "Senior"]
    df["AgeBin"] = pd.cut(
        df["Age"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df


def add_loan_amount_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorise loan size into risk tiers.

    Smaller loans have lower absolute loss given default (LGD);
    larger loans carry higher severity of loss.
    """
    percentiles = df["LoanAmount"].quantile([0, 0.33, 0.67, 1.0]).values
    bins   = list(percentiles)
    labels = ["SmallLoan", "MediumLoan", "LargeLoan"]

    # Ensure monotonically increasing unique bins
    bins = sorted(set(bins))
    if len(bins) < 4:
        # Fallback if data is degenerate
        df["LoanAmountBin"] = "MediumLoan"
        return df

    df["LoanAmountBin"] = pd.cut(
        df["LoanAmount"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master feature-engineering function
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    drop_intermediates: bool = False,
) -> pd.DataFrame:
    """
    Apply the complete feature-engineering pipeline to a DataFrame.

    This function is the single entry point called by training scripts and
    the prediction pipeline.  It must be idempotent: calling it twice on the
    same DataFrame should produce the same result.

    Parameters
    ----------
    df                 : DataFrame containing raw columns (numeric + categorical).
    drop_intermediates : If True, drop columns that were only used to construct
                         engineered features (rarely needed outside notebooks).

    Returns
    -------
    pd.DataFrame with all original columns plus new engineered features.
    """
    df = df.copy()

    logger.info("Engineering financial ratio features …")

    # ── Ratio features ───────────────────────────────────────────────────────
    df = add_debt_to_income(df)
    df = add_loan_to_income(df)
    df = add_emi_to_income(df)
    df = add_asset_to_liability(df)
    df = add_net_worth(df)
    df = add_loan_burden_ratio(df)

    # ── Composite index (depends on ratio features above) ────────────────────
    df = add_financial_stress_index(df)

    # ── Binned / ordinal features ─────────────────────────────────────────────
    df = add_credit_score_bin(df)
    df = add_age_bin(df)
    df = add_loan_amount_bin(df)

    engineered_cols = [
        "DebtToIncome", "LoanToIncome", "EMIToIncome",
        "AssetToLiability", "NetWorth", "LoanBurdenRatio",
        "FinancialStressIndex",
        "CreditScoreBin", "AgeBin", "LoanAmountBin",
    ]
    created = [c for c in engineered_cols if c in df.columns]
    logger.info("Engineered %d new features: %s", len(created), created)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_samples=200)
    df_eng = engineer_features(df)
    print("\nNew columns:", [c for c in df_eng.columns if c not in df.columns])
    print(df_eng[["DebtToIncome", "LoanToIncome", "FinancialStressIndex"]].describe())
