"""
utils/metrics.py
================
Supplementary financial and statistical metrics not in sklearn.

Financial-specific metrics
--------------------------
  Gini Coefficient       : Industry standard measure of model discriminatory power.
                           Gini = 2 × AUC − 1.  Gini ≥ 0.30 is generally acceptable;
                           Gini ≥ 0.50 is considered strong for retail credit models.

  KS Statistic           : Kolmogorov-Smirnov statistic.  Maximum separation between
                           the CDFs of the defaulter and non-defaulter score distributions.
                           KS ≥ 0.30 is a common internal threshold in Basel II PD models.

  Population Stability   : PSI measures whether the score distribution has shifted
  Index (PSI)              between training and production populations.
                           PSI < 0.10: stable.  0.10–0.25: slight shift.  > 0.25: alert.

  Expected Loss          : EL = PD × LGD × EAD.  If LGD and EAD are available,
                           this function computes portfolio-level expected credit loss.

  Lift / Gains curve     : Used in direct marketing / pre-screening to quantify how
                           much better the model performs vs random selection.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminatory power metrics
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Gini coefficient from AUC.

    Gini = 2 × AUC − 1

    A Gini of 0 means the model has no discriminatory power (equivalent to
    random guessing). A Gini of 1 means perfect separation.

    Parameters
    ----------
    y_true : Ground-truth binary labels (0/1).
    y_prob : Predicted probability of the positive class.

    Returns
    -------
    float in [-1, 1].  Negative values indicate the model is worse than random.
    """
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic for binary classifiers.

    Measures the maximum vertical distance between the cumulative distribution
    of scores for defaulters (positive class) and non-defaulters.

    Parameters
    ----------
    y_true : Binary labels (1 = default, 0 = no default).
    y_prob : Predicted probability of default.

    Returns
    -------
    float in [0, 1].
    """
    df = pd.DataFrame({"score": y_prob, "label": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos

    if n_pos == 0 or n_neg == 0:
        logger.warning("KS statistic undefined: one class has zero samples.")
        return 0.0

    df["cum_pos"] = df["label"].cumsum() / n_pos
    df["cum_neg"] = (1 - df["label"]).cumsum() / n_neg
    df["ks"]      = (df["cum_pos"] - df["cum_neg"]).abs()

    return float(df["ks"].max())


def iv_woe(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "RiskFlag",
    bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for a feature.

    WoE and IV are standard credit-scoring techniques:
      WoE_i = ln(Distribution_Events_i / Distribution_Non-Events_i)
      IV     = Σ (Distribution_Events_i − Distribution_Non-Events_i) × WoE_i

    IV interpretation:
      < 0.02  : Unpredictive
      0.02–0.1: Weak predictor
      0.1–0.3 : Medium predictor
      0.3–0.5 : Strong predictor
      > 0.5   : Suspicious (possible data leakage)

    Parameters
    ----------
    df          : DataFrame containing feature and target.
    feature_col : Column name of the feature to analyse.
    target_col  : Binary target column name.
    bins        : Number of quantile bins for continuous features.

    Returns
    -------
    (IV value, WoE DataFrame with per-bin statistics)
    """
    _EPS = 1e-9

    tmp = df[[feature_col, target_col]].copy().dropna()

    if tmp[feature_col].dtype in [object, "category"]:
        tmp["bin"] = tmp[feature_col].astype(str)
    else:
        try:
            tmp["bin"] = pd.qcut(tmp[feature_col], q=bins, duplicates="drop").astype(str)
        except ValueError:
            tmp["bin"] = pd.cut(tmp[feature_col], bins=bins, duplicates="drop").astype(str)

    total_events     = tmp[target_col].sum()
    total_non_events = len(tmp) - total_events

    stats = (
        tmp.groupby("bin")[target_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "events", "count": "total"})
        .reset_index()
    )
    stats["non_events"]      = stats["total"] - stats["events"]
    stats["pct_events"]      = stats["events"]      / (total_events     + _EPS)
    stats["pct_non_events"]  = stats["non_events"]  / (total_non_events + _EPS)
    stats["woe"]             = np.log(
        stats["pct_events"] / (stats["pct_non_events"] + _EPS)
    )
    stats["iv_bin"]          = (stats["pct_events"] - stats["pct_non_events"]) * stats["woe"]
    iv_total                 = stats["iv_bin"].sum()

    return float(iv_total), stats


# ─────────────────────────────────────────────────────────────────────────────
# Population Stability Index
# ─────────────────────────────────────────────────────────────────────────────

def population_stability_index(
    baseline_scores: np.ndarray,
    current_scores:  np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    Compares the score distribution at training time (baseline) vs. a later
    deployment period (current) to detect covariate shift.

    Parameters
    ----------
    baseline_scores : Model scores from the training / reference population.
    current_scores  : Model scores from the current / monitoring population.
    bins            : Number of equal-width bins.

    Returns
    -------
    PSI value.  < 0.10 stable, 0.10–0.25 slight shift, > 0.25 alert.
    """
    _EPS = 1e-9
    breakpoints = np.linspace(0.0, 1.0, bins + 1)

    baseline_pct = np.histogram(baseline_scores, bins=breakpoints)[0] / len(baseline_scores)
    current_pct  = np.histogram(current_scores,  bins=breakpoints)[0] / len(current_scores)

    # Replace zeros to avoid log(0)
    baseline_pct = np.where(baseline_pct == 0, _EPS, baseline_pct)
    current_pct  = np.where(current_pct  == 0, _EPS, current_pct)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    if psi < 0.10:
        label = "STABLE"
    elif psi < 0.25:
        label = "SLIGHT SHIFT"
    else:
        label = "SIGNIFICANT SHIFT — retrain recommended"

    logger.info("PSI = %.4f  → %s", psi, label)
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────────
# Expected Credit Loss
# ─────────────────────────────────────────────────────────────────────────────

def expected_credit_loss(
    pd_values:  np.ndarray,
    lgd_values: np.ndarray,
    ead_values: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute Expected Loss (EL) per record and portfolio total.

    EL = PD × LGD × EAD

    where:
      PD  = Probability of Default (model output)
      LGD = Loss Given Default     (1 − recovery rate; typically 40–60 %)
      EAD = Exposure at Default    (outstanding loan balance at time of default)

    This is the core formula underpinning IFRS 9 impairment provisioning
    and Basel III regulatory capital requirements.

    Parameters
    ----------
    pd_values  : Array of predicted probabilities of default (0–1).
    lgd_values : Array of loss-given-default rates (0–1).
    ead_values : Array of exposure-at-default amounts (currency units).

    Returns
    -------
    (per_record_EL array, total_portfolio_EL)
    """
    el_per_record  = pd_values * lgd_values * ead_values
    total_el       = float(el_per_record.sum())
    logger.info(
        "Portfolio Expected Loss: %.2f (%.2f%% of total exposure)",
        total_el, total_el / ead_values.sum() * 100,
    )
    return el_per_record, total_el


# ─────────────────────────────────────────────────────────────────────────────
# Gains / Lift table
# ─────────────────────────────────────────────────────────────────────────────

def gains_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute a decile-level gains table.

    The gains table shows how many defaults are captured in the top N%
    of the risk-score distribution.  It is the primary tool for
    communicating model value to credit-risk teams and senior management.

    Parameters
    ----------
    y_true    : True binary labels.
    y_prob    : Predicted default probabilities.
    n_deciles : Number of equal-size score buckets (default 10).

    Returns
    -------
    pd.DataFrame with decile-level statistics.
    """
    df = pd.DataFrame({"prob": y_prob, "label": y_true})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=n_deciles, labels=False) + 1

    total_events = df["label"].sum()

    table = (
        df.groupby("decile")
        .agg(
            n_records     = ("label", "count"),
            n_defaults    = ("label", "sum"),
            avg_prob      = ("prob",  "mean"),
        )
        .reset_index()
    )
    table["default_rate"]   = table["n_defaults"]  / table["n_records"]
    table["cum_defaults"]   = table["n_defaults"].cumsum()
    table["pct_defaults"]   = table["cum_defaults"] / total_events * 100
    table["lift"]           = table["default_rate"] / (total_events / len(df))

    return table
