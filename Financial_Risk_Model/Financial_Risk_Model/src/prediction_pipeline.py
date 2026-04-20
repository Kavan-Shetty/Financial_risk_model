"""
src/prediction_pipeline.py
===========================
End-to-end prediction pipeline for financial risk scoring.

This module provides:
  1. RiskPredictor class — loads artefacts and scores individual records or
     batches from a CSV.
  2. A risk-tier mapping (Low / Medium / High / Very High) based on the
     predicted probability of default.
  3. CLI integration (called from main.py with --predict flag).

Risk tier definitions
---------------------
These thresholds are illustrative and should be calibrated against the
institution's actual Expected Loss (EL) targets and risk appetite statement.

  Tier          PD Range        Action
  ─────────────────────────────────────────────────────
  Low Risk      [0.00, 0.20)    Auto-approve; standard rate
  Medium Risk   [0.20, 0.40)    Approve with conditions / higher rate
  High Risk     [0.40, 0.60)    Refer to underwriter; collateral required
  Very High     [0.60, 1.00]    Decline or intensive underwriting

Operational integration
-----------------------
In a production lending system this pipeline would be exposed as a REST
microservice (FastAPI / Flask) or invoked via a batch scoring job.  The
current implementation uses file I/O for portability.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Risk tier mapping
# ─────────────────────────────────────────────────────────────────────────────

RISK_TIERS = [
    (0.00, 0.20, "Low Risk",       "Auto-approve. Standard lending rate applies."),
    (0.20, 0.40, "Medium Risk",    "Approve with conditions. Consider risk-based pricing."),
    (0.40, 0.60, "High Risk",      "Refer to underwriter. Collateral / guarantor required."),
    (0.60, 1.01, "Very High Risk", "Decline or subject to intensive credit review."),
]


def probability_to_risk_tier(probability: float) -> Dict[str, str]:
    """
    Map a predicted default probability to a labelled risk tier.

    Parameters
    ----------
    probability : Predicted probability of default in [0, 1].

    Returns
    -------
    Dict with keys: tier, recommendation, risk_score (0–1000 credit-score proxy).
    """
    for lower, upper, tier, recommendation in RISK_TIERS:
        if lower <= probability < upper:
            # Map PD to a 0–1000 risk score (inverse: higher PD → lower score)
            risk_score = int(round((1 - probability) * 1000))
            return {
                "tier":           tier,
                "recommendation": recommendation,
                "risk_score":     risk_score,        # proprietary score
                "pd":             round(probability, 4),
            }
    # Edge case: probability == 1.0
    return {
        "tier":           "Very High Risk",
        "recommendation": "Decline or subject to intensive credit review.",
        "risk_score":     0,
        "pd":             round(probability, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RiskPredictor class
# ─────────────────────────────────────────────────────────────────────────────

class RiskPredictor:
    """
    Loads trained artefacts and scores new customer data.

    Usage
    -----
    >>> predictor = RiskPredictor()
    >>> predictor.load_artifacts()
    >>> results = predictor.predict_from_csv("data/new_customers.csv")
    >>> print(results[["CustomerID", "tier", "pd", "risk_score"]])
    """

    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir   = Path(artifacts_dir)
        self.model           = None
        self.preprocessor    = None
        self.feature_names   = None
        self._loaded         = False

    # ── Artefact loading ─────────────────────────────────────────────────────

    def load_artifacts(
        self,
        model_filename:        str = "risk_model.pkl",
        preprocessor_filename: str = "preprocessor.pkl",
        feature_names_file:    str = "feature_names.json",
    ) -> "RiskPredictor":
        """
        Load model, preprocessor, and feature names from disk.

        Returns self for method chaining.
        """
        model_path = self.artifacts_dir / model_filename
        prep_path  = self.artifacts_dir / preprocessor_filename
        feat_path  = self.artifacts_dir / feature_names_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run 'python main.py --train' first to train and save the model."
            )
        if not prep_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {prep_path}\n"
                "Run 'python main.py --train' first."
            )

        self.model        = joblib.load(model_path)
        self.preprocessor = joblib.load(prep_path)

        if feat_path.exists():
            with open(feat_path) as fh:
                self.feature_names = json.load(fh)
        else:
            logger.warning("Feature names file not found at %s.", feat_path)

        self._loaded = True
        logger.info(
            "Artefacts loaded — model: %s | preprocessor: %s",
            model_path.name, prep_path.name,
        )
        return self

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Apply feature engineering + preprocessor transform to raw data."""
        # Late import to avoid circular dependency
        from feature_engineering import engineer_features

        df_eng = engineer_features(df)
        X      = self.preprocessor.transform(df_eng)
        return X

    # ── Single-record prediction ─────────────────────────────────────────────

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single customer record provided as a Python dict.

        Parameters
        ----------
        record : Dict mapping column names to raw values.

        Returns
        -------
        Dict with original fields plus risk assessment outputs.
        """
        if not self._loaded:
            raise RuntimeError("Call load_artifacts() before predicting.")

        df       = pd.DataFrame([record])
        X        = self._preprocess(df)
        prob     = float(self.model.predict_proba(X)[0, 1])
        tier_info = probability_to_risk_tier(prob)

        return {**record, **tier_info}

    # ── Batch prediction from CSV ─────────────────────────────────────────────

    def predict_from_csv(
        self,
        input_csv: str,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score all customers in *input_csv* and return a results DataFrame.

        Parameters
        ----------
        input_csv  : Path to the input CSV (same schema as training data,
                     without the RiskFlag target column).
        output_csv : If provided, write results to this path.

        Returns
        -------
        pd.DataFrame with all input columns plus risk-assessment columns.
        """
        if not self._loaded:
            raise RuntimeError("Call load_artifacts() before predicting.")

        if not Path(input_csv).exists():
            raise FileNotFoundError(f"Input file not found: {input_csv}")

        logger.info("Loading prediction input from '%s' …", input_csv)
        df = pd.read_csv(input_csv, low_memory=False)
        logger.info("Scoring %d records …", len(df))

        X     = self._preprocess(df)
        probs = self.model.predict_proba(X)[:, 1]

        # Build output DataFrame
        results = df.copy()
        results["pd"]         = np.round(probs, 4)
        results["risk_score"] = (np.round((1 - probs) * 1000)).astype(int)
        results["tier"]       = [
            probability_to_risk_tier(p)["tier"] for p in probs
        ]
        results["recommendation"] = [
            probability_to_risk_tier(p)["recommendation"] for p in probs
        ]

        # Summary statistics
        tier_counts = results["tier"].value_counts()
        logger.info("\nRisk Tier Distribution:\n%s", tier_counts.to_string())
        logger.info(
            "Portfolio Average PD: %.2f%% | Median PD: %.2f%%",
            probs.mean() * 100, np.median(probs) * 100,
        )

        if output_csv:
            os.makedirs(Path(output_csv).parent, exist_ok=True)
            results.to_csv(output_csv, index=False)
            logger.info("Predictions written → %s", output_csv)

        return results

    # ── Portfolio summary ─────────────────────────────────────────────────────

    def portfolio_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate-level portfolio summary by risk tier.

        Returns
        -------
        DataFrame showing count, % share, avg PD, avg risk score per tier.
        """
        summary = (
            results.groupby("tier")
            .agg(
                count         = ("pd", "count"),
                avg_pd        = ("pd", "mean"),
                median_pd     = ("pd", "median"),
                avg_risk_score= ("risk_score", "mean"),
            )
            .reset_index()
        )
        summary["pct_share"] = (
            summary["count"] / summary["count"].sum() * 100
        ).round(2)
        summary["avg_pd"]         = summary["avg_pd"].round(4)
        summary["median_pd"]      = summary["median_pd"].round(4)
        summary["avg_risk_score"] = summary["avg_risk_score"].round(0).astype(int)

        logger.info("\nPortfolio Summary:\n%s", summary.to_string(index=False))
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def score_new_data(
    input_csv: str,
    output_csv: Optional[str] = None,
    artifacts_dir: str = "models",
) -> pd.DataFrame:
    """
    Convenience wrapper: load artefacts and score a CSV in one call.

    Parameters
    ----------
    input_csv     : Path to input CSV.
    output_csv    : Path to write scored output (optional).
    artifacts_dir : Directory containing model / preprocessor files.

    Returns
    -------
    pd.DataFrame with predictions.
    """
    predictor = RiskPredictor(artifacts_dir=artifacts_dir)
    predictor.load_artifacts()
    results   = predictor.predict_from_csv(input_csv, output_csv=output_csv)
    summary   = predictor.portfolio_summary(results)
    return results, summary


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test tier mapping
    for pd_val in [0.05, 0.25, 0.50, 0.75]:
        info = probability_to_risk_tier(pd_val)
        print(f"PD={pd_val:.2f} → {info['tier']} | Score={info['risk_score']}")
