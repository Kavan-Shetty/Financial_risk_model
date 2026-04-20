"""
utils/helpers.py
================
Miscellaneous utility functions shared across the project.

Covers:
  * Logging setup.
  * Directory management.
  * DataFrame display helpers.
  * Configuration loading with override support.
  * Reproducibility seeding.
  * Timer / profiling context manager.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure root logger with console (and optional file) handlers.

    Parameters
    ----------
    level    : Logging level (e.g. logging.DEBUG / logging.INFO).
    log_file : Path to write log file.  None = console only.
    fmt      : Log record format string.
    datefmt  : Date format for the timestamp.

    Returns
    -------
    Root logger instance.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(
    config_path: str = "config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load YAML configuration and apply optional runtime overrides.

    Parameters
    ----------
    config_path : Path to the YAML file.
    overrides   : Flat dict of dot-notation keys, e.g.
                  {'data.test_size': 0.15, 'project.random_state': 0}.

    Returns
    -------
    Nested dict from the YAML, with overrides applied.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if overrides:
        for key, value in overrides.items():
            parts  = key.split(".")
            target = cfg
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (if available) TensorFlow / PyTorch.

    Ensures reproducibility of data splitting, model initialisation, and
    stochastic training procedures (e.g. gradient boosting subsampling).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# File system
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create all directory paths if they do not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def project_root() -> Path:
    """Return the project root directory (parent of utils/)."""
    return Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────────────────────────────────────

def describe_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """
    Print a structured summary of a DataFrame.

    Includes shape, dtypes, missing values, and — if *target_col* is given —
    the class balance breakdown.
    """
    print("=" * 60)
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 60)

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\n⚠  Missing values:")
        for col, n in missing[missing > 0].items():
            pct = n / len(df) * 100
            print(f"   {col:35s}  {n:6d}  ({pct:.1f}%)")
    else:
        print("\n✅  No missing values detected.")

    # Data types
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())

    # Target distribution
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        total  = len(df)
        print(f"\nTarget distribution ({target_col}):")
        for label, n in counts.items():
            print(f"   Class {label}: {n:6d}  ({n/total*100:.1f}%)")

    print("=" * 60)


def profile_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing data per column.

    Returns
    -------
    pd.DataFrame with columns: column, missing_count, missing_pct, dtype.
    """
    missing_count = df.isnull().sum()
    missing_pct   = missing_count / len(df) * 100
    return (
        pd.DataFrame({
            "column":        missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct":   missing_pct.values.round(2),
            "dtype":         [str(df[c].dtype) for c in missing_count.index],
        })
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


def value_counts_pct(series: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """Return value counts with percentage column."""
    counts = series.value_counts()
    pcts   = series.value_counts(normalize=normalize) * 100
    return pd.DataFrame({"count": counts, "pct": pcts.round(2)})


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "Operation"):
    """
    Context manager that logs wall-clock time for a code block.

    Usage
    -----
    with timer("Model training"):
        model.fit(X_train, y_train)
    """
    logger = logging.getLogger(__name__)
    start  = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f seconds.", label, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Financial display helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_currency(amount: float, symbol: str = "₹", decimals: int = 0) -> str:
    """Format a number as a currency string."""
    if abs(amount) >= 1e7:
        return f"{symbol}{amount/1e7:.2f} Cr"
    elif abs(amount) >= 1e5:
        return f"{symbol}{amount/1e5:.2f} L"
    else:
        return f"{symbol}{amount:,.{decimals}f}"


def risk_score_to_grade(score: int) -> str:
    """
    Map a proprietary risk score (0–1000) to a letter grade.

    Grade  Score Range  Interpretation
    AAA    900–1000     Exceptional creditworthiness
    AA     800–899      Very strong capacity to meet obligations
    A      700–799      Strong; somewhat susceptible to economic downturns
    BBB    600–699      Adequate capacity; more exposed to adverse conditions
    BB     500–599      Speculative elements; significant uncertainty
    B      400–499      Currently meeting obligations; vulnerable
    CCC    300–399      Currently vulnerable; dependent on favourable conditions
    CC     200–299      Highly vulnerable; near default
    C      0–199        Near or in default
    """
    grade_map = [
        (900, "AAA"), (800, "AA"), (700, "A"),
        (600, "BBB"), (500, "BB"), (400, "B"),
        (300, "CCC"), (200, "CC"), (0, "C"),
    ]
    for threshold, grade in grade_map:
        if score >= threshold:
            return grade
    return "C"
