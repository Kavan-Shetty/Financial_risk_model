"""
main.py
=======
Unified command-line entry point for the Financial Risk Model pipeline.

Usage
-----
  # Step 1: generate synthetic data (for demo / testing)
  python main.py --generate

  # Step 2: train all models (with hyperparameter tuning)
  python main.py --train

  # Step 3: evaluate the best model on the held-out test set
  python main.py --evaluate

  # Step 4: score new customers from a CSV
  python main.py --predict data/raw/new_customers.csv

  # Run the full pipeline end-to-end
  python main.py --all

  # Options
  python main.py --train --no-tune          # skip hyperparam search
  python main.py --train --config path.yaml # custom config
  python main.py --predict input.csv --output results.csv
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

# Make src/ and utils/ importable regardless of working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "utils"))

from helpers import ensure_dirs, set_global_seed, setup_logging, timer, load_config
from data_loader import generate_synthetic_dataset, load_raw_data, split_data
from feature_engineering import engineer_features
from preprocessing import (
    fit_preprocessor,
    transform,
    save_feature_names,
    load_feature_names,
    load_preprocessor,
)
from model_training import (
    train_all_models,
    select_best_model,
    save_model,
    load_model,
    cross_validate_models,
    save_cv_results,
    get_model_catalogue,
)
from model_evaluation import evaluate_all_models
from prediction_pipeline import RiskPredictor

logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_generate(cfg: dict) -> None:
    """Generate a synthetic dataset and write to data/raw/."""
    logger.info("=" * 60)
    logger.info("STAGE: Generate Synthetic Dataset")
    logger.info("=" * 60)
    ensure_dirs("data/raw", "data/processed")
    generate_synthetic_dataset(
        n_samples=5000,
        random_state=cfg["project"]["random_state"],
        save_path=cfg["data"]["raw_path"],
    )
    logger.info("Dataset generated → %s", cfg["data"]["raw_path"])


def stage_train(cfg: dict, tune: bool = True) -> None:
    """
    Full training pipeline:
      load → feature-engineer → preprocess → CV → tune → train → select → save.
    """
    logger.info("=" * 60)
    logger.info("STAGE: Model Training")
    logger.info("=" * 60)

    ensure_dirs("data/processed", "models", "reports/figures")
    rs = cfg["project"]["random_state"]
    set_global_seed(rs)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    with timer("Data loading"):
        df = load_raw_data(config_path=args.config)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    with timer("Feature engineering"):
        df = engineer_features(df)

    # Save processed data
    df.to_csv(cfg["data"]["processed_path"], index=False)
    logger.info("Processed data saved → %s", cfg["data"]["processed_path"])

    # ── 3. Split ─────────────────────────────────────────────────────────────
    target_col = cfg["target"]["column"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col=target_col,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["validation_size"],
        random_state=rs,
    )

    # Persist test set for later evaluation
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv("data/processed/test_set.csv", index=False)

    # ── 4. Fit preprocessor (on training data only) ───────────────────────────
    with timer("Preprocessing"):
        preprocessor, feature_names = fit_preprocessor(
            X_train,
            config_path=args.config,
            save_path=os.path.join(cfg["artifacts"]["model_dir"],
                                   cfg["artifacts"]["preprocessor_name"]),
        )
        X_train_t = preprocessor.transform(X_train)
        X_val_t   = preprocessor.transform(X_val)
        X_test_t  = preprocessor.transform(X_test)

    save_feature_names(
        feature_names,
        path=os.path.join(cfg["artifacts"]["model_dir"],
                          cfg["artifacts"]["feature_names_file"]),
    )

    # ── 5. Cross-validation baseline ─────────────────────────────────────────
    with timer("Cross-validation"):
        catalogue   = get_model_catalogue(cfg)
        cv_results  = cross_validate_models(
            catalogue, X_train_t, y_train,
            cv_folds=cfg["tuning"]["cv_folds"],
            scoring=cfg["tuning"]["scoring"],
            random_state=rs,
        )
        save_cv_results(cv_results)

    # ── 6. Hyperparameter tuning + training ───────────────────────────────────
    with timer("Model training" + (" + tuning" if tune else "")):
        trained_models = train_all_models(
            X_train_t, y_train,
            config_path=args.config,
            tune=tune,
        )

    # ── 7. Select best model on validation set ────────────────────────────────
    best_name, best_model = select_best_model(
        trained_models, X_val_t, y_val,
        scoring=cfg["tuning"]["scoring"],
    )

    # ── 8. Save all models + best model alias ─────────────────────────────────
    model_dir = cfg["artifacts"]["model_dir"]
    for name, estimator in trained_models.items():
        save_model(estimator, name, model_dir=model_dir)

    save_model(
        best_model,
        best_name,
        model_dir=model_dir,
        filename=cfg["artifacts"]["best_model_name"],
    )
    logger.info("Best model '%s' saved as '%s'.", best_name, cfg["artifacts"]["best_model_name"])

    # Persist best model name for evaluation stage
    with open(os.path.join(model_dir, "best_model_name.txt"), "w") as fh:
        fh.write(best_name)

    logger.info("Training complete.")


def stage_evaluate(cfg: dict) -> None:
    """Load artefacts and run full evaluation on the held-out test set."""
    logger.info("=" * 60)
    logger.info("STAGE: Model Evaluation")
    logger.info("=" * 60)

    model_dir       = cfg["artifacts"]["model_dir"]
    plots_dir       = cfg["evaluation"]["plots_dir"]
    ensure_dirs(plots_dir)

    # Load test set
    test_path = "data/processed/test_set.csv"
    if not os.path.exists(test_path):
        logger.error("Test set not found at '%s'. Run --train first.", test_path)
        sys.exit(1)

    target_col = cfg["target"]["column"]
    test_df    = pd.read_csv(test_path)
    X_test     = test_df.drop(columns=[target_col])
    y_test     = test_df[target_col].values

    # Load artefacts
    preprocessor   = load_preprocessor(
        os.path.join(model_dir, cfg["artifacts"]["preprocessor_name"])
    )
    feature_names  = load_feature_names(
        os.path.join(model_dir, cfg["artifacts"]["feature_names_file"])
    )

    # Load all trained models
    model_names = ["logisticregression", "randomforest", "gradientboosting"]
    trained_models = {}
    for name in model_names:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            trained_models[name] = load_model(path)

    if not trained_models:
        logger.error("No trained model files found in '%s'. Run --train first.", model_dir)
        sys.exit(1)

    X_test_t = preprocessor.transform(X_test)

    # Run evaluation
    results_df = evaluate_all_models(
        trained_models, X_test_t, y_test, feature_names,
        plots_dir=plots_dir,
        threshold=cfg["evaluation"]["threshold"],
    )

    results_df.to_csv(os.path.join(model_dir, "evaluation_results.csv"))
    logger.info("Evaluation results saved → %s/evaluation_results.csv", model_dir)

    # Gains table for best model
    from metrics import gains_table, gini_coefficient, ks_statistic

    best_name_path = os.path.join(model_dir, "best_model_name.txt")
    if os.path.exists(best_name_path):
        with open(best_name_path) as fh:
            best_name = fh.read().strip().lower().replace(" ", "")
        if best_name in trained_models:
            best_model = trained_models[best_name]
            y_prob     = best_model.predict_proba(X_test_t)[:, 1]
            gini       = gini_coefficient(y_test, y_prob)
            ks         = ks_statistic(y_test, y_prob)
            gains      = gains_table(y_test, y_prob)
            logger.info("\nBest Model Advanced Metrics:")
            logger.info("  Gini Coefficient : %.4f", gini)
            logger.info("  KS Statistic     : %.4f", ks)
            logger.info("\nGains Table (decile):\n%s", gains.to_string(index=False))

    logger.info("Evaluation complete.")


def stage_predict(input_csv: str, output_csv: str, cfg: dict) -> None:
    """Score new customers from a CSV file."""
    logger.info("=" * 60)
    logger.info("STAGE: Prediction")
    logger.info("=" * 60)

    predictor = RiskPredictor(artifacts_dir=cfg["artifacts"]["model_dir"])
    predictor.load_artifacts(
        model_filename        = cfg["artifacts"]["best_model_name"],
        preprocessor_filename = cfg["artifacts"]["preprocessor_name"],
        feature_names_file    = cfg["artifacts"]["feature_names_file"],
    )

    results = predictor.predict_from_csv(input_csv, output_csv=output_csv)
    summary = predictor.portfolio_summary(results)

    print("\n" + "=" * 60)
    print("  RISK SCORING RESULTS")
    print("=" * 60)
    print(results[["CustomerID", "tier", "pd", "risk_score", "recommendation"]].to_string(index=False))
    print("\nPortfolio Summary:")
    print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Financial Risk Model — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--generate",  action="store_true",
                        help="Generate synthetic training data")
    parser.add_argument("--train",     action="store_true",
                        help="Train models")
    parser.add_argument("--no-tune",   action="store_true",
                        help="Skip hyperparameter tuning during --train")
    parser.add_argument("--evaluate",  action="store_true",
                        help="Evaluate trained models on the test set")
    parser.add_argument("--predict",   metavar="INPUT_CSV",
                        help="Score new customers from INPUT_CSV")
    parser.add_argument("--output",    metavar="OUTPUT_CSV", default=None,
                        help="Where to write prediction results (used with --predict)")
    parser.add_argument("--all",       action="store_true",
                        help="Run generate → train → evaluate end-to-end")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity level")
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    cfg = load_config(args.config)

    if not any([args.generate, args.train, args.evaluate, args.predict, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.all or args.generate:
        stage_generate(cfg)

    if args.all or args.train:
        stage_train(cfg, tune=not args.no_tune)

    if args.all or args.evaluate:
        stage_evaluate(cfg)

    if args.predict:
        output = args.output or args.predict.replace(".csv", "_scored.csv")
        stage_predict(args.predict, output, cfg)
