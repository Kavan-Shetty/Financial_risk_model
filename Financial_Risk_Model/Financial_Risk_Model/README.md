# 📊 Financial Risk Model

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready machine learning pipeline** for predicting credit/financial risk — classifying borrowers as low-risk or high-risk (potential defaulters) using demographic, financial, and loan-specific features.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Financial Risk Modelling Explained](#-financial-risk-modelling-explained)
3. [Dataset Description](#-dataset-description)
4. [Repository Structure](#-repository-structure)
5. [Feature Engineering](#-feature-engineering)
6. [Modelling Approach](#-modelling-approach)
7. [Evaluation Results](#-evaluation-results)
8. [Installation](#-installation)
9. [How to Run](#-how-to-run)
10. [Prediction API](#-prediction-api)
11. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

Financial institutions — banks, NBFCs, fintech lenders — must assess the creditworthiness of loan applicants before extending credit. Getting this wrong in either direction has a direct financial cost:

| Error Type | Description | Business Impact |
|---|---|---|
| False Negative | Defaulter approved | Credit loss, provisioning cost |
| False Positive | Good customer rejected | Lost revenue, reputational damage |

This project builds an end-to-end ML pipeline that:
- Ingests raw applicant data (demographics + financial indicators)
- Engineers **10 domain-specific financial ratio features**
- Trains and tunes **3 classifiers** (Logistic Regression, Random Forest, Gradient Boosting)
- Selects the best model using **stratified cross-validation + validation set AUC**
- Outputs a **Probability of Default (PD)** and **Risk Tier** for each applicant
- Reports **Gini coefficient, KS statistic, and Gains table** — standard model validation metrics used in Basel II / IFRS 9 frameworks

---

## 💡 Financial Risk Modelling Explained

### Probability of Default (PD)
The model predicts PD — the likelihood that a borrower will fail to meet their contractual obligations within a defined time horizon (typically 12 months). PD is one of three components of Expected Credit Loss (ECL):

```
ECL = PD × LGD × EAD
```

where:
- **PD** = Probability of Default (model output)
- **LGD** = Loss Given Default (1 − recovery rate; typically 40–60%)
- **EAD** = Exposure at Default (outstanding balance)

### Risk Tiers

| Tier | PD Range | Action |
|---|---|---|
| 🟢 Low Risk | 0% – 20% | Auto-approve; standard rate |
| 🟡 Medium Risk | 20% – 40% | Approve with conditions; risk-based pricing |
| 🟠 High Risk | 40% – 60% | Refer to underwriter; collateral required |
| 🔴 Very High Risk | 60% – 100% | Decline or intensive credit review |

### Regulatory Context
- **Basel II / III**: Requires banks to estimate PD using internal rating models validated with Gini ≥ 0.30
- **IFRS 9**: Mandates 12-month and lifetime ECL provisioning based on PD estimates
- **SR 11-7 (Fed)**: Requires model risk governance including feature explainability

---

## 📁 Dataset Description

The dataset contains **5,000 retail lending records** with the following features:

### Demographic Features
| Column | Type | Description |
|---|---|---|
| CustomerID | String | Unique customer identifier |
| Age | Integer | Applicant age (22–64) |
| MaritalStatus | Categorical | Single / Married / Divorced / Widowed |
| EducationLevel | Categorical | High School / Graduate / Post-Graduate / Doctorate |
| EmploymentType | Categorical | Salaried / Self-Employed / Business Owner / Unemployed |

### Financial Features
| Column | Type | Description |
|---|---|---|
| Income | Integer | Annual gross income (INR) |
| NetMonthlyIncome | Integer | Take-home monthly income |
| TotalAssets | Integer | Total asset value |
| TotalLiabilities | Integer | Total outstanding liabilities |
| CreditScore | Integer | Bureau credit score (300–900) |
| PropertyOwnership | Categorical | Owned / Rented / Mortgaged |

### Loan-Specific Features
| Column | Type | Description |
|---|---|---|
| LoanAmount | Integer | Requested loan amount (INR) |
| LoanTenure | Integer | Loan tenure in years (1–30) |
| MonthlyEMI | Integer | Monthly equated instalment |
| ExistingLoansCount | Integer | Number of active loans |
| LoanPurpose | Categorical | Home / Car / Education / Personal / Medical / Business |

### Target Variable
| Column | Values | Description |
|---|---|---|
| RiskFlag | 0 / 1 | 0 = Low Risk, 1 = High Risk (default) |

**Class Distribution**: ~50% high risk (synthetic dataset; real portfolios typically 5–20%)

---

## 🏗️ Repository Structure

```
Financial_Risk_Model/
│
├── data/
│   ├── raw/                        # Raw CSV data (generated or provided)
│   │   └── financial_risk_data.csv
│   └── processed/                  # Engineered + split data
│       ├── processed_data.csv
│       └── test_set.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA: distributions, correlations, class balance
│   ├── 02_feature_engineering.ipynb # DTI analysis, IV, FSI curve, binning
│   └── 03_model_training.ipynb     # CV, tuning, ROC/PR/calibration, Gini/KS, gains table
│
├── src/
│   ├── data_loader.py              # CSV loading, schema validation, train/val/test split
│   ├── preprocessing.py            # Imputation, outlier capping, scaling, OHE pipeline
│   ├── feature_engineering.py      # Financial ratio features, composite FSI, binning
│   ├── model_training.py           # Model catalogue, CV, grid search, model persistence
│   ├── model_evaluation.py         # Metrics, ROC/PR/calibration/confusion matrix plots
│   └── prediction_pipeline.py     # RiskPredictor class, risk tier mapping, batch scoring
│
├── models/                         # Serialised artefacts (git-ignored in production)
│   ├── risk_model.pkl              # Best fitted model
│   ├── preprocessor.pkl            # Fitted ColumnTransformer
│   ├── feature_names.json          # Post-encoding feature names
│   ├── cv_results.json             # Cross-validation audit trail
│   └── evaluation_results.csv      # Test-set metrics for all models
│
├── config/
│   └── config.yaml                 # All hyperparameters, paths, feature lists
│
├── utils/
│   ├── metrics.py                  # Gini, KS, IV/WoE, PSI, ECL, Gains table
│   └── helpers.py                  # Logging, seeding, timing, currency formatting
│
├── reports/
│   └── figures/                    # All visualisation outputs (PNG)
│
├── main.py                         # Unified CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

Ten domain-specific features are derived from the raw columns before modelling:

### Ratio Features (continuous)

```python
DebtToIncome       = (MonthlyEMI × 12) / Income
                     # CFPB QM hard cap at 43%

LoanToIncome       = LoanAmount / Income
                     # Basel III liquidity metric; >4× considered high risk

EMIToIncome        = MonthlyEMI / NetMonthlyIncome
                     # Monthly cash-flow pressure; >50% = distress signal

AssetToLiability   = TotalAssets / TotalLiabilities
                     # Coverage ratio; <1 = technically insolvent

NetWorth           = TotalAssets − TotalLiabilities
                     # Negative net worth strongly predicts default

LoanBurdenRatio    = LoanAmount / (NetMonthlyIncome × LoanTenure × 12)
                     # Fraction of lifetime earnings consumed by loan
```

### Composite Index

```python
FinancialStressIndex = 0.35 × norm(DebtToIncome)
                     + 0.25 × norm(LoanToIncome)
                     + 0.25 × norm(TotalLiabilities)
                     + 0.15 × norm(ExistingLoansCount)
```

### Binned Features (ordinal / categorical)

| Feature | Bins | Labels |
|---|---|---|
| CreditScoreBin | 300–579 / 580–669 / 670–739 / 740–799 / 800–900 | Poor / Fair / Good / VeryGood / Excellent |
| AgeBin | 18–29 / 30–44 / 45–59 / 60+ | YoungAdult / Adult / MiddleAged / Senior |
| LoanAmountBin | Tertiles | SmallLoan / MediumLoan / LargeLoan |

---

## 🤖 Modelling Approach

### Pipeline Architecture

```
Raw CSV
  └──> schema validation + sentinel replacement
         └──> feature engineering (10 new features)
                └──> outlier capping (IQR Winsorisation)
                       └──> ColumnTransformer
                              ├── numeric: SimpleImputer(median) → StandardScaler
                              └── categorical: SimpleImputer(mode) → OneHotEncoder(drop='first')
                                     └──> Classifier
                                            └──> Probability of Default (PD)
                                                   └──> Risk Tier + Risk Score
```

### Models

| Model | Rationale |
|---|---|
| **Logistic Regression** | Interpretable baseline; coefficients map directly to credit scorecard points; required for regulatory submissions |
| **Random Forest** | Handles non-linear interactions; robust to noise; Gini importances are transparent |
| **Gradient Boosting** | Typically achieves best AUC on tabular financial data; flexible loss functions |

### Class Imbalance Handling
- `class_weight='balanced'` in all sklearn estimators
- Optional SMOTE oversampling (imbalanced-learn) on the training set only — never on validation/test

### Hyperparameter Tuning
- `GridSearchCV` / `RandomizedSearchCV` with `StratifiedKFold` (5 folds)
- Primary metric: **ROC-AUC** (insensitive to threshold; handles imbalance)
- All tuning parameters defined in `config/config.yaml`

---

## 📈 Evaluation Results

### Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini |
|---|---|---|---|---|---|---|
| Logistic Regression | ~0.74 | ~0.72 | ~0.76 | ~0.74 | ~0.82 | ~0.64 |
| Random Forest | ~0.82 | ~0.80 | ~0.83 | ~0.81 | ~0.90 | ~0.80 |
| **Gradient Boosting** | **~0.84** | **~0.82** | **~0.85** | **~0.83** | **~0.92** | **~0.84** |

> Results are on the synthetic dataset. On real lending data, AUC typically ranges 0.70–0.85.

### Financial Metrics (Best Model)

| Metric | Value | Benchmark |
|---|---|---|
| Gini Coefficient | ~0.84 | ≥ 0.50 = Strong |
| KS Statistic | ~0.65 | ≥ 0.30 = Acceptable |
| Top-20% Capture | ~60% | Typical range: 40–70% |

### Gains Table (Best Model — Decile 1 = Highest Risk)

| Decile | Default Rate | Cumulative % Defaults Captured | Lift |
|---|---|---|---|
| 1 | ~75% | ~35% | ~3.0× |
| 2 | ~65% | ~60% | ~2.6× |
| 3 | ~50% | ~75% | ~2.0× |
| 5 | ~30% | ~88% | ~1.2× |

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kavan-Shetty/Financial_risk_model.git
cd Financial_risk_model

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Step by step

```bash
# Step 1: Generate synthetic data (skip if you have real data in data/raw/)
python main.py --generate

# Step 2: Train all models with hyperparameter tuning
python main.py --train

# Step 3: Evaluate on the held-out test set
python main.py --evaluate

# Step 4: Score new applicants
python main.py --predict data/raw/new_customers.csv --output results/scored.csv
```

### Option B — Full pipeline in one command

```bash
python main.py --all
```

### Option C — Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order: `01_data_exploration` → `02_feature_engineering` → `03_model_training`

### CLI Reference

```
usage: main.py [-h] [--config CONFIG] [--generate] [--train] [--no-tune]
               [--evaluate] [--predict INPUT_CSV] [--output OUTPUT_CSV]
               [--all] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Options:
  --config CONFIG       Path to config.yaml (default: config/config.yaml)
  --generate            Generate synthetic training dataset
  --train               Train + tune all models
  --no-tune             Skip hyperparameter search (faster, less accurate)
  --evaluate            Run full evaluation on test set
  --predict INPUT_CSV   Score new customers from a CSV file
  --output OUTPUT_CSV   Output path for predictions
  --all                 Run generate → train → evaluate end-to-end
  --log-level           Logging verbosity (default: INFO)
```

---

## 🔮 Prediction API

### Input CSV Schema

```csv
CustomerID,Age,Income,LoanAmount,LoanTenure,CreditScore,ExistingLoansCount,
MonthlyEMI,TotalAssets,TotalLiabilities,NetMonthlyIncome,EmploymentType,
MaritalStatus,EducationLevel,PropertyOwnership,LoanPurpose
CUST99001,34,850000,2500000,10,720,1,20833,5000000,1500000,70833,
Salaried,Married,Graduate,Owned,Home
```

### Output

```csv
CustomerID,Age,...,pd,risk_score,tier,recommendation
CUST99001,34,...,0.1823,818,Low Risk,Auto-approve. Standard lending rate applies.
```

### Python API

```python
from src.prediction_pipeline import RiskPredictor

predictor = RiskPredictor(artifacts_dir='models')
predictor.load_artifacts()

# Score a batch
results = predictor.predict_from_csv('new_customers.csv', output_csv='scored.csv')
summary = predictor.portfolio_summary(results)

# Score a single record
record = {'Age': 35, 'Income': 900000, 'LoanAmount': 3000000, ...}
result = predictor.predict_record(record)
print(f"PD: {result['pd']:.2%} | Tier: {result['tier']}")
```

---

## 🚀 Future Improvements

| Enhancement | Description | Priority |
|---|---|---|
| **SHAP Explainability** | Per-prediction SHAP values for model-risk governance and customer-facing explanations | High |
| **XGBoost / LightGBM** | Drop-in replacements for GradientBoosting; typically faster and higher AUC | High |
| **Model Calibration** | Platt scaling / isotonic regression to produce well-calibrated PD for IFRS 9 | High |
| **PSI Monitoring** | Population Stability Index dashboard to detect covariate shift in production | High |
| **FastAPI Service** | REST microservice with `/score` endpoint for real-time lending system integration | Medium |
| **MLflow Tracking** | Experiment tracking, model registry, and lineage for model risk governance | Medium |
| **Fairness Audit** | Bias analysis across gender, age, and marital status sub-groups | Medium |
| **Survival Analysis** | Time-to-default modelling with Cox Proportional Hazard / Weibull models | Low |
| **Deep Learning** | TabNet / NODE for learning complex feature interactions in large portfolios | Low |
| **Stress Testing** | Macro-economic scenario analysis (GDP shock, unemployment spike) on PD | Low |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Kavan Shetty**
- GitHub: [@Kavan-Shetty](https://github.com/Kavan-Shetty)

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — ML pipeline and model implementations
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE oversampling
- [Basel Committee on Banking Supervision](https://www.bis.org/bcbs/) — regulatory context
- [CFPB Qualified Mortgage Rule](https://www.consumerfinance.gov/) — DTI threshold guidance
