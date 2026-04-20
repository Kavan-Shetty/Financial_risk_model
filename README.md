# Financial Risk Model

End-to-end machine learning pipeline for credit risk classification.

The project trains multiple models, evaluates them with business-relevant metrics, and produces borrower-level risk predictions with tiered recommendations.

## Features

- Data loading and validation from CSV
- Domain-specific feature engineering for financial ratios
- Preprocessing pipeline with imputation, scaling, and one-hot encoding
- Model training for Logistic Regression, Random Forest, and Gradient Boosting
- Cross-validation and best-model selection
- Evaluation outputs including ROC-AUC, confusion matrix, Gini, KS, and gains table
- Batch prediction pipeline with risk tiers and recommendations

## Project Structure

```text
financial risk model/
|- main.py
|- config.yaml
|- requirements.txt
|- README.md
|- data_loader.py
|- feature_engineering.py
|- preprocessing.py
|- model_training.py
|- model_evaluation.py
|- prediction_pipeline.py
|- helpers.py
|- metrics.py
|- data/
|  |- raw/
|  \- processed/
|- models/
\- reports/
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the pipeline

Generate data:

```powershell
python main.py --generate
```

Train models:

```powershell
python main.py --train
```

Evaluate trained models:

```powershell
python main.py --evaluate
```

Run all stages (generate, train, evaluate):

```powershell
python main.py --all
```

Score new customers from CSV:

```powershell
python main.py --predict data/raw/new_customers.csv --output data/processed/new_customers_scored.csv
```

## CLI Options

```text
--config PATH      Path to config.yaml (default: config.yaml)
--generate         Generate synthetic training data
--train            Train models
--no-tune          Skip hyperparameter tuning during training
--evaluate         Evaluate models on test set
--predict CSV      Score customers from CSV
--output CSV       Output path for scored CSV
--all              Run generate, train, and evaluate sequentially
--log-level LEVEL  DEBUG | INFO | WARNING | ERROR
```

## Configuration

Key settings are defined in config.yaml:

- Data paths and split sizes
- Target column definition
- Feature groups
- Preprocessing behavior
- Model parameters and tuning grids
- Evaluation threshold and output paths

## Outputs

After training and evaluation, expected artifacts include:

- models/risk_model.pkl
- models/preprocessor.pkl
- models/feature_names.json
- models/cv_results.json
- models/evaluation_results.csv
- models/best_model_name.txt
- reports/figures/*

## Notes

- Python 3.10+ is recommended.
- The repository includes a nested copy under Financial_Risk_Model/Financial_Risk_Model for an alternate packaged layout.

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
