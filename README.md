# 💳 Loan Defaulter Prediction System

An end-to-end Machine Learning and Underwriting solution for predicting the probability of loan defaults based on applicant demographics, financial profile, loan terms, and credit history.

---

## 📌 Project Overview

This project builds an automated credit risk assessment engine that:
1. **Handles Real-World Dirty Data**: Resolves duplicate records, case variations, spelling typos, corrupted numerical values, invalid age ranges, missing values, and mixed date formats.
2. **Engineers Financial Indicators**: Computes Equated Monthly Installment (EMI), Debt-to-Income (DTI), Loan-to-Income ratio, total payable interest, per-dependent income, and credit risk tiers.
3. **Benchmarks Multiple ML Models**: Evaluates Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM with 5-fold Stratified Cross-Validation.
4. **Interactive Web Dashboard**: Streamlit interface for single-applicant underwriting, batch CSV file scoring, visual analytics, and model explainability.

---

## 📊 Dataset & Schema

| Column | Raw Observations & Cleaning Strategy |
| :--- | :--- |
| `loan_id` | Stripped whitespace, standardized case, removed 85 duplicate records. |
| `application_date` | Parsed multiple mixed formats (`DD-Mon-YYYY`, `DD/MM/YYYY`, `YYYY-MM-DD`). Extracted year, month, and day-of-week. |
| `age` | Replaced minors (`<18`) and extreme outliers (`>90`, e.g. `999.0`, `150.0`) with `NaN`, followed by median imputation. |
| `gender` | Standardized variations (`Male`, `male`, `M`, `m`, `Female`, `female`, `F`, `f`) to `Male` and `Female`. |
| `marital_status` | Unified variations and typos (`Marreid` -> `Married`, `Single`, `Divorced`). |
| `dependents` | Clipped outliers to domain range (`0` to `8`). |
| `employment_type` | Standardized categories (`Salaried`, `Self-Employed`, `Retired`, `Unemployed`). Replaced placeholder artifacts (`-`, `?`, `NA`) with standard labels. |
| `monthly_income` | Handled negative values (`-5000.0`) and extreme values (`50,000,000.0`), imputed with median. |
| `credit_score` | Clipped to valid credit bureau range (`300` - `900`), removing invalid values (`-100.0`, `0.0`, `9999.0`). |
| `loan_amount` | Corrected placeholder values (`0.0`, `-10000.0`, `999999999.0`). |
| `loan_term_months` | Corrected clerical typos (`1200` months -> `120` months). Valid terms: `12` to `120`. |
| `interest_rate` | Fixed typos (`150.0%` -> `15.0%`) and filtered non-positive rates. |
| `loan_purpose` | Grouped synonymous terms (`biz` -> `Business`, `Edu` -> `Education`, `housing` -> `Home`, `Car` -> `Auto`, `Personal`). |
| `existing_loans_count`| Replaced negative counts with `0`, clipped extreme outliers. |
| `city` | Mapped historical names (`Madras` -> `Chennai`, `Bombay` -> `Mumbai`, `Bangalore` -> `Bengaluru`, `Delhi` / `New Delhi` -> `Delhi`). |
| `loan_default` | Target binary variable: `0` (Paid / Non-Defaulter), `1` (Defaulted). Class distribution: ~58.4% non-defaulter, ~41.6% defaulter. |

---

## 🏆 Model Benchmarking & Performance

Evaluated on a 20% holdout test set with 5-fold Stratified Cross-Validation on the training set:

| Model | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **0.9008** | **81.25%** | **76.00%** | **80.12%** | **0.7801** | **0.8948** |
| Random Forest | 0.8921 | 81.00% | 75.57% | 80.12% | 0.7778 | 0.8892 |
| Gradient Boosting | 0.8882 | 81.00% | 77.11% | 77.11% | 0.7711 | 0.8811 |
| XGBoost | 0.8896 | 80.75% | 78.34% | 74.10% | 0.7616 | 0.8788 |
| LightGBM | 0.8897 | 80.25% | 76.05% | 76.51% | 0.7628 | 0.8742 |

### Key Risk Drivers (Top Predictors)
1. **Loan Amount (+2.66 Log-Odds)**: Higher loan amounts are the strongest driver of delinquency.
2. **Unemployed Status (+1.11 Log-Odds)**: Unemployed applicants carry high default risk.
3. **Monthly Income (-1.03 Log-Odds)**: Higher verified income strongly protects against default.
4. **Credit Score (-0.62 Log-Odds)**: Higher credit rating significantly reduces risk.
5. **Salaried Employment (-0.44 Log-Odds)**: Steady salaried employment correlates with consistent loan repayment.

---

## 🚀 Quickstart & Usage

### 1. Run the Interactive Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Python API Usage
```python
from predict import LoanDefaulterPredictor

# Initialize predictor
predictor = LoanDefaulterPredictor()

# Single applicant dictionary
applicant = {
    'age': 34,
    'gender': 'Male',
    'marital_status': 'Married',
    'dependents': 1,
    'employment_type': 'Salaried',
    'monthly_income': 65000,
    'credit_score': 740,
    'loan_amount': 250000,
    'loan_term_months': 36,
    'interest_rate': 9.5,
    'loan_purpose': 'Car',
    'existing_loans_count': 1,
    'city': 'Hyderabad'
}

result = predictor.predict_single(applicant)
print(result)
# Output:
# {
#   'prediction': 0,
#   'prediction_label': 'Non-Defaulter',
#   'default_probability_percent': 12.42,
#   'risk_grade': 'Low Risk (Approve)',
#   'risk_factors': ['Standard risk profile based on applicant parameters']
# }
```

### 3. Batch Scoring via Script
```python
import pandas as pd
from predict import LoanDefaulterPredictor

predictor = LoanDefaulterPredictor()
df_applicants = pd.read_csv("new_applications.csv")
scored_df = predictor.predict_dataframe(df_applicants)
scored_df.to_csv("scored_results.csv", index=False)
```

---

## 🧪 Running Unit Tests
```bash
python test_pipeline.py
```
