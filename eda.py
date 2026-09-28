import pandas as pd
import numpy as np

df = pd.read_csv('loan_data.csv')
print(f"Shape: {df.shape}")
print(f"Duplicate loan_ids: {df['loan_id'].str.strip().str.upper().duplicated().sum()}")
print(f"Total fully duplicated rows: {df.duplicated().sum()}")

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Target distribution ---")
print(df['loan_default'].value_counts(dropna=False, normalize=True) * 100)

print("\n--- Categorical columns inspection ---")
cat_cols = ['gender', 'marital_status', 'employment_type', 'loan_purpose', 'city']
for c in cat_cols:
    print(f"\nUnique values in {c}:")
    print(df[c].value_counts(dropna=False).head(20))

print("\n--- Numerical columns summary ---")
num_cols = ['age', 'dependents', 'monthly_income', 'credit_score', 'loan_amount', 'loan_term_months', 'interest_rate', 'existing_loans_count']
print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n--- Potential Anomalies Check ---")
print(f"Age < 18: {(df['age'] < 18).sum()}")
print(f"Age > 100: {(df['age'] > 100).sum()}")
print(f"Monthly Income <= 0: {(df['monthly_income'] <= 0).sum()}")
print(f"Monthly Income > 1,000,000: {(df['monthly_income'] > 1000000).sum()}")
print(f"Credit Score < 300: {(df['credit_score'] < 300).sum()}")
print(f"Credit Score > 900: {(df['credit_score'] > 900).sum()}")
print(f"Loan Amount <= 0: {(df['loan_amount'] <= 0).sum()}")
print(f"Loan Amount > 10,000,000: {(df['loan_amount'] > 10000000).sum()}")
print(f"Loan Term Months > 120: {(df['loan_term_months'] > 120).sum()}")
print(f"Interest Rate <= 0: {(df['interest_rate'] <= 0).sum()}")
print(f"Interest Rate > 40: {(df['interest_rate'] > 40).sum()}")
print(f"Dependents < 0: {(df['dependents'] < 0).sum()}")
print(f"Dependents > 10: {(df['dependents'] > 10).sum()}")
print(f"Existing loans < 0: {(df['existing_loans_count'] < 0).sum()}")
print(f"Existing loans > 10: {(df['existing_loans_count'] > 10).sum()}")
