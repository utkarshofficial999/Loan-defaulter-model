import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_categoricals(df):
    """Clean and standardize messy categorical values."""
    df = df.copy()
    
    # 1. Gender
    gender_map = {
        'MALE': 'Male', 'Male': 'Male', 'male': 'Male', 'M': 'Male', 'm': 'Male',
        'FEMALE': 'Female', 'Female': 'Female', 'female': 'Female', 'F': 'Female', 'f': 'Female'
    }
    df['gender'] = df['gender'].astype(str).str.strip().map(gender_map)
    # Fill remaining/unknown with most frequent or 'Unknown'
    df['gender'] = df['gender'].fillna('Male')
    
    # 2. Marital Status
    marital_map = {
        'SINGLE': 'Single', 'Single': 'Single', 'single': 'Single',
        'MARRIED': 'Married', 'Married': 'Married', 'married': 'Married', 'Marreid': 'Married',
        'DIVORCED': 'Divorced', 'Divorced': 'Divorced', 'divorced': 'Divorced'
    }
    df['marital_status'] = df['marital_status'].astype(str).str.strip().map(marital_map)
    df['marital_status'] = df['marital_status'].fillna('Single')
    
    # 3. Employment Type
    def clean_emp(val):
        if pd.isna(val):
            return 'Salaried'
        v = str(val).strip().lower()
        if 'salar' in v:
            return 'Salaried'
        if 'self' in v:
            return 'Self-Employed'
        if 'retir' in v:
            return 'Retired'
        if 'unemploy' in v:
            return 'Unemployed'
        return 'Salaried'
    
    df['employment_type'] = df['employment_type'].apply(clean_emp)
    
    # 4. Loan Purpose
    def clean_purpose(val):
        if pd.isna(val):
            return 'Personal'
        v = str(val).strip().lower()
        if 'auto' in v or 'car' in v:
            return 'Auto'
        if 'biz' in v or 'business' in v:
            return 'Business'
        if 'edu' in v:
            return 'Education'
        if 'home' in v or 'hous' in v:
            return 'Home'
        if 'person' in v:
            return 'Personal'
        return 'Personal'
        
    df['loan_purpose'] = df['loan_purpose'].apply(clean_purpose)
    
    # 5. City
    def clean_city(val):
        if pd.isna(val):
            return 'Hyderabad'
        v = str(val).strip().lower()
        if 'hyderabad' in v:
            return 'Hyderabad'
        if 'pune' in v:
            return 'Pune'
        if 'chennai' in v or 'madras' in v:
            return 'Chennai'
        if 'delhi' in v:
            return 'Delhi'
        if 'bangalore' in v or 'bengaluru' in v:
            return 'Bengaluru'
        if 'mumbai' in v or 'bombay' in v:
            return 'Mumbai'
        return 'Hyderabad'
        
    df['city'] = df['city'].apply(clean_city)
    
    return df

def clean_numerics_and_anomalies(df):
    """Handle corrupted numerical entries and domain constraints."""
    df = df.copy()
    
    # Clean loan_term_months: 1200 is clearly a typo for 120
    df.loc[df['loan_term_months'] == 1200, 'loan_term_months'] = 120
    # Restrict to reasonable valid terms: (12, 24, 36, 48, 60, 84, 120)
    df['loan_term_months'] = df['loan_term_months'].clip(12, 120)
    
    # Interest rate: <=0 is impossible, >50 likely typo (e.g. 150 -> 15.0)
    df.loc[df['interest_rate'] > 50, 'interest_rate'] = df.loc[df['interest_rate'] > 50, 'interest_rate'] / 10.0
    df.loc[df['interest_rate'] <= 0, 'interest_rate'] = np.nan
    
    # Age: < 18 or > 100 is invalid for loan applicant
    df.loc[(df['age'] < 18) | (df['age'] > 90), 'age'] = np.nan
    
    # Credit Score: 300 to 900
    df.loc[(df['credit_score'] < 300) | (df['credit_score'] > 900), 'credit_score'] = np.nan
    
    # Monthly Income: <= 0 or extreme billions
    df.loc[(df['monthly_income'] <= 0) | (df['monthly_income'] > 1000000), 'monthly_income'] = np.nan
    
    # Loan Amount: <= 0 or > 10M
    df.loc[(df['loan_amount'] <= 0) | (df['loan_amount'] > 10000000), 'loan_amount'] = np.nan
    
    # Dependents: clip to 0 - 8
    df['dependents'] = df['dependents'].clip(0, 8)
    
    # Existing Loans Count: clip to 0 - 10
    df['existing_loans_count'] = df['existing_loans_count'].clip(0, 10)
    
    return df

def feature_engineering(df):
    """Engineer financial ratios and temporal features."""
    df = df.copy()
    
    # Date features
    dates = pd.to_datetime(df['application_date'], format='mixed', errors='coerce')
    # Default year/month if parse fails
    df['app_year'] = dates.dt.year.fillna(2020).astype(int)
    df['app_month'] = dates.dt.month.fillna(6).astype(int)
    df['app_dayofweek'] = dates.dt.dayofweek.fillna(2).astype(int)
    
    # Monthly interest rate proxy
    r = (df['interest_rate'].fillna(11.0) / 100.0) / 12.0
    n = df['loan_term_months']
    P = df['loan_amount'].fillna(200000.0)
    
    # Standard EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
    # Using linear approximation if r == 0: P / n
    emi = P * (r * ((1 + r)**n)) / (((1 + r)**n) - 1 + 1e-6)
    df['estimated_emi'] = emi
    
    # Debt to Income Ratio (EMI / Monthly Income)
    inc = df['monthly_income'].fillna(48000.0).clip(lower=1000.0)
    df['dti_ratio'] = (df['estimated_emi'] / inc).clip(0, 10)
    
    # Loan to Annual Income Ratio
    df['loan_to_income'] = (P / (inc * 12)).clip(0, 20)
    
    # Total Interest Amount estimated over term
    total_payment = emi * n
    df['total_interest_payable'] = (total_payment - P).clip(lower=0)
    
    # Financial burden per dependent
    df['income_per_dependent'] = inc / (df['dependents'] + 1)
    
    # Credit score risk category
    cs = df['credit_score'].fillna(660.0)
    df['credit_risk_tier'] = pd.cut(
        cs,
        bins=[0, 580, 670, 740, 800, 1000],
        labels=['Very_Poor', 'Fair', 'Good', 'Very_Good', 'Exceptional'],
        right=False
    ).astype(str)
    
    return df

def run_pipeline():
    print("=" * 60)
    print("STARTING LOAN DEFAULTER MODEL PIPELINE")
    print("=" * 60)
    
    df_raw = pd.read_csv('loan_data.csv')
    print(f"Raw dataset shape: {df_raw.shape}")
    
    # Remove exact duplicate rows
    df = df_raw.drop_duplicates().copy()
    # Remove rows where loan_id is duplicate (keep first)
    df['clean_id'] = df['loan_id'].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=['clean_id']).drop(columns=['clean_id'])
    print(f"Shape after removing duplicates: {df.shape}")
    
    # Ensure target is clean int (0 or 1)
    df = df[df['loan_default'].isin([0, 1])].copy()
    df['loan_default'] = df['loan_default'].astype(int)
    
    # Standardize categoricals
    df = standardize_categoricals(df)
    
    # Clean numeric anomalies
    df = clean_numerics_and_anomalies(df)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Define features to use
    feature_cols = [
        'age', 'gender', 'marital_status', 'dependents', 'employment_type',
        'monthly_income', 'credit_score', 'loan_amount', 'loan_term_months',
        'interest_rate', 'loan_purpose', 'existing_loans_count', 'city',
        'app_year', 'app_month', 'app_dayofweek', 'estimated_emi',
        'dti_ratio', 'loan_to_income', 'total_interest_payable',
        'income_per_dependent', 'credit_risk_tier'
    ]
    
    X = df[feature_cols].copy()
    y = df['loan_default'].copy()
    
    print("\nTarget Class Distribution:")
    print(y.value_counts(normalize=True).apply(lambda v: f"{v*100:.2f}%"))
    
    # Strictly split train, test BEFORE fitting preprocessing pipeline (ML Best Practice)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Preprocessor definition
    num_cols = [
        'age', 'dependents', 'monthly_income', 'credit_score', 'loan_amount',
        'loan_term_months', 'interest_rate', 'existing_loans_count',
        'app_year', 'app_month', 'app_dayofweek', 'estimated_emi',
        'dti_ratio', 'loan_to_income', 'total_interest_payable',
        'income_per_dependent'
    ]
    
    cat_cols = [
        'gender', 'marital_status', 'employment_type', 'loan_purpose',
        'city', 'credit_risk_tier'
    ]
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, verbose=-1)
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("\n--- Training and Evaluating Models ---")
    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # 5-fold cross validation on training set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
        
        # Fit on full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predict on Test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        results[name] = {
            'CV_ROC_AUC_Mean': cv_scores.mean(),
            'CV_ROC_AUC_Std': cv_scores.std(),
            'Test_Accuracy': acc,
            'Test_Precision': prec,
            'Test_Recall': rec,
            'Test_F1': f1,
            'Test_ROC_AUC': roc_auc,
            'Test_PR_AUC': pr_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"[{name}]")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Find best model based on ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]['Test_ROC_AUC'])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n>>> Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['Test_ROC_AUC']:.4f}, F1: {results[best_model_name]['Test_F1']:.4f})")
    
    # Print Detailed Classification Report for Best Model
    print("\nClassification Report for Best Model on Test Set:")
    print(classification_report(y_test, results[best_model_name]['y_pred'], target_names=['Non-Defaulter (0)', 'Defaulter (1)']))
    
    # Save confusion matrix plot
    os.makedirs('artifacts', exist_ok=True)
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0 (Paid)', 'Pred 1 (Default)'],
                yticklabels=['Actual 0 (Paid)', 'Actual 1 (Default)'])
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Save ROC Curve Comparison
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 6))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['Test_ROC_AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/roc_comparison.png', dpi=300)
    plt.close()
    
    # Feature Importance for Tree-based models (if applicable)
    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        feat_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        feat_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title(f'Top 15 Feature Importances ({best_model_name})')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('artifacts/feature_importance.png', dpi=300)
        plt.close()
        print("\nTop 10 Most Important Features:")
        print(feat_df.head(10).to_string(index=False))
    
    # Save Model Bundle and Metadata
    bundle = {
        'model_name': best_model_name,
        'pipeline': best_pipeline,
        'feature_cols': feature_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'results': {k: {m: v for m, v in metrics.items() if not m.startswith('y_')} for k, metrics in results.items()}
    }
    joblib.dump(bundle, 'artifacts/loan_defaulter_model.joblib')
    print("\nModel artifact saved successfully to 'artifacts/loan_defaulter_model.joblib'")
    
    # Also save cleaned dataset for exploration / demo
    df.to_csv('artifacts/cleaned_loan_data.csv', index=False)
    print("Cleaned dataset saved to 'artifacts/cleaned_loan_data.csv'")

if __name__ == '__main__':
    run_pipeline()
