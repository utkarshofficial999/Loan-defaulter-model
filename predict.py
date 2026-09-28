import os
import joblib
import pandas as pd
import numpy as np

def standardize_record(df):
    """Clean and standardize input records."""
    df = df.copy()
    
    # Gender
    gender_map = {
        'MALE': 'Male', 'Male': 'Male', 'male': 'Male', 'M': 'Male', 'm': 'Male',
        'FEMALE': 'Female', 'Female': 'Female', 'female': 'Female', 'F': 'Female', 'f': 'Female'
    }
    df['gender'] = df['gender'].astype(str).str.strip().map(gender_map).fillna('Male')
    
    # Marital status
    marital_map = {
        'SINGLE': 'Single', 'Single': 'Single', 'single': 'Single',
        'MARRIED': 'Married', 'Married': 'Married', 'married': 'Married', 'Marreid': 'Married',
        'DIVORCED': 'Divorced', 'Divorced': 'Divorced', 'divorced': 'Divorced'
    }
    df['marital_status'] = df['marital_status'].astype(str).str.strip().map(marital_map).fillna('Single')
    
    # Employment type
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
    
    # Purpose
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
        return 'Personal'
        
    df['loan_purpose'] = df['loan_purpose'].apply(clean_purpose)
    
    # City
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
    
    # Numerics
    df['loan_term_months'] = pd.to_numeric(df['loan_term_months'], errors='coerce').fillna(36)
    df.loc[df['loan_term_months'] == 1200, 'loan_term_months'] = 120
    df['loan_term_months'] = df['loan_term_months'].clip(12, 120)
    
    df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce')
    df.loc[df['interest_rate'] > 50, 'interest_rate'] = df.loc[df['interest_rate'] > 50, 'interest_rate'] / 10.0
    df.loc[df['interest_rate'] <= 0, 'interest_rate'] = np.nan
    
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[(df['age'] < 18) | (df['age'] > 90), 'age'] = np.nan
    
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df.loc[(df['credit_score'] < 300) | (df['credit_score'] > 900), 'credit_score'] = np.nan
    
    df['monthly_income'] = pd.to_numeric(df['monthly_income'], errors='coerce')
    df.loc[(df['monthly_income'] <= 0) | (df['monthly_income'] > 1000000), 'monthly_income'] = np.nan
    
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df.loc[(df['loan_amount'] <= 0) | (df['loan_amount'] > 10000000), 'loan_amount'] = np.nan
    
    df['dependents'] = pd.to_numeric(df['dependents'], errors='coerce').fillna(0).clip(0, 8)
    df['existing_loans_count'] = pd.to_numeric(df['existing_loans_count'], errors='coerce').fillna(0).clip(0, 10)
    
    # Feature engineering
    if 'application_date' in df.columns:
        dates = pd.to_datetime(df['application_date'], format='mixed', errors='coerce')
        df['app_year'] = dates.dt.year.fillna(2023).astype(int)
        df['app_month'] = dates.dt.month.fillna(6).astype(int)
        df['app_dayofweek'] = dates.dt.dayofweek.fillna(2).astype(int)
    else:
        df['app_year'] = 2023
        df['app_month'] = 6
        df['app_dayofweek'] = 2
        
    r = (df['interest_rate'].fillna(11.0) / 100.0) / 12.0
    n = df['loan_term_months']
    P = df['loan_amount'].fillna(200000.0)
    emi = P * (r * ((1 + r)**n)) / (((1 + r)**n) - 1 + 1e-6)
    df['estimated_emi'] = emi
    
    inc = df['monthly_income'].fillna(48000.0).clip(lower=1000.0)
    df['dti_ratio'] = (df['estimated_emi'] / inc).clip(0, 10)
    df['loan_to_income'] = (P / (inc * 12)).clip(0, 20)
    df['total_interest_payable'] = (emi * n - P).clip(lower=0)
    df['income_per_dependent'] = inc / (df['dependents'] + 1)
    
    cs = df['credit_score'].fillna(660.0)
    df['credit_risk_tier'] = pd.cut(
        cs,
        bins=[0, 580, 670, 740, 800, 1000],
        labels=['Very_Poor', 'Fair', 'Good', 'Very_Good', 'Exceptional'],
        right=False
    ).astype(str)
    
    return df

class LoanDefaulterPredictor:
    def __init__(self, model_path='artifacts/loan_defaulter_model.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run clean_and_train.py first.")
        self.bundle = joblib.load(model_path)
        self.pipeline = self.bundle['pipeline']
        self.feature_cols = self.bundle['feature_cols']
        self.model_name = self.bundle.get('model_name', 'Trained Model')
        
    def predict_dataframe(self, df_input):
        """Clean, engineer features and predict for a pandas DataFrame."""
        df_clean = standardize_record(df_input)
        X = df_clean[self.feature_cols]
        
        preds = self.pipeline.predict(X)
        probs = self.pipeline.predict_proba(X)[:, 1]
        
        df_result = df_input.copy()
        df_result['default_prediction'] = preds
        df_result['default_probability'] = np.round(probs * 100, 2)
        
        def assign_risk(p):
            if p < 25:
                return 'Low Risk (Approve)'
            elif p < 50:
                return 'Moderate Risk (Review)'
            elif p < 75:
                return 'High Risk (Cautious)'
            else:
                return 'Critical Risk (Reject)'
                
        df_result['risk_grade'] = [assign_risk(p) for p in df_result['default_probability']]
        return df_result
        
    def predict_single(self, applicant_dict):
        """Predict for a single applicant dictionary."""
        df_single = pd.DataFrame([applicant_dict])
        res = self.predict_dataframe(df_single).iloc[0]
        
        prob = res['default_probability']
        pred = int(res['default_prediction'])
        grade = res['risk_grade']
        
        # Determine key risk reasons
        reasons = []
        if applicant_dict.get('employment_type', '').lower() == 'unemployed':
            reasons.append("Applicant is currently Unemployed")
        cs = float(applicant_dict.get('credit_score', 650) or 650)
        if cs < 600:
            reasons.append(f"Low Credit Score ({cs:.0f})")
        amt = float(applicant_dict.get('loan_amount', 0) or 0)
        inc = float(applicant_dict.get('monthly_income', 50000) or 50000)
        if inc > 0 and (amt / (inc * 12)) > 3.0:
            reasons.append(f"High Loan-to-Annual-Income ratio ({(amt / (inc * 12)):.1f}x)")
        
        return {
            'prediction': pred,
            'prediction_label': 'Defaulter' if pred == 1 else 'Non-Defaulter',
            'default_probability_percent': prob,
            'risk_grade': grade,
            'risk_factors': reasons if reasons else ['Standard risk profile based on applicant parameters']
        }

if __name__ == '__main__':
    predictor = LoanDefaulterPredictor()
    sample = {
        'age': 35,
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
    result = predictor.predict_single(sample)
    print("Sample Applicant Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
