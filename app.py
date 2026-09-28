import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from predict import LoanDefaulterPredictor

st.set_page_config(
    page_title="Loan Defaulter Prediction AI System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-safe {
        background-color: #d1fae5;
        color: #065f46;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .status-badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .status-badge-warn {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return LoanDefaulterPredictor('artifacts/loan_defaulter_model.joblib')

@st.cache_data
def load_cleaned_data():
    if os.path.exists('artifacts/cleaned_loan_data.csv'):
        return pd.read_csv('artifacts/cleaned_loan_data.csv')
    return None

try:
    predictor = load_predictor()
    bundle = predictor.bundle
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure clean_and_train.py has run successfully.")
    st.stop()

st.markdown('<div class="main-header">💳 Loan Defaulter Prediction & Underwriting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered credit risk assessment, anomaly detection, and automated underwriting platform</div>', unsafe_allow_html=True)

tabs = st.tabs(["🎯 Single Applicant Assessment", "📁 Batch CSV Scoring", "📊 Data Insights & EDA", "🧠 Model Benchmarks & Architecture"])

# ==========================================
# TAB 1: Single Applicant Assessment
# ==========================================
with tabs[0]:
    st.subheader("Applicant Information & Loan Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Personal Demographics")
        age = st.slider("Applicant Age", min_value=18, max_value=85, value=35, help="Valid age range: 18+")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
        dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4, 5, 6])
        city = st.selectbox("City", options=["Hyderabad", "Pune", "Chennai", "Delhi", "Bengaluru", "Mumbai"])
        
    with col2:
        st.markdown("##### 💼 Financial Background")
        employment_type = st.selectbox("Employment Type", options=["Salaried", "Self-Employed", "Retired", "Unemployed"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=5000.0, max_value=1000000.0, value=55000.0, step=2500.0)
        credit_score = st.slider("Credit Score (CIBIL / FICO)", min_value=300, max_value=900, value=720, help="Standard range: 300 to 900")
        existing_loans_count = st.selectbox("Existing Active Loans Count", options=[0, 1, 2, 3, 4, 5])
        
    with col3:
        st.markdown("##### 📄 Loan Request Details")
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000.0, max_value=5000000.0, value=250000.0, step=10000.0)
        loan_term_months = st.selectbox("Loan Term (Months)", options=[12, 24, 36, 48, 60, 84, 120], index=2)
        interest_rate = st.slider("Annual Interest Rate (%)", min_value=3.0, max_value=25.0, value=10.5, step=0.25)
        loan_purpose = st.selectbox("Loan Purpose", options=["Personal", "Business", "Education", "Auto", "Home"])

    # Computed financial indicators
    r = (interest_rate / 100.0) / 12.0
    n = loan_term_months
    emi = loan_amount * (r * ((1 + r)**n)) / (((1 + r)**n) - 1 + 1e-6)
    dti = (emi / monthly_income) * 100.0
    lti = loan_amount / (monthly_income * 12.0)

    st.markdown("---")
    st.markdown("##### 📐 Derived Financial Ratios (Pre-Assessment)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Estimated Monthly EMI", f"₹{emi:,.0f}")
    mcol2.metric("Debt-to-Income (DTI)", f"{dti:.1f}%", help="Recommended: below 40%")
    mcol3.metric("Loan-to-Annual Income", f"{lti:.2f}x", help="Lower is safer")
    mcol4.metric("Total Payable", f"₹{(emi * n):,.0f}")
    
    st.markdown("---")
    
    if st.button("🚀 Analyze Credit Risk & Predict Default", type="primary", use_container_width=True):
        applicant_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'dependents': dependents,
            'employment_type': employment_type,
            'monthly_income': monthly_income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_term_months': loan_term_months,
            'interest_rate': interest_rate,
            'loan_purpose': loan_purpose,
            'existing_loans_count': existing_loans_count,
            'city': city
        }
        
        result = predictor.predict_single(applicant_data)
        prob = float(result['default_probability_percent'])
        is_default = result['prediction'] == 1
        
        rcol1, rcol2 = st.columns([1, 1])
        
        with rcol1:
            st.markdown("### 📋 Prediction Decision")
            if is_default:
                if prob >= 75:
                    st.markdown('<div class="status-badge-danger">⚠️ HIGH RISK - LIKELY TO DEFAULT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-badge-warn">⚠️ MODERATE RISK - LIKELY TO DEFAULT</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-safe">✅ LOW RISK - NON-DEFAULTER (APPROVED)</div>', unsafe_allow_html=True)
            
            st.write("")
            st.metric("Estimated Default Probability", f"{prob:.1f}%")
            st.metric("Underwriting Risk Grade", result['risk_grade'])
            st.progress(prob / 100.0)

        with rcol2:
            st.markdown("### 🔍 Risk Factor Diagnostics")
            for reason in result['risk_factors']:
                st.info(f"📌 {reason}")
                
            if dti > 50:
                st.warning(f"⚠️ High Debt-to-Income burden ({dti:.1f}% of monthly income dedicated to EMI).")
            if credit_score < 620:
                st.warning("⚠️ Below-average credit score raises probability of delinquency.")
            if employment_type == 'Unemployed':
                st.error("🚨 Zero verified employment income presents critical credit hazard.")
            if is_default == 0 and prob < 25:
                st.success("🌟 Healthy financial leverage and positive credit rating profile.")

# ==========================================
# TAB 2: Batch CSV Scoring
# ==========================================
with tabs[1]:
    st.subheader("Batch Loan Defaulter Prediction via CSV Upload")
    st.write("Upload a CSV file containing loan applicant records with column headers matching the dataset schema.")
    
    # Download sample template
    df_clean = load_cleaned_data()
    if df_clean is not None:
        sample_csv = df_clean.head(5).drop(columns=['loan_default'], errors='ignore').to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Input Template (CSV)",
            data=sample_csv,
            file_name="sample_loan_applicants.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload Applicant CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(df_batch)} rows successfully.")
            
            with st.spinner("Scoring applicants and evaluating credit risks..."):
                df_scored = predictor.predict_dataframe(df_batch)
                
            st.markdown("### 📈 Batch Prediction Summary")
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            total = len(df_scored)
            defaults = (df_scored['default_prediction'] == 1).sum()
            non_defaults = total - defaults
            avg_prob = df_scored['default_probability'].mean()
            
            bcol1.metric("Total Applications", f"{total:,}")
            bcol2.metric("Approved (Non-Defaulters)", f"{non_defaults:,} ({(non_defaults/total)*100:.1f}%)")
            bcol3.metric("Flagged Defaulters", f"{defaults:,} ({(defaults/total)*100:.1f}%)")
            bcol4.metric("Average Portfolio Risk", f"{avg_prob:.1f}%")
            
            st.dataframe(df_scored[['loan_id', 'age', 'monthly_income', 'credit_score', 'loan_amount', 'default_prediction', 'default_probability', 'risk_grade']].head(50), use_container_width=True)
            
            csv_export = df_scored.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Scored Batch CSV",
                data=csv_export,
                file_name="scored_loan_predictions.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as ex:
            st.error(f"Error processing file: {ex}")

# ==========================================
# TAB 3: Data Insights & EDA
# ==========================================
with tabs[2]:
    st.subheader("Training Dataset Insights & Key Drivers")
    df_data = load_cleaned_data()
    
    if df_data is not None:
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Cleaned Records", f"{len(df_data):,}")
        col_stat2.metric("Default Rate", f"{(df_data['loan_default'].mean()*100):.1f}%")
        col_stat3.metric("Median Income", f"₹{df_data['monthly_income'].median():,.0f}")
        col_stat4.metric("Median Loan Amount", f"₹{df_data['loan_amount'].median():,.0f}")
        
        st.write("")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Default Rate by Employment Type")
            emp_rates = df_data.groupby('employment_type')['loan_default'].mean().reset_index()
            emp_rates['Default Rate (%)'] = emp_rates['loan_default'] * 100
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=emp_rates, x='employment_type', y='Default Rate (%)', palette='Blues_r', ax=ax1)
            ax1.set_ylabel("Default Rate (%)")
            ax1.set_xlabel("Employment Status")
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
            
        with c2:
            st.markdown("##### Default Rate by Loan Purpose")
            purp_rates = df_data.groupby('loan_purpose')['loan_default'].mean().reset_index()
            purp_rates['Default Rate (%)'] = purp_rates['loan_default'] * 100
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=purp_rates, x='loan_purpose', y='Default Rate (%)', palette='Purples_r', ax=ax2)
            ax2.set_ylabel("Default Rate (%)")
            ax2.set_xlabel("Loan Purpose")
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### Credit Score Distribution by Default Status")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df_data, x='credit_score', hue='loan_default', kde=True, bins=25, ax=ax3, palette=['#10b981', '#ef4444'])
            ax3.set_xlabel("Credit Score")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
            
        with c4:
            st.markdown("##### Loan Amount vs Monthly Income")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df_data.sample(min(400, len(df_data)), random_state=42),
                x='monthly_income', y='loan_amount', hue='loan_default', alpha=0.7,
                palette=['#10b981', '#ef4444'], ax=ax4
            )
            ax4.set_xlabel("Monthly Income (₹)")
            ax4.set_ylabel("Loan Amount (₹)")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

# ==========================================
# TAB 4: Model Benchmarks & Architecture
# ==========================================
with tabs[3]:
    st.subheader("Model Benchmark & Validation Performance")
    
    if 'results' in bundle:
        df_bench = pd.DataFrame(bundle['results']).T
        df_bench = df_bench[['CV_ROC_AUC_Mean', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC']]
        df_bench.columns = ['CV ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Test ROC-AUC']
        df_bench = df_bench.round(4)
        st.dataframe(df_bench, use_container_width=True)
        
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("##### Confusion Matrix (Holdout Test Set)")
        if os.path.exists('artifacts/confusion_matrix.png'):
            st.image('artifacts/confusion_matrix.png', caption=f"Confusion Matrix: {bundle.get('model_name', 'Model')}")
            
    with pcol2:
        st.markdown("##### ROC Curve Comparison across Models")
        if os.path.exists('artifacts/roc_comparison.png'):
            st.image('artifacts/roc_comparison.png', caption="ROC Curve Performance across all 5 benchmarked classifiers")
            
    st.markdown("##### Feature Importance & Predictor Impact")
    if os.path.exists('artifacts/feature_importance.png'):
        st.image('artifacts/feature_importance.png', caption="Top 15 Predictor Impacts on Loan Default")
