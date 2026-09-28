import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

bundle = joblib.load('artifacts/loan_defaulter_model.joblib')
pipe = bundle['pipeline']
clf = pipe.named_steps['classifier']
prep = pipe.named_steps['preprocessor']
feat_names = prep.get_feature_names_out()

if hasattr(clf, 'coef_'):
    coefs = clf.coef_[0]
    df_imp = pd.DataFrame({'feature': feat_names, 'coefficient': coefs, 'abs_coef': np.abs(coefs)})
    df_imp = df_imp.sort_values('abs_coef', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    colors = ['#ef4444' if c > 0 else '#10b981' for c in df_imp['coefficient']]
    sns.barplot(data=df_imp, x='coefficient', y='feature', palette=colors)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Top 15 Predictor Impacts on Loan Default (Red = Increases Default, Green = Decreases Default)', fontsize=12, fontweight='bold')
    plt.xlabel('Logistic Regression Coefficient (Log-Odds Impact)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=300)
    plt.close()
    print("Feature impact plot saved to artifacts/feature_importance.png")
