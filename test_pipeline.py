import unittest
import pandas as pd
import numpy as np
from predict import LoanDefaulterPredictor, standardize_record

class TestLoanDefaulterModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = LoanDefaulterPredictor('artifacts/loan_defaulter_model.joblib')

    def test_standardize_record_dirty_inputs(self):
        dirty_df = pd.DataFrame([{
            'loan_id': '  ln12345  ',
            'application_date': '30-Mar-2022',
            'age': 999.0,                      # corrupted age
            'gender': 'MALE',                  # uppercase
            'marital_status': 'Marreid',       # typo
            'dependents': 12,                  # extreme
            'employment_type': 'unemployed',   # lowercase
            'monthly_income': -5000.0,         # negative income
            'credit_score': 9999.0,            # corrupted credit score
            'loan_amount': 999999999.0,        # extreme loan amount
            'loan_term_months': 1200,          # typo 1200
            'interest_rate': 150.0,            # typo 150
            'loan_purpose': 'biz',             # synonym
            'existing_loans_count': -1,        # negative count
            'city': 'Madras'                   # historical synonym for Chennai
        }])
        
        cleaned = standardize_record(dirty_df)
        self.assertEqual(cleaned['gender'].iloc[0], 'Male')
        self.assertEqual(cleaned['marital_status'].iloc[0], 'Married')
        self.assertEqual(cleaned['employment_type'].iloc[0], 'Unemployed')
        self.assertEqual(cleaned['loan_purpose'].iloc[0], 'Business')
        self.assertEqual(cleaned['city'].iloc[0], 'Chennai')
        self.assertEqual(cleaned['loan_term_months'].iloc[0], 120)
        self.assertEqual(cleaned['existing_loans_count'].iloc[0], 0)
        self.assertTrue(np.isnan(cleaned['age'].iloc[0]))
        self.assertTrue(np.isnan(cleaned['monthly_income'].iloc[0]))
        self.assertTrue(np.isnan(cleaned['credit_score'].iloc[0]))
        self.assertTrue(np.isnan(cleaned['loan_amount'].iloc[0]))

    def test_single_prediction_output(self):
        sample = {
            'age': 40,
            'gender': 'Female',
            'marital_status': 'Married',
            'dependents': 2,
            'employment_type': 'Salaried',
            'monthly_income': 75000,
            'credit_score': 760,
            'loan_amount': 200000,
            'loan_term_months': 36,
            'interest_rate': 8.5,
            'loan_purpose': 'Home',
            'existing_loans_count': 1,
            'city': 'Bengaluru'
        }
        res = self.predictor.predict_single(sample)
        self.assertIn(res['prediction'], [0, 1])
        self.assertTrue(0.0 <= res['default_probability_percent'] <= 100.0)
        self.assertIn('risk_grade', res)
        self.assertIsInstance(res['risk_factors'], list)

    def test_batch_prediction_output(self):
        sample_df = pd.DataFrame([
            {
                'loan_id': 'LN1',
                'age': 40,
                'gender': 'Male',
                'marital_status': 'Married',
                'dependents': 1,
                'employment_type': 'Salaried',
                'monthly_income': 85000,
                'credit_score': 780,
                'loan_amount': 150000,
                'loan_term_months': 24,
                'interest_rate': 8.0,
                'loan_purpose': 'Car',
                'existing_loans_count': 0,
                'city': 'Mumbai'
            },
            {
                'loan_id': 'LN2',
                'age': 22,
                'gender': 'Female',
                'marital_status': 'Single',
                'dependents': 3,
                'employment_type': 'Unemployed',
                'monthly_income': 15000,
                'credit_score': 450,
                'loan_amount': 700000,
                'loan_term_months': 60,
                'interest_rate': 16.0,
                'loan_purpose': 'Personal',
                'existing_loans_count': 4,
                'city': 'Delhi'
            }
        ])
        scored = self.predictor.predict_dataframe(sample_df)
        self.assertEqual(len(scored), 2)
        self.assertIn('default_prediction', scored.columns)
        self.assertIn('default_probability', scored.columns)
        self.assertIn('risk_grade', scored.columns)
        # LN1 should have lower default probability than LN2
        self.assertLess(scored['default_probability'].iloc[0], scored['default_probability'].iloc[1])

if __name__ == '__main__':
    unittest.main()
