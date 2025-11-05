"""
Housing Price Prediction using OLS Regression
Built with statsmodels.api and statsmodels.formula.api
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

class HousingPricePrediction:
    """OLS regression analysis for housing data."""
    
    def __init__(self, data_path=None):
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = self._generate_synthetic_data()
        self.model = None
        self.results = None
    
    def _generate_synthetic_data(self, n_samples=200):
        np.random.seed(42)
        sq_ft = np.random.uniform(800, 5000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.uniform(1, 4, n_samples)
        age = np.random.uniform(0, 100, n_samples)
        
        price = (50 * sq_ft + 30000 * bedrooms + 25000 * bathrooms - 
                500 * age + np.random.normal(0, 50000, n_samples))
        
        return pd.DataFrame({
            'Price': price,
            'SquareFeet': sq_ft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age': age
        })
    
    def fit_model(self, formula='Price ~ SquareFeet + Bedrooms + Bathrooms + Age'):
        self.model = smf.ols(formula, data=self.data)
        self.results = self.model.fit()
        return self.results
    
    def get_summary(self):
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit_model() first.")
        return self.results.summary()
    
    def get_coefficients(self):
        if self.results is None:
            raise ValueError("Model not fitted yet.")
        return pd.DataFrame({
            'Coefficient': self.results.params,
            'Std Error': self.results.bse,
            't-statistic': self.results.tvalues,
            'p-value': self.results.pvalues
        })
    
    def model_performance_metrics(self):
        if self.results is None:
            raise ValueError("Model not fitted yet.")
        y_actual = self.data['Price']
        y_pred = self.results.fittedvalues
        return pd.Series({
            'R-squared': self.results.rsquared,
            'Adjusted R-squared': self.results.rsquared_adj,
            'RMSE': np.sqrt(mean_squared_error(y_actual, y_pred)),
            'MAE': mean_absolute_error(y_actual, y_pred)
        })

def main():
    print("="*70)
    print("HOUSING PRICE PREDICTION WITH OLS REGRESSION")
    print("="*70)
    
    hp = HousingPricePrediction()
    hp.fit_model()
    
    print("\n" + "="*70)
    print("OLS REGRESSION RESULTS")
    print("="*70)
    print(hp.get_summary())
    
    print("\n" + "="*70)
    print("COEFFICIENTS")
    print("="*70)
    print(hp.get_coefficients())
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    print(hp.model_performance_metrics())
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
