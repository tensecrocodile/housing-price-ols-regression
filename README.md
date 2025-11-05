# Housing Price Prediction with OLS Regression

A comprehensive Python project demonstrating **Ordinary Least Squares (OLS) regression** using **statsmodels** for housing price prediction, statistical inference, and model diagnostics.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Understanding OLS Regression](#understanding-ols-regression)
- [How to Use](#how-to-use)
- [Project Output](#project-output)
- [Learning Resources](#learning-resources)

## 🎯 Overview

This project demonstrates core statistical concepts in regression analysis using Python's **statsmodels** library. It uses real estate data to predict housing prices while emphasizing understanding over predictions.

**Key Concepts Covered:**
- Ordinary Least Squares (OLS) regression
- Statistical hypothesis testing (p-values)
- Confidence intervals for parameters
- Residual diagnostics and model validation
- Multicollinearity detection (VIF)
- Model performance metrics (R², RMSE, MAE)

## ✨ Features

✅ **OLS Regression Model** - Linear regression using statsmodels  
✅ **Statistical Summary** - Detailed coefficients, p-values, and confidence intervals  
✅ **Diagnostic Plots** - Q-Q plots, residual analysis, scale-location plots  
✅ **Performance Metrics** - R², Adjusted R², AIC, BIC, RMSE, MAE, MAPE  
✅ **Multicollinearity Check** - Variance Inflation Factor (VIF) calculation  
✅ **Synthetic Data Generation** - Built-in realistic housing dataset generator  
✅ **Publication-Ready Visualizations** - High-quality diagnostic and prediction plots

## 📋 Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## 🚀 Installation

### Clone the repository:
```bash
git clone https://github.com/tensecrocodile/housing-price-ols-regression.git
cd housing-price-ols-regression
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎶 Quick Start

### Run the main analysis:
```bash
python analysis.py
```

This will:
1. Generate synthetic housing data
2. Fit an OLS regression model
3. Display regression summary with coefficients and p-values
4. Calculate performance metrics
5. Generate diagnostic plots (saved as PNG files)

## 📂 Project Structure

```
housing-price-ols-regression/
├── analysis.py                 # Main analysis script with HousingPricePrediction class
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── diagnostics.png             # Generated diagnostic plots
└── predictions_vs_actual.png   # Generated prediction plots
```

## 📚 Understanding OLS Regression

### What is OLS?

**Ordinary Least Squares (OLS)** is a statistical method for finding the best-fitting line through data by minimizing the sum of squared residuals (errors).

### Key Components Explained:

#### 1. **Coefficients (β)**
- Estimated parameters of the linear model
- **Example:** β1 = 50 means for every additional square foot, price increases by $50

#### 2. **R-squared (R²)**
- Measure of model fit: percentage of variance explained
- Range: 0 to 1 (higher is better)
- **R² = 0.95** means the model explains 95% of price variation

#### 3. **P-values**
- Probability that coefficient is actually zero
- **p < 0.05**: Strong evidence that variable matters
- **p > 0.05**: Weak evidence; variable may not be important

#### 4. **Confidence Intervals**
- Range of values where true coefficient likely lies (95% confidence)
- **[40, 60]** means we're 95% confident true effect is between 40 and 60

#### 5. **Residuals**
- Differences between actual and predicted values
- Used to check if model assumptions are met
- Should be normally distributed with mean = 0

## 🔓 How to Use

### Basic Usage:

```python
from analysis import HousingPricePrediction

# Initialize with synthetic data
hp = HousingPricePrediction()

# Fit the model
hp.fit_model()

# View detailed results
print(hp.get_summary())

# Get coefficients with confidence intervals
print(hp.get_coefficients())

# Check for multicollinearity
print(hp.calculate_vif())

# Generate diagnostic plots
fig = hp.plot_diagnostics()
```

## 📄 Learning Resources

- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Understanding OLS Regression](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [Statistical Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [Residual Diagnostics](https://en.wikipedia.org/wiki/Errors_and_residuals)

---

**Created with ♥ for learning statistics and regression analysis**
