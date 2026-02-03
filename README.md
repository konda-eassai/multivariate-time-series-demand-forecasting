# Multivariate Time-Series Demand Forecasting using LSTM & GRU

## Project Overview
This project implements an end-to-end **multivariate time-series forecasting system**
to predict retail demand using classical Machine Learning and Deep Learning models.
The solution focuses on learning temporal dependencies and translating forecasts
into actionable inventory risk insights.

## Problem Statement
Retail demand is influenced by seasonality, holidays, store characteristics,
and external economic factors. Traditional forecasting approaches often fail
to capture these complex temporal relationships, leading to stock-outs or overstocking.

## Dataset
- Walmart Store Sales Forecasting Dataset (Kaggle)
- Weekly sales data with store, department, holiday, and external indicators

## Modeling Approach
- **Baseline ML Models**: Linear Regression, Random Forest
- **Deep Learning Models**: LSTM, GRU
- Time-aware train-test split to avoid data leakage
- Evaluation using RMSE and MAPE

## Key Results
- Deep learning models outperform classical ML baselines
- GRU achieves comparable or better performance with lower complexity
- Forecast outputs are translated into stock-out and overstock risk signals

## Deployment
- Interactive Streamlit dashboard for model comparison and demand forecasting
- Displays forecasts, error metrics, and inventory risk classification

## Tools & Technologies
- Python, Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Matplotlib, Seaborn
