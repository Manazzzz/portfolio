# 📈 Reliance Stock Price Forecasting

## Overview
This project focuses on forecasting stock prices of Reliance Industries Ltd. using historical data and machine learning models.

## Dataset
- 10 years of historical stock data
- Features: Date, Open, High, Low, Price, Volume, Change%
- Target: Closing Price (cleaned from Price column)

## Feature Engineering
- 5-day Moving Average (MA_5)
- 10-day Moving Average (MA_10)

## Models Used
- Linear Regression (baseline)
- LightGBM (final model)

## Evaluation
- MAE (Mean Absolute Error)
- R² Score

## Results
- Linear Regression performed slightly better
- LightGBM used for future forecasting due to scalability

## Forecasting
- Forecasted stock prices for August 2025
- Used iterative prediction with dynamic moving averages

## Visualization
- Actual vs Predicted graph
- Forecast trend graph

## Future Improvements
- Add more features (Volume, Open, High, Low)
- Hyperparameter tuning
- Model explainability (SHAP)
- Deployment using Streamlit

## 📊 Output
(Add graph here after upload)
