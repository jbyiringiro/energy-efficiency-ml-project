# Predicting Household Appliance Energy Consumption Using ML and Deep Learning

**Summative Assignment: Introduction to Machine Learning**

**Author:** Byiringiro Josue

## Overview

This project presents a systematic comparison of traditional machine learning and deep learning approaches for predicting household appliance energy consumption. Using the UCI Appliances Energy Prediction dataset (19,735 samples from a low-energy house in Belgium), nine regression experiments were conducted across five ML models and four DL architectures.

## Dataset

- **Source:** [UCI Machine Learning Repository - Appliances Energy Prediction](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
- **Samples:** 19,735 observations at 10-minute intervals
- **Features:** Temperature and humidity from 9 indoor sensors, outdoor weather variables
- **Target:** Appliance energy consumption (Wh)

## Experiments

| # | Model | Test R² | Test RMSE |
|---|-------|---------|-----------|
| 1 | Linear Regression | 0.6123 | 34.43 |
| 2 | Ridge Regression | 0.6255 | 33.84 |
| 3 | Random Forest | 0.6209 | 34.05 |
| 4 | Gradient Boosting | 0.6407 | 33.15 |
| 5 | SVR (RBF) | 0.2590 | 47.60 |
| 6 | Sequential Shallow NN | 0.6521 | 32.62 |
| 7 | Sequential Deep + L2 | 0.6708 | 31.73 |
| 8 | Multi-Branch Functional API | 0.6394 | 33.21 |
| 9 | Residual Network + tf.data | 0.6650 | 32.01 |

## Key Findings

- Feature engineering (lag features, cyclical encodings, temperature differentials) proved critical across all model families
- Deep Sequential network with L2 regularization achieved the best test R² of 0.6708
- Gradient Boosting was the top traditional ML model (R² = 0.6407)
- Deep learning modestly outperformed ensemble methods, but the margin is narrow

## Tools and Frameworks

- Python, Scikit-learn, TensorFlow/Keras
- Pandas, NumPy, Matplotlib, Seaborn

## Demo Video

[Watch the demo](https://drive.google.com/file/d/1ddehB9hWIc5j0E1H-pUOIx6__aTFiStD/view?usp=sharing)
