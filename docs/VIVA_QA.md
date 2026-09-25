# Viva Q&A — Improved V2

## Why is this a regression problem?
Because the target is a continuous numerical house price in Indian Rupees.

## What is the dataset source?
Kaggle — House Price by Juhi Bhojani: https://www.kaggle.com/datasets/juhibhojani/house-price

## How large is the dataset?
187,531 raw rows and 21 columns. After cleaning and building usable model features, 63,592 rows remain.

## Which models were compared?
Linear Regression, HistGradientBoosting Regressor, and LightGBM Regressor.

## Which model was selected?
LightGBM Regressor because it achieved the lowest validation RMSE.

## Final model performance?
Test R² = 0.8649, MAE ≈ ₹20.73 lakh, RMSE ≈ ₹42.44 lakh.

## Why did performance improve?
V2 adds title-derived locality, society/project information, more engineered features, and LightGBM's nonlinear categorical modeling.

## Why not use `Price (in rupees)` as a feature?
It behaves like price-per-square-foot and is directly related to total price. Using it would create a circular/leaky prediction setup.

## How is the test set protected?
The split is 70% train, 15% validation and 15% test. Models are compared on validation RMSE; the test set is used only after selecting the final model.

## Can this reach R² = 0.99?
Not honestly guaranteed with this dataset. A 0.99 score may indicate duplicates, price-derived leakage or train/test overlap. V2 prioritizes honest generalization.
