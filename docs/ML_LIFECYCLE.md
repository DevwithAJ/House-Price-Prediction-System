# Improved V2 ML Lifecycle

1. **Data Collection** — Public Kaggle House Price dataset by Juhi Bhojani.
2. **Data Understanding** — 187,531 raw rows and 21 raw columns; target is `Amount(in rupees)`.
3. **Data Cleaning** — Lac/Cr price parsing, area-unit conversion, floor/count parsing, invalid-value filtering, duplicate-like row removal.
4. **EDA** — Price distribution, top locations, area-price relationship and categorical patterns.
5. **Preprocessing** — Missing-value handling; target encoding for baseline models; native categorical preprocessing for LightGBM.
6. **Feature Engineering** — BHK/property type from title, title-derived locality, society/project, area_per_bhk, floor_ratio, is_highrise, bath_per_bhk, balcony_per_bhk, has_society, has_parking.
7. **Model Training** — Linear Regression, HistGradientBoosting Regressor, LightGBM Regressor.
8. **Model Comparison** — Validation MAE, RMSE and R²; primary selection metric is validation RMSE.
9. **Evaluation** — 70/15/15 train-validation-test split; final test set is not used for model selection.
10. **Model Saving** — Complete feature engineering + category preprocessing + LightGBM model saved as Joblib Pipeline.
11. **Application/API** — Flask UI and POST `/api/predict`.
12. **Deployment** — Gunicorn + Render configuration.

## Leakage prevention

`Price (in rupees)` (price-per-area style field) is intentionally excluded because it is circular with the target total price.
