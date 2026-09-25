# House Price Prediction System

A complete end-to-end Machine Learning regression project built on a real public Indian real-estate dataset.

## Final Deployable Version

This final version includes a FinanceAI-style full-screen prediction loader, mobile/tablet/desktop responsive layout, LightGBM V2 model, JSON API, health check, GitHub-friendly compressed dataset, and Render production configuration.

**Final model:** LightGBM Regressor  
**Held-out test R²:** 0.8649 (86.49%)  
**Held-out test MAE:** about ₹20.73 lakh  
**Held-out test RMSE:** about ₹42.44 lakh

Runtime installation for the deployed web app uses `requirements.txt`. For notebook retraining and tests use `requirements-dev.txt`.


## Dataset

**Source:** Kaggle — House Price by Juhi Bhojani  
**URL:** https://www.kaggle.com/datasets/juhibhojani/house-price  
**Raw shape:** 187,531 rows × 21 columns  
**Locations:** 81  
**Clean modeling rows:** 63,474

The original target is `Amount(in rupees)`. Values such as `42 Lac` and `1.40 Cr` are converted to numeric INR.

`Price (in rupees)` is intentionally excluded from predictors because it behaves like price-per-square-foot and would make total-price prediction circular.

## Complete ML Lifecycle

Data Collection → Data Understanding → Data Cleaning → EDA → Preprocessing → Feature Engineering → Model Training → Model Comparison → Evaluation → Model Saving → Application/API Development → Deployment

## Models Compared

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| HistGradientBoosting Regressor | ₹2,546,454 | ₹5,013,561 | 0.8062 |
| Random Forest Regressor | ₹2,488,132 | ₹5,095,177 | 0.7999 |
| Linear Regression | ₹3,796,889 | ₹6,794,586 | 0.6441 |

**Selected model:** HistGradientBoosting Regressor  
**Selection rule:** Lowest validation RMSE

### Final held-out test performance

- MAE: ₹2,638,177
- RMSE: ₹5,497,350
- R²: 0.7850
- 3-fold CV mean RMSE: ₹5,178,173

## Main Cleaning & Feature Engineering

- Lac/Cr → numeric INR
- Carpet Area / Super Area unit conversion to sq.ft
- BHK extraction from Title
- Property type extraction from Title
- Floor split into current and total floors
- Bathroom, balcony and parking parsing
- Missing-value handling
- Duplicate-like row removal
- Reasonable outlier / sanity filtering
- `area_per_bhk`
- `floor_ratio`
- `is_highrise`

## Web Application

Routes:

- `/` — Home dashboard
- `/predict` — prediction form
- `/dataset` — dataset + EDA
- `/model-info` — model comparison and metrics
- `/about` — ML lifecycle
- `/api/predict` — JSON prediction API
- `/health` — deployment health endpoint

## Project Structure

```text
House_Price_Prediction_System_New_Dataset/
├── app.py
├── data_utils.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── render.yaml
├── Procfile
├── data/
│   ├── house_prices.csv.gz
│   ├── cleaned_model_data.csv
│   └── test_predictions_sample.csv
├── models/
│   ├── best_house_price_pipeline.joblib
│   ├── model_metadata.json
│   └── model_comparison.csv
├── notebooks/
│   └── House_Price_Prediction_System_Complete.ipynb
├── templates/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── docs/
└── tests/
```

## Run on Windows / VS Code

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

To retrain:

```powershell
pip install -r requirements-dev.txt
python train_model.py
```

## API Example

POST `/api/predict`

```json
{
  "location": "bangalore",
  "property_type": "apartment",
  "bhk": 2,
  "area_sqft": 1200,
  "current_floor": 5,
  "total_floors": 14,
  "bathrooms": 2,
  "balconies": 1,
  "parking": 1,
  "transaction": "resale",
  "furnishing": "semi-furnished",
  "facing": "east",
  "ownership": "freehold",
  "overlooking": "garden/park"
}
```

## Render Deployment

The repository is deployment-ready.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check: `/health`

The actual public deployment requires access to the project owner's GitHub and Render account.

## Important Limitation

The output is a machine-learning estimate based on public listing data, not a certified property valuation. Listing price may differ from final transaction price and market conditions change over time.
