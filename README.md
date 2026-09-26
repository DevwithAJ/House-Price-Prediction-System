# 🏠 House Price Prediction System

> An End-to-End Machine Learning Regression Project for Predicting House Prices in India using a real-world real estate dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Regressor-orange)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

---

## 📌 Project Overview

The **House Price Prediction System** is a complete Machine Learning project that predicts residential property prices using a large-scale Indian real estate dataset.

This project demonstrates the complete ML lifecycle:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Deployment
      ↓
Flask Web Application
```

The system provides accurate price estimation through a modern responsive web application and JSON API.

---

## 🚀 Features

- End-to-End Machine Learning Pipeline
- Modern Flask Web Application
- Responsive UI (Mobile, Tablet & Desktop)
- Real Estate Dataset Analysis
- Interactive Dashboard
- Price Prediction Form
- Model Performance Analytics
- REST API Support
- Deployment Ready Configuration
- Health Monitoring Endpoint

---

# 📊 Dataset Information

| Attribute | Value |
|------------|------------|
| Source | Kaggle |
| Records | 187,531 |
| Features | 21 |
| Locations | 81 |
| Clean Modeling Rows | 63,474 |
| Target Variable | Amount (INR) |

### Dataset Source

**Kaggle — House Price Dataset by Juhi Bhojani**

https://www.kaggle.com/datasets/juhibhojani/house-price

---

# 🤖 Model Performance

## Final Model

**LightGBM Regressor**

### Test Performance

| Metric | Value |
|----------|----------|
| R² Score | **86.49%** |
| MAE | ₹20.73 Lakh |
| RMSE | ₹42.44 Lakh |

The model explains approximately **86.49%** of the variance in unseen house prices.

---

# 🧠 Feature Engineering

The following preprocessing and feature engineering techniques were applied:

### Data Cleaning

- Lac → Numeric INR Conversion
- Crore → Numeric INR Conversion
- Area Standardization
- Missing Value Handling
- Duplicate Removal
- Outlier Filtering

### Feature Extraction

- BHK Extraction
- Property Type Extraction
- Floor Information Parsing
- Bathroom Extraction
- Balcony Extraction
- Parking Extraction

### Engineered Features

- Area Per BHK
- Floor Ratio
- High Rise Indicator

---

# 📈 Models Compared

| Model | Validation R² |
|---------|---------|
| HistGradientBoosting Regressor | 0.8062 |
| Random Forest Regressor | 0.7999 |
| Linear Regression | 0.6441 |

### Selected Model

🏆 **LightGBM Regressor**

Selected for its superior predictive performance and generalization.

---

# 🌐 Web Application

### Available Routes

| Route | Description |
|---------|-------------|
| `/` | Home Dashboard |
| `/predict` | House Price Prediction |
| `/dataset` | Dataset Analysis |
| `/model-info` | Model Metrics |
| `/about` | Project Workflow |
| `/api/predict` | JSON Prediction API |
| `/health` | Health Check Endpoint |

---

# 📂 Project Structure

```text
House_Price_Prediction_System/

├── app.py
├── data_utils.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── requirements-dev.txt
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

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/DevwithAJ/House-Price-Prediction-System.git

cd House-Price-Prediction-System
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

# 🔄 Retraining the Model

```bash
pip install -r requirements-dev.txt

python train_model.py
```

---

# 🔌 API Example

### POST `/api/predict`

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

---

# ☁️ Deployment

### Render Configuration

#### Build Command

```bash
pip install -r requirements.txt
```

#### Start Command

```bash
gunicorn app:app
```

#### Health Check

```text
/health
```

---

# 📚 Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- Joblib

### Web Development

- Flask
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

### Deployment

- Render
- Gunicorn

---

# ⚠️ Disclaimer

This application provides machine-learning-based property price estimates using publicly available listing data.

Predicted values should not be considered official property valuations and may differ from actual market transaction prices.

---

# 👨‍💻 Author

### Ajit Kumar

**B.Tech Computer Science & Engineering**  
Sandip University

🔗 GitHub: https://github.com/DevwithAJ

⭐ If you found this project useful, consider giving it a Star.