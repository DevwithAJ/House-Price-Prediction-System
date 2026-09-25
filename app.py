
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, render_template, request

from data_utils import RAW_INPUT_FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_house_price_pipeline_v2.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def indian_grouping(number: int) -> str:
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        return sign + digits
    last = digits[-3:]
    rest = digits[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return sign + ",".join(groups) + "," + last

def inr(value):
    try:
        return "₹" + indian_grouping(int(round(float(value))))
    except (TypeError, ValueError):
        return value

def compact_inr(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} Lakh"
    return inr(value)

app.jinja_env.filters["inr"] = inr
app.jinja_env.filters["compact_inr"] = compact_inr

NUMERIC_FIELDS = [
    "bhk", "area_sqft", "current_floor", "total_floors",
    "bathrooms", "balconies", "parking"
]

def _number(payload, key, required=False, default=None):
    value = payload.get(key, default)
    if value in (None, "", "null"):
        if required:
            raise ValueError(f"{key.replace('_', ' ').title()} is required.")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key.replace('_', ' ').title()} must be numeric.")

def prepare_payload(payload):
    meta = load_metadata()
    row = {
        "location": str(payload.get("location", "")).strip().lower(),
        "locality_hint": str(payload.get("locality_hint", "unknown")).strip().lower() or "unknown",
        "society": str(payload.get("society", "unknown")).strip().lower() or "unknown",
        "property_type": str(payload.get("property_type", "apartment")).strip().lower(),
        "bhk": _number(payload, "bhk", required=True),
        "area_sqft": _number(payload, "area_sqft", required=True),
        "current_floor": _number(payload, "current_floor"),
        "total_floors": _number(payload, "total_floors"),
        "bathrooms": _number(payload, "bathrooms"),
        "balconies": _number(payload, "balconies"),
        "parking": _number(payload, "parking"),
        "transaction": str(payload.get("transaction", "resale")).strip().lower(),
        "furnishing": str(payload.get("furnishing", "semi-furnished")).strip().lower(),
        "facing": str(payload.get("facing", "unknown")).strip().lower() or "unknown",
        "ownership": str(payload.get("ownership", "unknown")).strip().lower() or "unknown",
        "overlooking": str(payload.get("overlooking", "unknown")).strip().lower() or "unknown",
    }

    warnings = []
    if row["location"] not in meta["locations"]:
        raise ValueError("Please select a location available in the training dataset.")

    ranges = meta["numeric_ranges"]
    for key in ["bhk", "area_sqft", "current_floor", "total_floors", "bathrooms", "balconies", "parking"]:
        value = row[key]
        if value is None:
            continue
        lo, hi = ranges[key]
        if value < lo or value > hi:
            warnings.append(
                f"{key.replace('_', ' ').title()} is outside the observed training range "
                f"({lo:g}–{hi:g})."
            )

    if row["bhk"] <= 0 or row["area_sqft"] <= 0:
        raise ValueError("BHK and area must be greater than zero.")
    if row["total_floors"] is not None and row["total_floors"] < 0:
        raise ValueError("Total floors cannot be negative.")
    if (
        row["current_floor"] is not None
        and row["total_floors"] not in (None, 0)
        and row["current_floor"] > row["total_floors"]
    ):
        warnings.append("Current floor is greater than total floors; please verify the input.")

    return row, warnings

def prediction_details(row):
    return {
        "Location": row["location"].replace("-", " ").title(),
        "Locality / Area": row["locality_hint"].replace("-", " ").title() if row["locality_hint"] != "unknown" else "Not provided",
        "Society / Project": row["society"].title() if row["society"] != "unknown" else "Not provided",
        "Property Type": row["property_type"].title(),
        "BHK": int(row["bhk"]) if row["bhk"] is not None else "—",
        "Area": f"{int(row['area_sqft']):,} sq.ft",
        "Bathrooms": int(row["bathrooms"]) if row["bathrooms"] is not None else "Unknown",
        "Balconies": int(row["balconies"]) if row["balconies"] is not None else "Unknown",
        "Floor": (
            f"{int(row['current_floor']) if row['current_floor'] is not None else '?'} "
            f"of {int(row['total_floors']) if row['total_floors'] is not None else '?'}"
        ),
        "Parking": int(row["parking"]) if row["parking"] is not None else "Unknown",
        "Transaction": row["transaction"].title(),
        "Furnishing": row["furnishing"].title(),
        "Facing": row["facing"].title(),
        "Ownership": row["ownership"].title(),
        "Overlooking": row["overlooking"].title(),
    }

@app.get("/")
def home():
    return render_template("index.html", metadata=load_metadata())

@app.route("/predict", methods=["GET", "POST"])
def predict():
    metadata = load_metadata()
    if request.method == "GET":
        return render_template("predict.html", metadata=metadata, form={})

    try:
        row, warnings = prepare_payload(request.form.to_dict())
        frame = pd.DataFrame([row], columns=RAW_INPUT_FEATURES)
        predicted = max(0.0, float(load_model().predict(frame)[0]))
        return render_template(
            "result.html",
            metadata=metadata,
            predicted_price=predicted,
            details=prediction_details(row),
            warnings=warnings,
        )
    except ValueError as exc:
        flash(str(exc), "danger")
        return render_template(
            "predict.html",
            metadata=metadata,
            form=request.form,
        ), 400
    except Exception:
        app.logger.exception("Prediction failed")
        flash("Prediction could not be completed. Please check your values and try again.", "danger")
        return render_template("predict.html", metadata=metadata, form=request.form), 500

@app.get("/dataset")
def dataset():
    return render_template("dataset.html", metadata=load_metadata())

@app.get("/model-info")
def model_info():
    return render_template("model_info.html", metadata=load_metadata())

@app.get("/about")
def about():
    return render_template("about.html", metadata=load_metadata())

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        row, warnings = prepare_payload(payload)
        frame = pd.DataFrame([row], columns=RAW_INPUT_FEATURES)
        prediction = max(0.0, float(load_model().predict(frame)[0]))
        return jsonify({
            "success": True,
            "prediction_inr": prediction,
            "prediction_formatted": inr(prediction),
            "prediction_compact": compact_inr(prediction),
            "warnings": warnings,
            "model": load_metadata()["best_model"],
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("API prediction failed")
        return jsonify({"success": False, "error": "Prediction failed."}), 500

@app.get("/health")
def health():
    try:
        load_model()
        return jsonify({
            "status": "ok",
            "project": load_metadata()["project_name"],
            "model_loaded": True,
        })
    except Exception as exc:
        return jsonify({"status": "error", "model_loaded": False, "detail": str(exc)}), 500

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
