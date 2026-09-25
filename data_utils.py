from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATASET_SOURCE_NAME = "House Price by Juhi Bhojani"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/juhibhojani/house-price"
AREA_FACTORS = {"sqft":1.0,"sqyrd":9.0,"sqm":10.7639104167,"acre":43560.0,"ground":2400.0,"bigha":27225.0,"marla":272.25,"kanal":5445.0,"cent":435.6,"hectare":107639.104167,"aankadam":72.0}
RAW_INPUT_FEATURES = ["location","locality_hint","society","property_type","bhk","area_sqft","current_floor","total_floors","bathrooms","balconies","parking","transaction","furnishing","facing","ownership","overlooking"]

def _norm_text(value):
    if pd.isna(value): return "unknown"
    text = re.sub(r"[^a-z0-9 /&-]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip() or "unknown"

def parse_money(value):
    if pd.isna(value): return np.nan
    text = str(value).strip().replace(",", "").replace("₹", "").replace("INR", "")
    low = text.lower()
    if not text or "call" in low or "request" in low: return np.nan
    m = re.search(r"([\d.]+)", text)
    if not m: return np.nan
    n = float(m.group(1))
    if "cr" in low: return n * 1e7
    if "lac" in low or "lakh" in low: return n * 1e5
    if "thousand" in low: return n * 1e3
    return n

def parse_area(value):
    if pd.isna(value): return np.nan
    m = re.search(r"([\d.]+)\s*([a-z]+)", str(value).strip().lower().replace(",", ""))
    if not m: return np.nan
    factor = AREA_FACTORS.get(m.group(2))
    return float(m.group(1)) * factor if factor else np.nan

def parse_floor(value):
    if pd.isna(value): return np.nan, np.nan
    text = str(value).strip().lower(); parts = text.split(" out of ")
    def conv(v):
        if v is None: return np.nan
        if v == "ground": return 0.0
        if "lower basement" in v: return -2.0
        if "basement" in v: return -1.0
        m = re.search(r"-?\d+", v); return float(m.group()) if m else np.nan
    return conv(parts[0].strip()), conv(parts[1].strip() if len(parts)>1 else None)

def parse_count(value, gt10_value=11.0):
    if pd.isna(value): return np.nan
    text = str(value).strip().lower()
    if "> 10" in text or ">10" in text: return gt10_value
    m = re.search(r"\d+", text); return float(m.group()) if m else np.nan

def parse_parking(value):
    if pd.isna(value): return np.nan
    m = re.match(r"\s*(\d+)", str(value))
    if not m: return np.nan
    n = float(m.group(1)); return n if n <= 20 else np.nan

def extract_bhk(title):
    if pd.isna(title): return np.nan
    text = str(title).lower(); m = re.search(r"(\d+)\s*bhk", text)
    if m:
        n = float(m.group(1)); return n if n <= 20 else np.nan
    return 1.0 if "studio" in text else np.nan

def extract_property_type(title):
    text = _norm_text(title)
    if "studio" in text: return "studio"
    if "penthouse" in text: return "penthouse"
    if "villa" in text: return "villa"
    if "builder floor" in text: return "builder floor"
    if "house" in text and "flat" not in text: return "house"
    if "plot" in text and "flat" not in text: return "plot"
    if "flat" in text or "apartment" in text: return "apartment"
    return "other"

def normalize_category(series):
    return series.astype("string").str.strip().str.lower().fillna("unknown").replace({"<na>":"unknown","nan":"unknown","":"unknown"})

def extract_locality_hint(title, society, city):
    t = _norm_text(title); soc = _norm_text(society); city = _norm_text(city)
    if "for sale" in t: t = t.split("for sale",1)[1].strip()
    if t.startswith("in "): t = t[3:].strip()
    if soc not in {"unknown","nan"}: t = t.replace(soc, " ", 1)
    t = re.sub(r"\s+", " ", t).strip()
    if city != "unknown" and t.endswith(" " + city):
        candidate = t[:-(len(city)+1)].strip()
        if candidate: t = candidate
    return t or "unknown"

def build_modeling_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    d["price"] = raw["Amount(in rupees)"].map(parse_money)
    d["location"] = normalize_category(raw["location"])
    d["society"] = normalize_category(raw["Society"])
    d["locality_hint"] = [extract_locality_hint(t,s,c) for t,s,c in zip(raw["Title"], raw["Society"], raw["location"])]
    d["property_type"] = raw["Title"].map(extract_property_type)
    d["bhk"] = raw["Title"].map(extract_bhk)
    d["area_sqft"] = raw["Carpet Area"].map(parse_area).fillna(raw["Super Area"].map(parse_area))
    floors = raw["Floor"].map(parse_floor); d["current_floor"]=[x[0] for x in floors]; d["total_floors"]=[x[1] for x in floors]
    d["bathrooms"] = raw["Bathroom"].map(parse_count); d["balconies"] = raw["Balcony"].map(parse_count); d["parking"] = raw["Car Parking"].map(parse_parking)
    d["transaction"] = normalize_category(raw["Transaction"]); d["furnishing"] = normalize_category(raw["Furnishing"]); d["facing"] = normalize_category(raw["facing"]); d["ownership"] = normalize_category(raw["Ownership"]); d["overlooking"] = normalize_category(raw["overlooking"])
    d = d.dropna(subset=["price","location","area_sqft","bhk"])
    d = d[d["price"].between(5e5,1.5e8) & d["area_sqft"].between(150,15000) & d["bhk"].between(1,10)]
    d = d[d["bathrooms"].isna() | d["bathrooms"].between(1,10)]
    d = d[d["current_floor"].isna() | d["current_floor"].between(-2,100)]
    d = d[d["total_floors"].isna() | d["total_floors"].between(0,100)]
    return d.drop_duplicates(subset=RAW_INPUT_FEATURES+["price"]).reset_index(drop=True)

def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    return build_modeling_table(pd.read_csv(csv_path))
