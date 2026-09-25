from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

RAW_INPUT_FEATURES=["location","locality_hint","society","property_type","bhk","area_sqft","current_floor","total_floors","bathrooms","balconies","parking","transaction","furnishing","facing","ownership","overlooking"]
CATEGORICAL_FEATURES=["location","locality_hint","society","property_type","transaction","furnishing","facing","ownership","overlooking"]
ENGINEERED_FEATURES=["area_per_bhk","floor_ratio","is_highrise","bath_per_bhk","balcony_per_bhk","has_society","has_parking"]
MODEL_FEATURES=RAW_INPUT_FEATURES+ENGINEERED_FEATURES
NUMERIC_FEATURES=[c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]

class FeatureEngineerV2(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        d=X.copy() if isinstance(X,pd.DataFrame) else pd.DataFrame(X,columns=RAW_INPUT_FEATURES)
        for c in RAW_INPUT_FEATURES:
            if c not in d.columns: d[c]=np.nan
        for c in CATEGORICAL_FEATURES:
            d[c]=d[c].astype("string").str.strip().str.lower().fillna("unknown").replace({"<na>":"unknown","nan":"unknown","":"unknown"})
        for c in [x for x in RAW_INPUT_FEATURES if x not in CATEGORICAL_FEATURES]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["area_per_bhk"]=d["area_sqft"]/d["bhk"].replace(0,np.nan)
        d["floor_ratio"]=d["current_floor"]/d["total_floors"].replace(0,np.nan)
        d["is_highrise"]=(d["total_floors"]>=10).astype(float)
        d["bath_per_bhk"]=d["bathrooms"]/d["bhk"].replace(0,np.nan)
        d["balcony_per_bhk"]=d["balconies"]/d["bhk"].replace(0,np.nan)
        d["has_society"]=(d["society"]!="unknown").astype(float)
        d["has_parking"]=(d["parking"].fillna(0)>0).astype(float)
        return d[MODEL_FEATURES]

class NativeCategoryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,min_high_card_frequency=3): self.min_high_card_frequency=min_high_card_frequency
    def fit(self,X,y=None):
        d=X.copy(); self.numeric_medians_={c:float(pd.to_numeric(d[c],errors="coerce").median()) for c in NUMERIC_FEATURES}; self.category_levels_={}
        for c in CATEGORICAL_FEATURES:
            values=d[c].astype("string").str.strip().str.lower().fillna("unknown").replace({"<na>":"unknown","nan":"unknown","":"unknown"}); counts=values.value_counts()
            levels=counts[counts>=self.min_high_card_frequency].index.astype(str).tolist() if c in {"society","locality_hint"} else counts.index.astype(str).tolist()
            for special in ["unknown","other"]:
                if special not in levels: levels.append(special)
            self.category_levels_[c]=levels
        return self
    def transform(self,X):
        d=X.copy()
        for c in NUMERIC_FEATURES: d[c]=pd.to_numeric(d[c],errors="coerce").fillna(self.numeric_medians_[c])
        for c in CATEGORICAL_FEATURES:
            values=d[c].astype("string").str.strip().str.lower().fillna("unknown").replace({"<na>":"unknown","nan":"unknown","":"unknown"}).astype(str)
            levels=self.category_levels_[c]; allowed=set(levels); values=values.where(values.isin(allowed),"other"); d[c]=pd.Categorical(values,categories=levels)
        return d[MODEL_FEATURES]

def build_target_encoder_preprocessor():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("target_encoder",TargetEncoder(target_type="continuous",smooth="auto",cv=5,shuffle=True,random_state=42))])
    return ColumnTransformer([("numeric",num,NUMERIC_FEATURES),("categorical",cat,CATEGORICAL_FEATURES)])
