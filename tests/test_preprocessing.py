import pandas as pd
from data_utils import parse_money, parse_area, extract_locality_hint
from preprocessing import FeatureEngineerV2, NativeCategoryPreprocessor

def test_parse_money():
    assert parse_money('42 Lac') == 4200000
    assert parse_money('1.40 Cr') == 14000000

def test_parse_area():
    assert parse_area('500 sqft') == 500
    assert parse_area('100 sqyrd') == 900

def test_locality_extraction():
    assert extract_locality_hint('2 BHK Flat for sale in Dosti Vihar Pokhran Road','Dosti Vihar','thane') == 'pokhran road'

def test_feature_engineering_v2():
    row=pd.DataFrame([{'location':'thane','locality_hint':'pokhran road','society':'dosti vihar','property_type':'apartment','bhk':2,'area_sqft':1200,'current_floor':5,'total_floors':10,'bathrooms':2,'balconies':1,'parking':1,'transaction':'resale','furnishing':'semi-furnished','facing':'east','ownership':'freehold','overlooking':'garden/park'}])
    out=FeatureEngineerV2().fit_transform(row)
    assert out.iloc[0]['area_per_bhk']==600
    assert out.iloc[0]['floor_ratio']==0.5
    assert out.iloc[0]['is_highrise']==1
    assert out.iloc[0]['has_society']==1
