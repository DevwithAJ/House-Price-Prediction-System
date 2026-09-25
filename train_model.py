from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_utils import DATASET_SOURCE_NAME, DATASET_SOURCE_URL, RAW_INPUT_FEATURES, load_and_clean
from preprocessing import FeatureEngineerV2, NativeCategoryPreprocessor, build_target_encoder_preprocessor

BASE_DIR=Path(__file__).resolve().parent
DATA_PATH=BASE_DIR/'data'/'house_prices.csv.gz'; MODEL_DIR=BASE_DIR/'models'; STATIC_IMG=BASE_DIR/'static'/'images'
MODEL_DIR.mkdir(exist_ok=True); STATIC_IMG.mkdir(parents=True,exist_ok=True)
RANDOM_STATE=42

def metrics(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
def target_pipe(model): return Pipeline([('feature_engineering',FeatureEngineerV2()),('preprocessing',build_target_encoder_preprocessor()),('model',model)])
def lgbm_pipe():
    return Pipeline([('feature_engineering',FeatureEngineerV2()),('preprocessing',NativeCategoryPreprocessor(min_high_card_frequency=3)),('model',LGBMRegressor(n_estimators=1800,learning_rate=.025,num_leaves=63,min_child_samples=20,subsample=.90,colsample_bytree=.90,reg_lambda=3.0,random_state=RANDOM_STATE,n_jobs=-1,verbosity=-1))])
def candidates():
    return {
        'Linear Regression': target_pipe(LinearRegression()),
        'HistGradientBoosting Regressor': target_pipe(HistGradientBoostingRegressor(max_iter=550,learning_rate=.055,max_leaf_nodes=63,l2_regularization=3.0,random_state=RANDOM_STATE)),
        'LightGBM Regressor': lgbm_pipe(),
    }

def main():
    raw=pd.read_csv(DATA_PATH); clean=load_and_clean(DATA_PATH); clean.to_csv(BASE_DIR/'data'/'cleaned_model_data_v2.csv',index=False)
    X=clean[RAW_INPUT_FEATURES].copy(); y=clean['price'].copy()
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.30,random_state=RANDOM_STATE)
    X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.50,random_state=RANDOM_STATE)
    rows=[]
    for name,pipe in candidates().items():
        st=time.time(); pipe.fit(X_train,y_train); pred=pipe.predict(X_val); row={'Model':name,**metrics(y_val,pred),'Training_Seconds':round(time.time()-st,3)}; rows.append(row); print(name,row,flush=True)
    comparison=pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True); best_name=str(comparison.iloc[0]['Model'])
    X_trainval=pd.concat([X_train,X_val],ignore_index=True); y_trainval=pd.concat([y_train,y_val],ignore_index=True)
    final=candidates()[best_name]; final.fit(X_trainval,y_trainval); test_pred=final.predict(X_test); test_metrics=metrics(y_test,test_pred)
    model_path=MODEL_DIR/'best_house_price_pipeline_v2.joblib'; joblib.dump(final,model_path); comparison.to_csv(MODEL_DIR/'model_comparison_v2.csv',index=False)
    sample=X_test.reset_index(drop=True).copy(); sample['actual_price']=y_test.reset_index(drop=True); sample['predicted_price']=test_pred; sample['absolute_error']=(sample['actual_price']-sample['predicted_price']).abs(); sample.head(750).to_csv(BASE_DIR/'data'/'test_predictions_sample_v2.csv',index=False)
    # plots
    plt.figure(figsize=(8,4.5)); plt.hist(clean['price']/1e5,bins=45,edgecolor='black'); plt.xlabel('Price (₹ Lakhs)'); plt.ylabel('Property Count'); plt.title('House Price Distribution'); plt.tight_layout(); plt.savefig(STATIC_IMG/'eda_price_distribution.png',dpi=150); plt.close()
    top=clean['location'].value_counts().head(12).sort_values(); plt.figure(figsize=(8,5)); plt.barh(top.index,top.values); plt.xlabel('Clean Property Records'); plt.ylabel('Location'); plt.title('Top Locations by Listing Count'); plt.tight_layout(); plt.savefig(STATIC_IMG/'eda_top_locations.png',dpi=150); plt.close()
    sm=clean.sample(min(6000,len(clean)),random_state=RANDOM_STATE); plt.figure(figsize=(8,5)); plt.scatter(sm['area_sqft'],sm['price']/1e5,alpha=.30,s=12); plt.xlabel('Area (sq.ft)'); plt.ylabel('Price (₹ Lakhs)'); plt.title('Area vs House Price'); plt.tight_layout(); plt.savefig(STATIC_IMG/'eda_area_vs_price.png',dpi=150); plt.close()
    plot_df=comparison.sort_values('RMSE'); plt.figure(figsize=(8,4.7)); plt.barh(plot_df['Model'],plot_df['RMSE']/1e5); plt.xlabel('Validation RMSE (₹ Lakhs)'); plt.title('Improved V2 Model Comparison'); plt.tight_layout(); plt.savefig(STATIC_IMG/'model_comparison.png',dpi=150); plt.close()
    plt.figure(figsize=(6,6)); plt.scatter(y_test/1e5,test_pred/1e5,alpha=.30,s=12); lo=min((y_test/1e5).min(),(test_pred/1e5).min()); hi=max((y_test/1e5).max(),(test_pred/1e5).max()); plt.plot([lo,hi],[lo,hi],linestyle='--'); plt.xlabel('Actual Price (₹ Lakhs)'); plt.ylabel('Predicted Price (₹ Lakhs)'); plt.title('Actual vs Predicted — Improved V2'); plt.tight_layout(); plt.savefig(STATIC_IMG/'actual_vs_predicted_v2.png',dpi=150); plt.close()
    locations=sorted(clean['location'].astype(str).unique().tolist()); cats={c:sorted(clean[c].astype(str).unique().tolist()) for c in ['property_type','transaction','furnishing','facing','ownership','overlooking']}
    ranges={c:[float(clean[c].min(skipna=True)),float(clean[c].max(skipna=True))] for c in ['bhk','area_sqft','current_floor','total_floors','bathrooms','balconies','parking']}
    top_localities=clean.loc[clean['locality_hint']!='unknown','locality_hint'].value_counts().head(300).index.astype(str).tolist(); top_societies=clean.loc[clean['society']!='unknown','society'].value_counts().head(300).index.astype(str).tolist()
    metadata={'project_name':'House Price Prediction System','version':'Improved V2','dataset':{'name':DATASET_SOURCE_NAME,'url':DATASET_SOURCE_URL,'raw_rows':int(raw.shape[0]),'raw_columns':int(raw.shape[1]),'clean_rows':int(len(clean)),'locations':int(clean['location'].nunique()),'locality_hints':int(clean['locality_hint'].nunique()),'societies':int(clean['society'].nunique()),'target':'Amount(in rupees) converted to numeric INR'},'split':{'train_rows':int(len(X_train)),'validation_rows':int(len(X_val)),'test_rows':int(len(X_test)),'random_state':RANDOM_STATE},'candidate_models':comparison.to_dict(orient='records'),'best_model':best_name,'selection_rule':'Lowest validation RMSE','test_metrics':test_metrics,'evaluation_design':'70% train / 15% validation / 15% untouched test','input_features':RAW_INPUT_FEATURES,'numeric_ranges':ranges,'locations':locations,'categories':cats,'top_localities':top_localities,'top_societies':top_societies,'excluded_for_leakage':['Price (in rupees) / price-per-square-foot is excluded because it is circular with total price.'],'improvements':['Added title-derived locality/area feature','Added optional Society/Project feature','Added area_per_bhk, floor_ratio, high-rise, bathroom/BHK, balcony/BHK, society and parking indicators','Added LightGBM for high-cardinality nonlinear tabular modeling','Rare/unseen locality and society values are handled safely by the saved preprocessing pipeline','Kept train/validation/test separation and selected only on validation RMSE'],'final_model_file':'models/best_house_price_pipeline_v2.joblib'}
    with open(MODEL_DIR/'model_metadata.json','w',encoding='utf-8') as f: json.dump(metadata,f,indent=2)
    loaded=joblib.load(model_path); sanity=pd.DataFrame([{'location':'thane','locality_hint':'pokhran road','society':'dosti vihar','property_type':'apartment','bhk':2,'area_sqft':1200,'current_floor':5,'total_floors':14,'bathrooms':2,'balconies':1,'parking':1,'transaction':'resale','furnishing':'semi-furnished','facing':'east','ownership':'freehold','overlooking':'garden/park'}]); print('\nSelected:',best_name); print('Final test:',test_metrics); print('Sanity prediction:',float(loaded.predict(sanity)[0])); print('Saved:',model_path)
if __name__=='__main__': main()
