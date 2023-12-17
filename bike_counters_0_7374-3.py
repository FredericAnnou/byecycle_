# -*- coding: utf-8 -*-
"""bike_counters_0.7374.ipynb

"""

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

"""# Data Loading"""

# Load the Training Data
problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred

def get_train_data(path="../input/mdsb-2023/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array
X, y = get_train_data()

# We upload the scholar breaks data

link = 'https://raw.githubusercontent.com/AntoineAugusti/vacances-scolaires/master/data.csv'
data_school_holidays = pd.read_csv(link)
data_school_holidays['date'] = pd.to_datetime(data_school_holidays['date'])
data_school_holidays = data_school_holidays.dropna(axis=1)
keep_columns = ["date","vacances_zone_c"]
data_school_holidays = data_school_holidays[keep_columns]
data_school_holidays['vacances_zone_c'] = data_school_holidays['vacances_zone_c'].fillna(False)
data_school_holidays['vacances_zone_c'] = data_school_holidays['vacances_zone_c'].astype(int)
data_school_holidays

"""# Data Splitting"""

def train_test_split_temporal(X, y, delta_threshold="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid
X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

"""# Model Training"""

# updated version
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, holiday_dates):
        self.holiday_dates = holiday_dates

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['weekday'] = X['date'].dt.weekday
        X['hour'] = X['date'].dt.hour
        X['is_workday'] = np.where(X['weekday'] < 5, 1, 0)
        X['season'] = X['month'].apply(lambda x: (x%12 + 3)//3)

        X['day_sin'] = np.sin(2 * np.pi * X['weekday']/7)
        X['day_cos'] = np.cos(2 * np.pi * X['weekday']/7)
        X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
        X['month_cos'] = np.cos(2 * np.pi * X['month']/12)

        X['vacances_zone_c'] = X['date'].isin(self.holiday_dates).astype(int)
        return X.drop(columns=["date"])

# Instantiate the DateEncoder with the holiday dates
# date_encoder = DateEncoder(holiday_dates)
date_encoder = DateEncoder(data_school_holidays[data_school_holidays['vacances_zone_c']==1 & data_school_holidays['date'].isin(X['date'])])


categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]

preprocessor = ColumnTransformer(
    [
        ("num", 'passthrough', ["month", "day", "weekday", "hour","season","day_sin","day_cos","month_sin","month_cos","is_workday","vacances_zone_c","latitude","longitude"]),
        ("cat", categorical_encoder, categorical_cols),
    ]
)

rg1 = LGBMRegressor(colsample_bytree=0.40927396991732046,
                    learning_rate=0.1956637125175451,
                    min_child_samples= 321,
                    min_child_weight=0.009676548190436696,
                    num_leaves= 118,
                    reg_alpha=0,
                    reg_lambda= 5,
                    subsample=0.4355591136556686
                   )

rg2 = XGBRegressor(colsample_bytree=0.6186133958679972,
                   learning_rate=0.04053474178612112,
                   max_depth=8,
                   min_child_weight=6,
                   n_estimators=602,
                   reg_alpha=0.4878098008087379,
                   reg_lambda=0.8508175178361462,
                   subsample=0.3615213371781007
                  )

rg3 = CatBoostRegressor(bagging_temperature=0.0906064345328208,
                        border_count=239,
                        depth=6,
                        iterations=975,
                        l2_leaf_reg=6.142344384136116
                       )

vrg = VotingRegressor(estimators=[('lgb', rg1), ('xgb', rg2), ('cgb', rg3)], weights=[1, 1, 1])

pipe = make_pipeline(date_encoder, preprocessor, vrg)

pipe.fit(X_train, y_train)

"""# Cross Validation"""

# from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# cv = TimeSeriesSplit(n_splits=6)

# # When using a scorer in scikit-learn it always needs to be better when smaller, hence the minus sign.
# scores = cross_val_score(
#     pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
# )
# print("RMSE: ", scores)
# print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")

"""# Prediction"""

final_test_path = '../input/mdsb-2023/final_test.parquet'
final_test = pd.read_parquet(final_test_path)

predicted_log_bike_count = pipe.predict(final_test)

predictions = pd.DataFrame({
    "Id": final_test.index,
    "log_bike_count": predicted_log_bike_count
})

predictions.to_csv('submission.csv', index=False)

