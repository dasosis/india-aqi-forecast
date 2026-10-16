"""The forecasting model: ridge regression on the log of PM2.5.

It learns log(tomorrow) from log(today) and the other features, which is the
"predict the change" idea from the notebook.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aqi_forecast.config import CITIES
from aqi_forecast.features import FEATURES

LOG_COLS = ["pm25", "pm25_lag1", "pm25_7d_mean", "cams_tomorrow"]


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for col in LOG_COLS:
        x[col] = np.log(x[col].fillna(df.pm25))  # missing lag -> assume same as today
    # fixed categories so training and prediction always get the same columns
    x["city"] = pd.Categorical(x.city, categories=list(CITIES))
    x["dow"] = pd.Categorical(x.dow, categories=range(7))
    return pd.get_dummies(x, columns=["city", "dow"], dtype=float).fillna(0)


def new_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def fit(df: pd.DataFrame):
    return new_model().fit(model_inputs(df), np.log(df.target))


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(model_inputs(df)))
