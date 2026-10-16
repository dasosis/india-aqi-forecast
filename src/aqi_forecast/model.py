"""The forecasting models.

Both predict the *change* in PM2.5 on a log scale (the idea from the notebook):
  ridge       a regularised straight-line model (the stage 2 winner)
  lgb_change  LightGBM trees on the same target

Each is a scikit-learn estimator, so MLflow can save and load it.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aqi_forecast.config import CITIES
from aqi_forecast.features import FEATURES

MODEL_KINDS = ("ridge", "lgb_change")
# Classes inside our saved models that MLflow must be told are safe to load
TRUSTED_TYPES = [
    "aqi_forecast.model.Forecaster",
    "collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor",
]
LOG_COLS = ["pm25", "pm25_lag1", "pm25_7d_mean", "cams_tomorrow"]
LGB_SETTINGS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs for the ridge model: logs, one-hot city and weekday, no gaps."""
    x = df[FEATURES].copy()
    for col in LOG_COLS:
        x[col] = np.log(x[col].fillna(df.pm25))  # missing lag -> assume same as today
    # fixed categories so training and prediction always get the same columns
    x["city"] = pd.Categorical(x.city, categories=list(CITIES))
    x["dow"] = pd.Categorical(x.dow, categories=range(7))
    return pd.get_dummies(x, columns=["city", "dow"], dtype=float).fillna(0)


def tree_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs for LightGBM: raw features, city as a category, gaps allowed."""
    x = df[FEATURES].copy()
    x["city"] = pd.Categorical(x.city, categories=list(CITIES))
    return x


class Forecaster(BaseEstimator, RegressorMixin):
    """Predicts tomorrow's PM2.5 from a features table (see features.make_features)."""

    def __init__(self, kind: str = "ridge"):
        self.kind = kind

    def fit(self, X: pd.DataFrame, y):
        if self.kind == "ridge":
            self.model_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
            self.model_.fit(model_inputs(X), np.log(y))
        elif self.kind == "lgb_change":
            self.model_ = lgb.LGBMRegressor(**LGB_SETTINGS)
            self.model_.fit(tree_inputs(X), np.log(y) - np.log(X.pm25))
        else:
            raise ValueError(f"unknown model kind {self.kind!r}; choose from {MODEL_KINDS}")
        return self

    def predict(self, X: pd.DataFrame):
        if self.kind == "ridge":
            return np.exp(self.model_.predict(model_inputs(X)))
        return X.pm25.to_numpy() * np.exp(self.model_.predict(tree_inputs(X)))


def fit(df: pd.DataFrame, kind: str = "ridge") -> Forecaster:
    return Forecaster(kind).fit(df, df.target)


def predict(model: Forecaster, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df)
