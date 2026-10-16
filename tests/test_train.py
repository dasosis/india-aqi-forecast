"""Tests for model choice and the champion rule."""

import numpy as np
import pandas as pd
import pytest

from aqi_forecast import model as m
from aqi_forecast.features import FEATURES
from aqi_forecast.train import is_new_champion


def feature_table(n=60):
    rng = np.random.default_rng(0)
    pm25 = 50 + 20 * np.sin(np.arange(n) / 5) + rng.normal(0, 3, n)
    df = pd.DataFrame({c: 0.0 for c in FEATURES}, index=range(n))
    df["city"] = "Delhi"
    df["dow"] = np.arange(n) % 7
    df["pm25"], df["pm25_lag1"], df["pm25_7d_mean"], df["cams_tomorrow"] = pm25, pm25, pm25, pm25
    df["target"] = pm25 * 1.05
    return df


@pytest.mark.parametrize("kind", m.MODEL_KINDS)
def test_every_model_kind_trains_and_predicts(kind):
    df = feature_table()
    pred = m.predict(m.fit(df, kind), df)
    assert pred.shape == (len(df),) and (pred > 0).all()


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown model kind"):
        m.fit(feature_table(), "magic")


def test_first_model_becomes_champion():
    assert is_new_champion(7.5, None)


def test_champion_only_changes_for_a_better_score():
    assert is_new_champion(7.0, 7.1)
    assert not is_new_champion(7.1, 7.1)
    assert not is_new_champion(7.4, 7.1)
