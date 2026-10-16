"""Tests for cleaning, validation and feature building, on small made-up data."""

import pandas as pd
import pytest

from aqi_forecast import model as m
from aqi_forecast.features import (
    DataValidationError, city_daily, clean_stations, make_features, validate_raw,
)


def stations_for(day, values, city="Delhi", coverage=100.0):
    return pd.DataFrame({
        "city": city, "sensor_id": range(len(values)), "date": pd.Timestamp(day),
        "pm25": values, "coverage_pct": coverage,
    })


def raw_data():
    days = pd.date_range("2026-01-01", "2026-01-10")
    stations = pd.concat(
        [stations_for(d, [50.0 + i, 52.0 + i, 54.0 + i]).assign(sensor_id=[1, 2, 3])
         for i, d in enumerate(days)], ignore_index=True)
    weather = pd.DataFrame({
        "city": "Delhi", "date": days, "temperature_2m_mean": 15.0,
        "relative_humidity_2m_mean": 60, "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0, "wind_direction_10m_dominant": 90,
    })
    cams = pd.DataFrame({"city": "Delhi", "date": days, "cams_pm25": [40.0 + i for i in range(len(days))]})
    return {"stations": stations, "weather": weather, "cams": cams}


# ---------- validation ----------

def test_valid_data_passes():
    validate_raw(raw_data())


def test_missing_column_is_caught():
    raw = raw_data()
    raw["weather"] = raw["weather"].drop(columns="precipitation_sum")
    with pytest.raises(DataValidationError, match="precipitation_sum"):
        validate_raw(raw)


def test_duplicate_rows_are_caught():
    raw = raw_data()
    raw["stations"] = pd.concat([raw["stations"], raw["stations"].head(1)])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_raw(raw)


# ---------- cleaning ----------

def test_impossible_readings_and_partial_days_are_dropped():
    # values seen in the real data: sensor faults of -4210 and 104,000 µg/m³
    s = pd.concat([
        stations_for("2026-01-01", [-4210.0, 0.0, 85.0, 104000.0]),
        stations_for("2026-01-01", [90.0], coverage=50.0),
    ])
    assert clean_stations(s).pm25.tolist() == [85.0]


def test_city_value_is_the_median_and_needs_three_stations():
    s = pd.concat([
        stations_for("2026-01-01", [40.0, 50.0, 900.0]),  # one bad-but-plausible station
        stations_for("2026-01-02", [40.0, 50.0]),  # too few stations
    ])
    daily = city_daily(s)
    assert daily.date.tolist() == [pd.Timestamp("2026-01-01")]
    assert daily.pm25.tolist() == [50.0]


# ---------- features ----------

def test_target_is_tomorrow_and_lag_is_yesterday():
    raw = raw_data()
    df = make_features(city_daily(raw["stations"]), raw["weather"], raw["cams"])
    row = df[df.date == "2026-01-05"].iloc[0]
    assert row.pm25 == 56.0  # median of 54, 56, 58
    assert row.pm25_lag1 == 55.0
    assert row.target == 57.0
    assert row.cams_tomorrow == 45.0  # CAMS value for Jan 6, not Jan 5


def test_no_feature_uses_tomorrows_measurement():
    raw = raw_data()
    df = make_features(city_daily(raw["stations"]), raw["weather"], raw["cams"])
    feature_cols = df.columns.drop(["date", "target", "city"])
    for col in feature_cols:
        assert not (df[col] == df.target).all(), f"{col} looks like a copy of the target"


def test_latest_day_is_kept_only_for_prediction():
    raw = raw_data()
    daily = city_daily(raw["stations"])
    assert make_features(daily, raw["weather"], raw["cams"]).date.max() == pd.Timestamp("2026-01-09")
    latest = make_features(daily, raw["weather"], raw["cams"], keep_unlabelled=True)
    assert latest.date.max() == pd.Timestamp("2026-01-10")


def test_missing_day_leaves_a_gap_not_a_wrong_lag():
    raw = raw_data()
    daily = city_daily(raw["stations"])
    daily = daily[daily.date != "2026-01-04"]
    df = make_features(daily, raw["weather"], raw["cams"])
    assert pd.isna(df.loc[df.date == "2026-01-05", "pm25_lag1"]).all()
    assert "2026-01-03" not in df.date.astype(str).tolist()  # its target day is missing


# ---------- model ----------

def test_model_inputs_have_the_same_columns_for_any_city_mix():
    raw = raw_data()
    df = make_features(city_daily(raw["stations"]), raw["weather"], raw["cams"])
    one_day = df.tail(1)
    assert list(m.model_inputs(df).columns) == list(m.model_inputs(one_day).columns)


def test_model_trains_and_predicts_positive_values():
    raw = raw_data()
    df = make_features(city_daily(raw["stations"]), raw["weather"], raw["cams"])
    model = m.fit(df)
    assert (m.predict(model, df) > 0).all()
