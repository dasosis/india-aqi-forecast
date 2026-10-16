"""Stage 3: turn raw CSVs into a model-ready table.

Same logic as notebooks/02_first_model.ipynb, moved into functions so that
training, prediction and tests all use exactly the same code.
"""

import numpy as np
import pandas as pd

from aqi_forecast.config import RAW_DIR

# Cleaning rules (see the notebook for why)
PM25_MIN, PM25_MAX = 1, 1000  # µg/m³; outside this is a sensor fault
MIN_COVERAGE_PCT = 75  # a station must have run for most of the day
MIN_STATIONS = 3  # a city-day needs at least this many stations

FEATURES = [
    "pm25", "city", "pm25_lag1", "pm25_7d_mean",
    "temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum",
    "wind_speed_10m_max", "wind_dir_sin", "wind_dir_cos",
    "cams_tomorrow", "doy_sin", "doy_cos", "dow",
]


class DataValidationError(ValueError):
    """Raised when input data breaks a rule the pipeline relies on."""


def load_raw(raw_dir=RAW_DIR) -> dict[str, pd.DataFrame]:
    files = {"stations": "pm25_stations.csv", "weather": "weather.csv", "cams": "cams_pm25.csv"}
    return {k: pd.read_csv(raw_dir / f, parse_dates=["date"]) for k, f in files.items()}


def validate_raw(raw: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the raw data is broken in a way cleaning can't fix."""
    expected = {
        "stations": {"city", "sensor_id", "date", "pm25", "coverage_pct"},
        "weather": {"city", "date", "temperature_2m_mean", "relative_humidity_2m_mean",
                    "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"},
        "cams": {"city", "date", "cams_pm25"},
    }
    for name, cols in expected.items():
        missing = cols - set(raw[name].columns)
        if missing:
            raise DataValidationError(f"{name} is missing columns: {sorted(missing)}")
        if raw[name].empty:
            raise DataValidationError(f"{name} has no rows")
    if raw["stations"].duplicated(["sensor_id", "date"]).any():
        raise DataValidationError("stations has duplicate (sensor_id, date) rows")
    for name in ("weather", "cams"):
        if raw[name].duplicated(["city", "date"]).any():
            raise DataValidationError(f"{name} has duplicate (city, date) rows")


def clean_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible readings and partial days."""
    ok = stations.pm25.between(PM25_MIN, PM25_MAX) & (stations.coverage_pct >= MIN_COVERAGE_PCT)
    return stations[ok]


def city_daily(stations: pd.DataFrame) -> pd.DataFrame:
    """One PM2.5 value per city per day: the median across reliable stations."""
    daily = (
        clean_stations(stations)
        .groupby(["city", "date"]).pm25.agg(pm25="median", n_stations="size")
        .reset_index()
    )
    return daily[daily.n_stations >= MIN_STATIONS].drop(columns="n_stations")


def make_features(daily: pd.DataFrame, weather: pd.DataFrame, cams: pd.DataFrame,
                  keep_unlabelled: bool = False) -> pd.DataFrame:
    """Build features for predicting the day after `date`.

    Each row uses only what is known on the evening of `date`.
    `target` is the next day's PM2.5. With keep_unlabelled=True, rows whose
    target isn't known yet (the latest day) are kept, which is what predict needs.
    """
    frames = []
    for city, g in daily.groupby("city"):
        # one row per calendar day, so shift(1) always means "the previous day"
        days = pd.date_range(g.date.min(), g.date.max(), name="date")
        d = g.set_index("date").reindex(days)[["pm25"]]
        d["city"] = city
        d["pm25_lag1"] = d.pm25.shift(1)
        d["pm25_7d_mean"] = d.pm25.rolling(7, min_periods=4).mean()
        d["target"] = d.pm25.shift(-1)
        frames.append(d.reset_index())
    df = pd.concat(frames, ignore_index=True)

    df = df.merge(weather, on=["city", "date"], how="left")
    rad = np.deg2rad(df.pop("wind_direction_10m_dominant"))
    df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(rad), np.cos(rad)

    tomorrow_cams = cams.assign(date=cams.date - pd.Timedelta(days=1))
    df = df.merge(tomorrow_cams.rename(columns={"cams_pm25": "cams_tomorrow"}),
                  on=["city", "date"], how="left")

    tomorrow = df.date + pd.Timedelta(days=1)
    df["doy_sin"] = np.sin(2 * np.pi * tomorrow.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * tomorrow.dt.dayofyear / 365.25)
    df["dow"] = tomorrow.dt.dayofweek

    df = df.dropna(subset=["pm25"])
    if not keep_unlabelled:
        df = df.dropna(subset=["target"])
    return df.reset_index(drop=True)


def build_dataset(raw_dir=RAW_DIR, keep_unlabelled: bool = False) -> pd.DataFrame:
    raw = load_raw(raw_dir)
    validate_raw(raw)
    return make_features(city_daily(raw["stations"]), raw["weather"], raw["cams"], keep_unlabelled)
