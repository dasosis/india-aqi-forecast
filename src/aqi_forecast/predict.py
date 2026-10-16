"""Stage 3: forecast tomorrow's PM2.5 for each city with the saved model.

    uv run python -m aqi_forecast.predict
"""

import joblib
import pandas as pd

from aqi_forecast import model as m
from aqi_forecast.features import build_dataset
from aqi_forecast.train import AQI_BANDS, MODELS_DIR

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def forecast(model, df: pd.DataFrame) -> pd.DataFrame:
    """Use each city's most recent day to forecast the day after it."""
    latest = df.sort_values("date").groupby("city").tail(1)
    pm25 = m.predict(model, latest)
    return pd.DataFrame({
        "city": latest.city.values,
        "forecast_for": (latest.date + pd.Timedelta(days=1)).dt.date.values,
        "pm25": pm25.round(1),
        "category": pd.cut(pm25, AQI_BANDS, labels=AQI_LABELS),
    })


def main() -> None:
    model = joblib.load(MODELS_DIR / "model.joblib")
    result = forecast(model, build_dataset(keep_unlabelled=True))
    print(result.to_string(index=False))
    result.to_csv(MODELS_DIR / "latest_forecast.csv", index=False)


if __name__ == "__main__":
    main()
