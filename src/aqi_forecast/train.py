"""Stage 3: train the model, check it beats the baselines, save it.

    uv run python -m aqi_forecast.train
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from aqi_forecast import model as m
from aqi_forecast.config import ROOT
from aqi_forecast.features import build_dataset

MODELS_DIR = ROOT / "models"
TEST_START = "2026-01-01"

AQI_BANDS = [0, 30, 60, 90, 120, 250, np.inf]  # India's PM2.5 breakpoints


def category_accuracy(actual, predicted) -> float:
    return float((pd.cut(actual, AQI_BANDS) == pd.cut(predicted, AQI_BANDS)).mean())


def evaluate(test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "mae": round(float(mean_absolute_error(test.target, pred)), 2),
            "category_accuracy": round(category_accuracy(test.target, pred), 3),
        }
        for name, pred in predictions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PM2.5 model.")
    parser.add_argument("--test-start", default=TEST_START, help="first day of the test period")
    args = parser.parse_args()

    df = build_dataset()
    train, test = df[df.date < args.test_start], df[df.date >= args.test_start]
    print(f"train {len(train)} rows, test {len(test)} rows (from {args.test_start})")

    model = m.fit(train)
    metrics = evaluate(test, {
        "persistence": test.pm25,
        "cams": test.cams_tomorrow.fillna(test.pm25),
        "model": m.predict(model, test),
    })
    for name, s in metrics.items():
        print(f"  {name:<12} MAE {s['mae']:6.2f}   category right {s['category_accuracy']:.0%}")

    if metrics["model"]["mae"] >= metrics["persistence"]["mae"]:
        raise SystemExit("Model does not beat persistence. Not saving it.")

    # Retrain on everything for the model we actually use; the test score above is our estimate.
    final = m.fit(df)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(
        {"test_start": args.test_start, "train_rows": len(train), "test_rows": len(test),
         "last_date": str(df.date.max().date()), "metrics": metrics}, indent=2))
    print(f"Saved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
