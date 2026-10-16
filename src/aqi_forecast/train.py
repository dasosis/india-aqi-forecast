"""Train a model, check it beats the baselines, log the run to MLflow, save it.

    uv run python -m aqi_forecast.train --model ridge
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://localhost:5000
"""

import argparse
import hashlib
import json
import os
import subprocess

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error

from aqi_forecast import model as m
from aqi_forecast.config import RAW_DIR, ROOT
from aqi_forecast.features import FEATURES, build_dataset

MODELS_DIR = ROOT / "models"
TEST_START = "2026-01-01"
EXPERIMENT = "next-day-pm25"
REGISTERED_MODEL = "pm25-forecaster"
CHAMPION = "champion"  # registry alias for the best model so far

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


def data_fingerprint() -> str:
    """Short hash of the raw CSVs, so each run records exactly which data it saw."""
    h = hashlib.md5()
    for f in sorted(RAW_DIR.glob("*.csv")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def champion_mae(client: mlflow.MlflowClient) -> float | None:
    """Test MAE of the current champion, or None if there isn't one yet."""
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get("mae_model")


def is_new_champion(new_mae: float, current_mae: float | None) -> bool:
    return current_mae is None or new_mae < current_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PM2.5 model.")
    parser.add_argument("--test-start", default=TEST_START, help="first day of the test period")
    parser.add_argument("--model", default="ridge", choices=m.MODEL_KINDS)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(EXPERIMENT)

    df = build_dataset()
    train, test = df[df.date < args.test_start], df[df.date >= args.test_start]
    print(f"train {len(train)} rows, test {len(test)} rows (from {args.test_start})")

    with mlflow.start_run(run_name=args.model) as run:
        mlflow.log_params({
            "model": args.model, "test_start": args.test_start,
            "train_rows": len(train), "test_rows": len(test),
            "last_date": str(df.date.max().date()), "features": len(FEATURES),
            "data_fingerprint": data_fingerprint(), "git_commit": git_commit(),
        })

        model = m.fit(train, args.model)
        predictions = {
            "persistence": test.pm25,
            "cams": test.cams_tomorrow.fillna(test.pm25),
            "model": m.predict(model, test),
        }
        metrics = evaluate(test, predictions)
        for name, s in metrics.items():
            print(f"  {name:<12} MAE {s['mae']:6.2f}   category right {s['category_accuracy']:.0%}")
            mlflow.log_metrics({f"mae_{name}": s["mae"], f"category_accuracy_{name}": s["category_accuracy"]})
        for city, g in test.assign(pred=predictions["model"]).groupby("city"):
            mlflow.log_metric(f"mae_model_{city}", round(float(mean_absolute_error(g.target, g.pred)), 2))

        if metrics["model"]["mae"] >= metrics["persistence"]["mae"]:
            mlflow.set_tag("outcome", "rejected: does not beat persistence")
            raise SystemExit("Model does not beat persistence. Not saving it.")

        # Retrain on everything for the model we actually use; the test score above is our estimate.
        final = m.fit(df, args.model)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(final, MODELS_DIR / "model.joblib")
        (MODELS_DIR / "metrics.json").write_text(json.dumps(
            {"model": args.model, "test_start": args.test_start, "train_rows": len(train),
             "test_rows": len(test), "last_date": str(df.date.max().date()), "metrics": metrics},
            indent=2))

        # Put the model on the registry "shelf"; label it champion if it's the best so far.
        client = mlflow.MlflowClient()
        current = champion_mae(client)
        logged = mlflow.sklearn.log_model(final, name="model", registered_model_name=REGISTERED_MODEL,
                                          input_example=df[FEATURES].head(3),
                                          skops_trusted_types=m.TRUSTED_TYPES)
        version = logged.registered_model_version
        if is_new_champion(metrics["model"]["mae"], current):
            client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, version)
            mlflow.set_tag("outcome", f"new champion (version {version})")
            print(f"Registered version {version} as the new champion (MAE {metrics['model']['mae']}"
                  + (f", was {current})" if current is not None else ")"))
        else:
            mlflow.set_tag("outcome", f"registered version {version}, champion kept")
            print(f"Registered version {version}; champion stays (its MAE {current} is better)")

    print(f"Saved models/model.joblib and models/metrics.json (MLflow run {run.info.run_id[:8]})")


if __name__ == "__main__":
    main()
