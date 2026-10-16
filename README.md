# India AQI Forecast

AQI forecasting for Indian cities, built to learn MLOps. We start by predicting next-day PM2.5 and AQI for Delhi, Mumbai and Bengaluru.

## Quick start

```bash
# install uv once: https://docs.astral.sh/uv/getting-started/installation/
git clone https://github.com/dasosis/india-aqi-forecast.git
cd india-aqi-forecast
uv sync          # creates .venv and installs everything from uv.lock
uv run pytest    # checks the setup works
```

## Download the data

```bash
cp .env.example .env    # then paste your free OpenAQ key into .env
uv run python -m aqi_forecast.ingest --days 90
```

This writes three CSVs to `data/raw/`:

| File | What's in it | Source |
| --- | --- | --- |
| `pm25_stations.csv` | Daily PM2.5 measured at every station within 25 km of each city | [OpenAQ](https://openaq.org) (CPCB and other stations) |
| `weather.csv` | Daily temperature, humidity, rain and wind for each city | [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) |
| `cams_pm25.csv` | Daily PM2.5 predicted by the CAMS model, our baseline to beat | [Open-Meteo air quality](https://open-meteo.com/en/docs/air-quality-api) |

## Train and forecast

```bash
uv run python -m aqi_forecast.train     # trains, compares with baselines, saves models/model.joblib
uv run python -m aqi_forecast.predict   # forecasts the next day for each city
uv run pytest                           # checks cleaning, features and validation rules
```

`train` refuses to save a model that doesn't beat "tomorrow equals today".

| Module | What it does |
| --- | --- |
| `ingest.py` | Downloads raw data |
| `features.py` | Validates raw data, cleans it, builds features |
| `model.py` | The ridge model and its inputs |
| `train.py` | Time-based split, scores against baselines, saves model and metrics |
| `predict.py` | Loads the saved model and forecasts tomorrow |

## Run the whole pipeline with DVC

[DVC](https://dvc.org) is like Git for data and models. `dvc.yaml` lists the steps (ingest, train, predict) and what each one depends on. `params.yaml` holds the settings. `dvc.lock` records the exact fingerprint (hash) of every input and output from the last run, and it's committed to Git, so every commit says exactly which data trained which model.

```bash
uv run dvc repro                 # run only the steps whose inputs changed
uv run dvc repro --force ingest  # download fresh data, then rerun what depends on it
uv run dvc push                  # back up data and models to the DVC store (../dvc-storage)
uv run dvc metrics show          # this run's scores
uv run dvc metrics diff main     # compare scores with the main branch
```

Go back to the data and model from any past commit:

```bash
git checkout <commit> && uv run dvc checkout
```

## Results so far

Next-day PM2.5, tested on Jan to Oct 2026 (see [`notebooks/02_first_model.ipynb`](notebooks/02_first_model.ipynb)):

| Forecast | Average error (µg/m³) | AQI category right |
| --- | --- | --- |
| Tomorrow = today (persistence) | 8.4 | 80% |
| CAMS model | 18.8 | 62% |
| **Ridge on log change (ours)** | **7.1** | **81%** |

To rerun it: `uv run jupyter lab notebooks/02_first_model.ipynb`

## Layout

| Folder | What goes there |
| --- | --- |
| `data/raw/` | Data exactly as downloaded (not committed to Git) |
| `data/processed/` | Cleaned data and features (not committed to Git) |
| `notebooks/` | Exploration and experiments |
| `src/aqi_forecast/` | The project's Python code |
| `tests/` | Automated tests |
| `models/` | Trained models (not committed to Git) |

## Roadmap

Each stage adds one working piece and teaches one MLOps idea.

- [x] 0. Project setup: reproducible environment
- [x] 1. Get the data: data ingestion
- [x] 2. First model: baselines and evaluation
- [x] 3. Notebook to pipeline: pipelines and data validation
- [x] 4. Version the data: DVC
- [ ] 5. Track experiments: MLflow
- [ ] 6. Run it every day: orchestration with Prefect
- [ ] 7. Serve predictions: FastAPI and Docker
- [ ] 8. Automate checks: CI/CD with GitHub Actions
- [ ] 9. Watch it in production: monitoring and drift

## Data sources

- [OpenAQ](https://openaq.org) for measured station data (free API key)
- [Open-Meteo](https://open-meteo.com) for weather and CAMS air-quality forecasts
