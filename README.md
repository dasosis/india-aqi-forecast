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
- [ ] 3. Notebook to pipeline: pipelines and data validation
- [ ] 4. Version the data: DVC
- [ ] 5. Track experiments: MLflow
- [ ] 6. Run it every day: orchestration with Prefect
- [ ] 7. Serve predictions: FastAPI and Docker
- [ ] 8. Automate checks: CI/CD with GitHub Actions
- [ ] 9. Watch it in production: monitoring and drift

## Data sources

- [OpenAQ](https://openaq.org) for measured station data (free API key)
- [Open-Meteo](https://open-meteo.com) for weather and CAMS air-quality forecasts
