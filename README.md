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
- [ ] 1. Get the data: data ingestion
- [ ] 2. First model: baselines and evaluation
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
- [Kaggle: Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) (CPCB, 2015 to 2020)
